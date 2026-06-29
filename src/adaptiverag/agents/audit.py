"""Append-only, tamper-evident audit log for every tool call.

Each line is one JSON object (JSON Lines format). Entries are linked in a
hash chain: every entry stores the hash of the entry before it, so editing
any past entry breaks the chain and verify() can detect it.
See CLAUDE.md Block 3.1 for the why.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from pathlib import Path
from datetime import datetime, timezone

# The first entry has no predecessor, so we chain it off this fixed value.
# 64 hex chars = the width of a SHA-256 digest, so it "looks like" a real hash.
GENESIS_HASH = "0" * 64

_MAX_RESULT_CHARS = 2000


def _canonical(payload: dict) -> str:
    """Serialize a record to one deterministic string.

    sort_keys → key order never changes the output bytes, so the same
    logical record always hashes identically. default=str → anything not
    natively JSON-serializable (a datetime, a custom object in a result)
    is coerced to its string form instead of raising.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

def _truncate(result) -> str:
    """Coerce a tool result to a bounded string for the log.

    Two jobs: keep the log readable/bounded, and guarantee the stored value
    is JSON-native so the canonical bytes survive a write→read round-trip —
    critical, because verify() must recompute the *exact same* tag later.
    """
    text = result if isinstance(result, str) else repr(result)
    if len(text) > _MAX_RESULT_CHARS:
        text = text[:_MAX_RESULT_CHARS] + f"...[truncated {len(text) - _MAX_RESULT_CHARS} chars]"
    return text


def _compute_tag(payload: dict, key: bytes) -> str:
    """Keyed HMAC-SHA256 fingerprint (hex) of a canonicalized record.

    HMAC = a hash mixed with a secret key. Without the key you cannot
    produce a valid tag, so an attacker who rewrites the whole file still
    can't forge matching fingerprints. The payload still includes prev_hash,
    so the chain property holds *on top of* the key property.
    """
    return hmac.new(key, _canonical(payload).encode("utf-8"), hashlib.sha256).hexdigest()

class AuditLog:
    """Append-only, HMAC-chained log of tool calls.

    One instance wraps one log file. Build it once at startup and hand it
    to the ToolRegistry; every registry.call() appends exactly one entry.
    """

    def __init__(self, path: str | Path, key: str | None = None):
        # The key turns tamper-EVIDENT into tamper-evident-AND-unforgeable.
        # Keep it OUT of the repo: read from env, fail loud if missing.
        # Production: source this from Azure Key Vault via Managed Identity.
        key = key or os.getenv("AUDIT_HMAC_KEY")
        if not key:
            raise ValueError(
                "Audit log requires an HMAC key (pass key= or set AUDIT_HMAC_KEY)."
            )
        self._key = key.encode("utf-8")

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # serialize concurrent appends

        # Recover the chain head so a restart CONTINUES the same chain
        # instead of starting fresh from GENESIS.
        self._last_hash, self._seq = self._recover_state()

    def _recover_state(self) -> tuple[str, int]:
        """Read the file's last line once → (chain head hash, next seq).

        Fresh/empty file → (GENESIS_HASH, 0). Recovery only locates where to
        attach the next link; verify() is what proves the file is intact.
        """
        if not self._path.exists():
            return GENESIS_HASH, 0
        last_line = None
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line is None:
            return GENESIS_HASH, 0
        last = json.loads(last_line)
        return last["entry_hash"], last["seq"] + 1
    
    def append(self, tool: str, args: dict, result, ok: bool) -> dict:
        """Record one tool call as the next link in the chain.

        Builds the payload, HMAC-tags it (over contents *including* prev_hash),
        writes one JSON line, advances the chain head. Returns the entry.
        """
        with self._lock:
            payload = {
                "seq": self._seq,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool": tool,
                "args": args,
                "result": _truncate(result),
                "ok": ok,
                "prev_hash": self._last_hash,
            }
            tag = _compute_tag(payload, self._key)
            entry = {**payload, "entry_hash": tag}

            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

            self._last_hash = tag
            self._seq += 1
            return entry
        
    def verify(self) -> tuple[bool, str | None]:
        """Re-walk the chain with the key; report the first broken link.

        Returns (True, None) if intact, else (False, reason). Catches edited
        contents (tag won't recompute), a forged/snipped prev_hash (link won't
        match), and inserted/reordered/deleted lines.
        """
        if not self._path.exists():
            return True, None  # no file = empty = trivially intact

        prev = GENESIS_HASH
        with self._path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f):
                if not line.strip():
                    continue
                entry = json.loads(line)
                stored_tag = entry.pop("entry_hash")  # leaves the original payload

                # (a) does this entry link to the one before it?
                if entry.get("prev_hash") != prev:
                    return False, f"broken link at seq {entry.get('seq')} (line {lineno})"

                # (b) recompute the tag over the payload; constant-time compare
                if not hmac.compare_digest(_compute_tag(entry, self._key), stored_tag):
                    return False, f"bad tag at seq {entry.get('seq')} (line {lineno})"

                prev = stored_tag

        return True, None