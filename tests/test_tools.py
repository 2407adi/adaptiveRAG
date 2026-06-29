# tests/test_tools.py
"""Consolidated tests for Block 3.1 — tool registry, sandbox, and audit log.

Offline by design: the LLM is never called, and web_search runs with no Tavily
key so it degrades instead of hitting the network. The sandbox drills DO spawn
real short-lived processes — that's the point: to prove the blocking is real.
"""

import pytest

from adaptiverag.agents.tools import build_default_registry
from adaptiverag.agents.sandbox import run_python as sandbox_run   # the ENGINE, for direct sandbox drills
from adaptiverag.agents.audit import AuditLog
from adaptiverag.config import ToolsConfig, SandboxConfig, TavilyConfig, AuditConfig

import json


# ── A fake archive: anything with .retrieve(query) → list of results works (duck typing) ──
class _FakeResult:
    """Stand-in for a SearchResult — the librarian only reads .text and .metadata."""
    def __init__(self, text, source):
        self.text = text
        self.metadata = {"source": source}


class _FakeRagChain:
    """A pretend document archive that always returns two canned passages."""
    def retrieve(self, query):
        return [
            _FakeResult("Q3 revenue was $4.2M.", "/docs/financials.pdf"),
            _FakeResult("Revenue grew 30% YoY.", "/docs/financials.pdf"),
        ]


@pytest.fixture
def registry(tmp_path):
    """A fully-staffed agency whose logbook is a throwaway file in a temp folder.

    tmp_path is a pytest built-in: a fresh empty directory unique to each test,
    auto-deleted afterward — so tests never touch your real data/audit/ logbook.
    """
    cfg = ToolsConfig(
        sandbox=SandboxConfig(timeout=2.0, cpu_seconds=1, max_memory_mb=256),  # short caps so the loop drill is quick
        tavily=TavilyConfig(enabled=True, max_results=3),
        audit=AuditConfig(path=str(tmp_path / "audit.jsonl")),                 # logbook in the temp folder
    )
    return build_default_registry(
        _FakeRagChain(),          # the fake archive bound into the librarian
        cfg,
        hmac_key="test-key",      # any key works for the drill; real one lives in .env
        tavily_api_key=None,      # no web account → researcher degrades, stays offline
    )

def test_exactly_three_tools_with_schemas(registry):
    """Drill 1: count the staff and read their business cards."""
    cards = registry.list_tools()

    assert len(cards) == 3                                    # exactly three contractors — no more, no fewer
    names = {c["name"] for c in cards}
    assert names == {"run_python", "search_documents", "web_search"}

    for card in cards:                                       # every card has all three sections filled in
        assert card["name"]
        assert card["description"]                            # bio — came from the function's docstring
        assert card["parameters"]["type"] == "object"         # the JSON-Schema "what I need" shape

    by_name = {c["name"]: c for c in cards}
    # the required-args on each card match what the boss must actually supply:
    assert by_name["run_python"]["parameters"]["required"] == ["code"]
    assert by_name["search_documents"]["parameters"]["required"] == ["query"]
    assert by_name["web_search"]["parameters"]["required"] == ["query"]
    # and the baked-in secrets NEVER leaked onto a card:
    assert "rag_chain" not in by_name["search_documents"]["parameters"]["properties"]
    assert "api_key" not in by_name["web_search"]["parameters"]["properties"]


def test_run_python_returns_four(registry):
    """Drill 2: hand the accountant a simple sum; expect 4 back (as text)."""
    result = registry.call("run_python", {"code": "2 + 2"})
    assert result == "4"                                      # the value of the last expression, rendered as text


def test_run_python_captures_print_and_value(registry):
    """Both buckets come back: the out-tray (print) AND the final value."""
    result = registry.call("run_python", {"code": "print('hi')\n21 * 2"})
    assert "hi" in result                                    # captured from the out-tray (stdout)
    assert "42" in result                                    # the last expression's value


def test_search_documents_returns_passages(registry):
    """Drill: the librarian fetches the canned passages, labeled by source."""
    result = registry.call("search_documents", {"query": "revenue"})
    assert "Q3 revenue was $4.2M." in result                 # the fake archive's passage text
    assert "financials.pdf" in result                        # just the filename, not the full /docs/... path


def test_sandbox_blocks_file_access():
    """Saboteur tries to read a file — the room has no `open`, so it's refused."""
    outcome = sandbox_run("open('/etc/passwd').read()")
    assert outcome.ok is False                          # blocked, not executed
    assert outcome.error is not None and "NameError" in outcome.error                # `open` simply isn't in the room's rulebook


def test_sandbox_blocks_network_import():
    """Saboteur tries `import socket` — no __import__ in the room, so imports fail."""
    outcome = sandbox_run("import socket")
    assert outcome.ok is False
    assert outcome.error is not None and "ImportError" in outcome.error              # the room can't reach the network because it can't import


def test_sandbox_times_out_on_infinite_loop():
    """Saboteur loops forever — the supervisor evicts the room at the wall-clock limit."""
    outcome = sandbox_run("while True:\n    pass", timeout=0.5)   # short clock so the drill is quick
    assert outcome.ok is False
    assert outcome.error is not None and "Timed out" in outcome.error                # the stopwatch caught it, not a CPU/memory cap


def test_hostile_code_through_tool_does_not_crash(registry):
    """Through the front desk, a blocked script comes back as a calm note — no meltdown."""
    result = registry.call("run_python", {"code": "open('secrets').read()"})
    assert result.startswith("Error:")                  # the tool flattened the SandboxResult into a plain note


def test_every_call_is_logged(registry, tmp_path):
    """Drill: two jobs through the desk → two stamped pages in the logbook."""
    registry.call("run_python", {"code": "1 + 1"})
    registry.call("search_documents", {"query": "revenue"})

    log_file = tmp_path / "audit.jsonl"
    lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]

    assert len(lines) == 2                                   # one page per call, nothing skipped
    assert lines[0]["tool"] == "run_python"
    assert lines[0]["args"] == {"code": "1 + 1"}             # the exact arguments were recorded
    assert lines[1]["tool"] == "search_documents"
    assert "prev_hash" in lines[0] and "entry_hash" in lines[0]   # each page carries its chain links


def test_audit_verify_passes_on_honest_log(registry, tmp_path):
    """An untouched logbook passes inspection."""
    registry.call("run_python", {"code": "2 + 2"})
    registry.call("web_search", {"query": "x"})

    auditor = AuditLog(str(tmp_path / "audit.jsonl"), key="test-key")   # same stamp key as the fixture
    ok, reason = auditor.verify()
    assert ok is True
    assert reason is None


def test_audit_tamper_is_detected(registry, tmp_path):
    """Forge a past page → the chain's stamps no longer line up → caught."""
    registry.call("run_python", {"code": "2 + 2"})
    registry.call("run_python", {"code": "3 + 3"})

    log_file = tmp_path / "audit.jsonl"
    lines = log_file.read_text().splitlines()

    # The saboteur edits the FIRST entry's recorded result, but can't re-stamp
    # it (no key), so they leave entry_hash as-is.
    entry = json.loads(lines[0])
    entry["result"] = "999"                                 # quietly change what was recorded
    lines[0] = json.dumps(entry)
    log_file.write_text("\n".join(lines) + "\n")

    auditor = AuditLog(str(log_file), key="test-key")
    ok, reason = auditor.verify()
    assert ok is False                                      # the forgery is detected
    assert reason is not None                               # …and verify() points to where