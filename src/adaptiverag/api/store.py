"""The ledger cabinet: conversations that survive the night (server restarts)."""
from __future__ import annotations

import sqlite3                                  # Python's built-in librarian for SQLite files
from contextlib import contextmanager           # turns _drawer() into a `with`-able ritual
from pathlib import Path
from datetime import datetime, timezone        # add to the imports at the top


def _now() -> str:
    """One clock for the whole cabinet — UTC ISO strings, same style as memory.py/audit.py."""
    return datetime.now(timezone.utc).isoformat()


# The printed ruled columns (the SCHEMA). Two tables: folder tabs + pages.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (      -- the folder tabs
    id         TEXT PRIMARY KEY,                -- the ticket number (uuid); unique, finds a folder instantly
    title      TEXT,                            -- the tab label; NULL (empty) until the auto-titler writes one
    created_at TEXT NOT NULL                    -- when the folder was opened (ISO timestamp, same style as memory.py)
);
CREATE TABLE IF NOT EXISTS turns (              -- the pages inside the folders
    id              INTEGER PRIMARY KEY,        -- page number; SQLite hands out 1, 2, 3... automatically
    conversation_id TEXT NOT NULL               -- the folder stamp (FOREIGN KEY: which folder owns this page)
                    REFERENCES conversations(id),
    role            TEXT NOT NULL,              -- who spoke: "user" / "assistant"
    content         TEXT NOT NULL,              -- what they said
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_conversation   -- a card catalog: "all pages of folder X"
    ON turns(conversation_id);                      -- without it SQLite reads EVERY page to find one folder's
"""


class ConversationStore:
    """The counter clerk for the cabinet. routes.py talks ONLY to these methods."""

    def __init__(self, db_path: str | Path):
        self._path = str(db_path)                        # where the ledger file lives
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)  # make sure the room exists
        with self._drawer() as conn:                     # first open: print the ruled columns
            conn.executescript(_SCHEMA)                  # executescript = run several statements at once

    @contextmanager
    def _drawer(self):
        """The drawer ritual: open → errand → commit → close. One trip per errand,
        so FastAPI's parallel workers (threads) never share an open drawer."""
        conn = sqlite3.connect(self._path)               # open the drawer
        conn.row_factory = sqlite3.Row                   # rows come back name-addressable: row["title"]
        try:
            yield conn                                   # ...the caller's errand happens here...
            conn.commit()                                # press the pencil marks in permanently
        finally:
            conn.close()                                 # close the drawer even if the errand blew up

    # ---- write errands (inside ConversationStore) ----

    def create(self, conversation_id: str, title: str | None = None) -> None:
        """Open a folder for this ticket. Already exists? Do nothing (idempotent)."""
        with self._drawer() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
                (conversation_id, title, _now()),     # values ride separately — the form boxes
            )

    def append_turn(self, conversation_id: str, role: str, content: str) -> None:
        """Add one page to a folder. New ticket? The tab is made first, so a page
        can never be filed into a folder that doesn't exist (the FOREIGN KEY would refuse)."""
        self.create(conversation_id)                  # harmless if the folder is already there
        with self._drawer() as conn:
            conn.execute(
                "INSERT INTO turns (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, _now()),
            )                                         # note: no `id` — SQLite stamps the page number itself

    def set_title(self, conversation_id: str, title: str) -> None:
        """Write the label on ONE folder's tab (the auto-titler calls this once per chat)."""
        with self._drawer() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),             # WHERE picks the folder; without it = relabel ALL
            )

    # ---- read errands (inside ConversationStore) ----

    def exists(self, conversation_id: str) -> bool:
        """Is there a folder with this ticket? (routes.py's 404 check.)"""
        with self._drawer() as conn:
            row = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?",   # SELECT 1: we only care IF a row exists
                (conversation_id,),                           # note the comma — a 1-item tuple
            ).fetchone()                                      # first matching row, or None
        return row is not None

    def get_turns(self, conversation_id: str) -> list[dict]:
        """Every page in this folder, in page-number order. Shaped exactly like the
        old whiteboard entries ({"role", "content"}) so the API response doesn't change."""
        with self._drawer() as conn:
            rows = conn.execute(
                "SELECT role, content FROM turns WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),                           # WHERE = riffle the pile for this stamp
            ).fetchall()                                      # all matches, as a list of Rows
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def get_title(self, conversation_id: str) -> str | None:
        """Read the tab label. None = folder missing OR label not written yet —
        either way the auto-titler knows it still has work to do."""
        with self._drawer() as conn:
            row = conn.execute(
                "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row["title"] if row else None

    def list_conversations(self) -> list[dict]:
        """The sidebar list: every folder tab + its page count, newest folder first."""
        with self._drawer() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, COUNT(t.id) AS turns     -- tab + counted pages
                FROM conversations c
                LEFT JOIN turns t ON t.conversation_id = c.id  -- line pages up next to their tab
                GROUP BY c.id                                  -- fold back to one row per folder
                ORDER BY c.created_at DESC                     -- newest chat on top (sidebar order)
                """
            ).fetchall()
        return [{"id": r["id"], "title": r["title"], "turns": r["turns"]} for r in rows]