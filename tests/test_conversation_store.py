"""Block 4.3a drills: the ledger cabinet (SQLite ConversationStore)."""
import pytest

from adaptiverag.api.store import ConversationStore


@pytest.fixture
def db_path(tmp_path):
    """A throwaway address for the ledger file — fresh folder per test."""
    return tmp_path / "conversations.db"


@pytest.fixture
def store(db_path):
    """A freshly built cabinet on that address."""
    return ConversationStore(db_path)


def test_round_trip(store):
    """File two pages, read them back — shape identical to the old whiteboard."""
    store.append_turn("abc", "user", "What is a reranker?")
    store.append_turn("abc", "assistant", "A second-pass scorer.")
    assert store.get_turns("abc") == [
        {"role": "user", "content": "What is a reranker?"},
        {"role": "assistant", "content": "A second-pass scorer."},
    ]


def test_turn_order_preserved(store):
    """Pages come back in page-number order, not pile order."""
    for i in range(5):
        store.append_turn("abc", "user", f"message {i}")
    contents = [t["content"] for t in store.get_turns("abc")]
    assert contents == [f"message {i}" for i in range(5)]


def test_exists(store):
    store.create("abc")
    assert store.exists("abc")
    assert not store.exists("nope")          # the 404 check


def test_create_is_idempotent(store):
    """Second create with the same ticket is silently ignored — title survives."""
    store.create("abc", title="First label")
    store.create("abc", title="Impostor label")   # INSERT OR IGNORE → no effect
    assert store.get_title("abc") == "First label"


def test_title_lifecycle(store):
    """Blank tab → labeled tab. None also for unknown folders."""
    store.create("abc")
    assert store.get_title("abc") is None    # folder exists, tab still blank
    assert store.get_title("ghost") is None  # folder doesn't exist at all
    store.set_title("abc", "Reranker basics")
    assert store.get_title("abc") == "Reranker basics"


def test_folders_are_isolated(store):
    """Pages carry their folder stamp — chat B never sees chat A's pages."""
    store.append_turn("chat-A", "user", "secret ZANZIBAR-7")
    store.append_turn("chat-B", "user", "hello")
    assert "ZANZIBAR" not in str(store.get_turns("chat-B"))
    assert len(store.get_turns("chat-A")) == 1


def test_list_conversations_counts_and_order(store):
    """Sidebar list: page counts right, empty folders kept (LEFT JOIN)."""
    store.append_turn("old", "user", "q")
    store.append_turn("old", "assistant", "a")
    store.create("empty")                    # zero pages — must still be listed
    listing = {c["id"]: c["turns"] for c in store.list_conversations()}
    assert listing == {"old": 2, "empty": 0}


def test_restart_survival(db_path):
    """THE block test: new store instance on the same file = uvicorn restart."""
    first = ConversationStore(db_path)
    first.append_turn("abc", "user", "survive this")
    first.set_title("abc", "Survival test")
    del first                                          # the server dies

    reborn = ConversationStore(db_path)                # the server comes back
    assert reborn.get_turns("abc") == [{"role": "user", "content": "survive this"}]
    assert reborn.get_title("abc") == "Survival test"  # schema re-run didn't wipe anything


def test_unknown_conversation_reads_are_calm(store):
    """Reading a folder that never existed: empty list / None, never an exception."""
    assert store.get_turns("ghost") == []
    assert store.get_title("ghost") is None

# ---- browser-local tenancy (owner column, post-4.3b) ----

def test_owner_filtered_listing(store):
    """Each browser sees ONLY its own drawer; no owner filter = whole cabinet."""
    store.append_turn("a1", "user", "hi", owner="browser-A")
    store.append_turn("b1", "user", "yo", owner="browser-B")
    assert [c["id"] for c in store.list_conversations(owner="browser-A")] == ["a1"]
    assert [c["id"] for c in store.list_conversations(owner="browser-B")] == ["b1"]
    assert len(store.list_conversations()) == 2          # unfiltered legacy view

def test_foreign_folder_looks_missing(store):
    """exists() with an owner: a stranger's folder answers 'no' (routes 404 it)."""
    store.append_turn("a1", "user", "hi", owner="browser-A")
    assert store.exists("a1", owner="browser-A") is True
    assert store.exists("a1", owner="browser-B") is False
    assert store.exists("a1") is True                    # ownerless check unchanged

def test_first_visitor_owns_the_folder(store):
    """INSERT OR IGNORE: a second visitor appending to a known id can't steal it."""
    store.append_turn("a1", "user", "mine", owner="browser-A")
    store.append_turn("a1", "assistant", "reply", owner="browser-B")  # same folder id
    assert store.exists("a1", owner="browser-A") is True # stamp unchanged
    assert store.exists("a1", owner="browser-B") is False

def test_legacy_null_owner_hidden_from_filtered_listing(store):
    """Pre-tenancy folders (owner NULL) vanish from every filtered sidebar."""
    store.append_turn("old", "user", "ancient demo chat")            # no owner
    assert store.list_conversations(owner="browser-A") == []
    assert [c["id"] for c in store.list_conversations()] == ["old"]

def test_delete_owned_shreds_only_that_drawer(store):
    store.append_turn("a1", "user", "hi", owner="browser-A")
    store.append_turn("a2", "user", "hey", owner="browser-A")
    store.append_turn("b1", "user", "yo", owner="browser-B")
    assert store.delete_owned("browser-A") == 2
    assert store.list_conversations(owner="browser-A") == []
    assert [c["id"] for c in store.list_conversations(owner="browser-B")] == ["b1"]
    assert store.get_turns("b1") != []                   # B's pages untouched

def test_owner_migration_on_old_cabinet(db_path, monkeypatch):
    """A DB created BEFORE the owner column gets ALTERed on next boot, not wiped."""
    import sqlite3
    conn = sqlite3.connect(db_path)                      # print the OLD ruled columns by hand
    conn.executescript("""
        CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT, created_at TEXT NOT NULL);
        CREATE TABLE turns (id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL
            REFERENCES conversations(id), role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL);
        INSERT INTO conversations VALUES ('legacy', 'Old chat', '2026-01-01');
    """)
    conn.commit()
    conn.close()

    migrated = ConversationStore(db_path)                # boot on the old file
    assert migrated.get_title("legacy") == "Old chat"    # nothing lost
    assert migrated.list_conversations(owner="anyone") == []      # NULL owner stays hidden
    migrated.append_turn("fresh", "user", "hi", owner="anyone")   # new column usable
    assert [c["id"] for c in migrated.list_conversations(owner="anyone")] == ["fresh"]
