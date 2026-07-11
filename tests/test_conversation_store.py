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