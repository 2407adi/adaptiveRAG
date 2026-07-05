
# tests/test_memory.py
"""Consolidated tests for Block 3.3 — conversation memory (buffer + vector).

Offline by design (same spirit as test_tools.py): the LLM is never called and
the embedder is a deterministic bag-of-keywords fake, so recall ordering is
predictable without a model download. The logic tests use an in-memory fake
VectorStore; the ONE persistence test uses a real ChromaStore on a temp dir,
because "survives a restart" is a real-backend behaviour worth exercising.
"""

import math

import pytest

from adaptiverag.agents.memory import (
    Turn, BufferMemory, VectorMemory, ConversationMemory,
)
from adaptiverag.retrieve.vector_store import StoredChunk, SearchResult


# ── A deterministic, offline embedder ────────────────────────────────────────
class _FakeEmbedder:
    """Bag-of-keywords stamp machine: dimension i = count of keyword i, plus one
    shared constant dim so an all-zero text still has a non-zero norm. Texts that
    share words get similar vectors → meaningful (and predictable) cosine recall."""

    _VOCAB = ("funding", "series", "headcount", "employees", "revenue",
              "pricing", "weather", "hello")

    def embed(self, text: str) -> list[float]:
        t = text.lower()
        v = [float(t.count(w)) for w in self._VOCAB]
        v.append(0.1)                       # shared dim → no zero-norm vectors
        return v

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return len(self._VOCAB) + 1


# ── An in-memory VectorStore stand-in (only what VectorMemory needs) ──────────
class _FakeStore:
    """Keeps StoredChunks in a list, ranks by cosine on .search — mirrors
    ChromaStore's contract (score = similarity, higher is better)."""

    def __init__(self):
        self._chunks: list[StoredChunk] = []

    def add(self, chunks: list[StoredChunk]) -> None:
        self._chunks.extend(chunks)

    @staticmethod
    def _cos(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query_vector, k: int = 5) -> list[SearchResult]:
        scored = [
            SearchResult(chunk_id=c.id, text=c.text,
                         score=self._cos(query_vector, c.embedding),
                         metadata=c.metadata)
            for c in self._chunks
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._chunks)


# ── Short-term: the desk clipboard (BufferMemory) ────────────────────────────
def test_buffer_evicts_oldest_when_full():
    """The sliding window: a K+1th page pushes the oldest off the front."""
    buf = BufferMemory(max_turns=4)
    for i in range(6):                       # add 6 into a board that holds 4
        buf.add("user", f"msg{i}")
    contents = [t.content for t in buf.get()]
    assert contents == ["msg2", "msg3", "msg4", "msg5"]   # msg0 + msg1 slid off
    assert len(buf) == 4


def test_buffer_as_prompt_formats_roles():
    """as_prompt() renders a readable 'User:/Assistant:' transcript."""
    buf = BufferMemory(max_turns=10)
    buf.add("user", "hello there")
    buf.add("assistant", "hi, how can I help")
    text = buf.as_prompt()
    assert "User: hello there" in text
    assert "Assistant: hi, how can I help" in text


def test_buffer_clear_empties_board():
    """clear() wipes the board (and an empty board renders as '')."""
    buf = BufferMemory(max_turns=10)
    buf.add("user", "something")
    buf.clear()
    assert len(buf) == 0
    assert buf.as_prompt() == ""


# ── Long-term: the archive (VectorMemory) ────────────────────────────────────
def test_vector_recall_returns_most_relevant():
    """recall() pulls the past turn closest in meaning to the query."""
    vec = VectorMemory(_FakeEmbedder(), _FakeStore())
    vec.add("user", "tell me about the funding and series A")
    vec.add("user", "what is the headcount and employees count")
    hits = vec.recall("funding series", k=1)
    assert len(hits) == 1
    assert "funding" in hits[0].text.lower()


def test_vector_recall_role_filter_returns_only_user_turns():
    """'What did I ask earlier?' → recall(role='user') returns only questions."""
    vec = VectorMemory(_FakeEmbedder(), _FakeStore())
    vec.add("user", "a question about revenue")
    vec.add("assistant", "here is the revenue answer")
    hits = vec.recall("revenue", k=5, role="user")
    assert hits                                           # at least one survivor
    assert all(h.metadata.get("role") == "user" for h in hits)
    assert any("question" in h.text.lower() for h in hits)


def test_vector_recall_scopes_by_conversation_id():
    """conversation_id keeps one conversation's cards from bleeding into another."""
    vec = VectorMemory(_FakeEmbedder(), _FakeStore())
    vec.add("user", "funding talk in conversation A", conversation_id="A")
    vec.add("user", "funding talk in conversation B", conversation_id="B")
    hits = vec.recall("funding", k=5, conversation_id="A")
    assert hits
    assert all(h.metadata.get("conversation_id") == "A" for h in hits)


# ── The clerk (ConversationMemory) ───────────────────────────────────────────
def test_conversation_add_turn_writes_both_tiers():
    """One add_turn() lands in BOTH the buffer and the vector archive."""
    buf, vec = BufferMemory(max_turns=10), VectorMemory(_FakeEmbedder(), _FakeStore())
    cm = ConversationMemory(buf, vec, recall_k=3, recall_score_threshold=0.0,
                            conversation_id="c1")
    cm.add_turn("user", "hello about funding")
    assert len(buf) == 1                                  # clipboard got it
    assert any("funding" in h.text.lower()               # archive got it too
               for h in vec.recall("funding", k=5))


def test_build_context_combines_recency_and_relevance():
    """build_context = relevant older card (from archive) + recent turns (buffer),
    with the desk-duplicate dropped so nothing is listed twice."""
    buf = BufferMemory(max_turns=2)                      # tiny board → old turns evict
    vec = VectorMemory(_FakeEmbedder(), _FakeStore())
    cm = ConversationMemory(buf, vec, recall_k=3, recall_score_threshold=0.1,
                            conversation_id="c1")
    cm.add_turn("user", "earlier I asked about pricing")  # evicted from buffer, kept in archive
    cm.add_turn("assistant", "noted")
    cm.add_turn("user", "now what about the weather")     # buffer now holds only the last 2

    ctx = cm.build_context("pricing")
    # relevance: the evicted 'pricing' turn is recalled from the archive
    assert "Relevant earlier" in ctx and "pricing" in ctx.lower()
    # recency: the latest buffer turn shows up
    assert "Recent conversation" in ctx and "weather" in ctx.lower()
    # dedupe: a turn still on the desk isn't ALSO under 'Relevant earlier'
    relevant_section = ctx.split("Recent conversation")[0].lower()
    assert "weather" not in relevant_section


def test_build_context_empty_on_first_turn():
    """Nothing said yet → empty briefing (so the caller can skip the section)."""
    cm = ConversationMemory(BufferMemory(), VectorMemory(_FakeEmbedder(), _FakeStore()),
                            conversation_id="c1")
    assert cm.build_context("anything at all") == ""


# ── Persistence across sessions (real ChromaStore) ───────────────────────────
def test_long_term_memory_persists_across_sessions(tmp_path):
    """Write in 'session 1', drop the objects, re-open a fresh store at the SAME
    path in 'session 2' — the memory is still recallable (survived the restart)."""
    from adaptiverag.retrieve.vector_store import ChromaStore

    path = str(tmp_path / "mem_store")

    # session 1: file a memory, then "go home"
    store1 = ChromaStore(collection_name="mem", persist_directory=path)
    VectorMemory(_FakeEmbedder(), store1).add(
        "user", "we agreed the pricing tier was premium"
    )
    del store1

    # session 2: brand-new store object at the same path (simulates a restart)
    store2 = ChromaStore(collection_name="mem", persist_directory=path)
    hits = VectorMemory(_FakeEmbedder(), store2).recall("pricing", k=1)
    assert hits and "pricing" in hits[0].text.lower()