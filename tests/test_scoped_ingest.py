"""Block 4.2b — chat-scoped ingestion & retrieval.

Isolation drills, all offline (no LLM, no network):
a book stamped for chat A's locker must NEVER surface in chat B's searches;
the public shelf ("shared") stays visible to everyone; callers that pass no
guest list (eval suite, old code) still see everything.
"""

from types import SimpleNamespace

import pytest

from adaptiverag.retrieve.vector_store import ChromaStore, FAISSStore, StoredChunk
from adaptiverag.retrieve.hybrid import BM25Retriever, HybridRetriever
from adaptiverag.scope import SHARED_SCOPE, chat_scope, scopes_for, current_scopes
from adaptiverag.agents.tools import make_search_documents


# ---------------------------------------------------------------- fixtures

DIM = 8


def _vec(seed: str) -> list[float]:
    """Deterministic tiny embedding — similarity is irrelevant to these
    drills (we always fetch more than the corpus holds and assert on
    MEMBERSHIP), it just has to be stable."""
    return [(hash((seed, i)) % 100) / 100.0 for i in range(DIM)]


def _chunk(cid: str, text: str, scope: str | None) -> StoredChunk:
    meta = {"source": f"{cid}.md", "chunk_index": 0}
    if scope is not None:            # scope=None simulates a pre-4.2b "legacy" chunk (no stamp)
        meta["scope"] = scope
    return StoredChunk(id=cid, text=text, embedding=_vec(cid), metadata=meta)


@pytest.fixture()
def corpus() -> list[StoredChunk]:
    """One book per shelf: public, locker A, locker B, one legacy unstamped
    book from before the clerk owned an ink stamp — plus two filler books so
    BM25's rarity statistics stay positive (a word in HALF the corpus gets
    an IDF of exactly zero in BM25Okapi and would never score)."""
    return [
        _chunk("pub", "solstice public handbook uptime", SHARED_SCOPE),
        _chunk("a1", "alpha secret armadillo report", chat_scope("A")),
        _chunk("b1", "bravo secret barracuda report", chat_scope("B")),
        _chunk("old", "legacy unstamped parchment", None),
        _chunk("f1", "orchid nebula travel guide", SHARED_SCOPE),
        _chunk("f2", "quantum pastry cookbook volume", SHARED_SCOPE),
    ]

ALL_IDS = {"pub", "a1", "b1", "old", "f1", "f2"}
SHELF_PLUS_A = {"pub", "f1", "f2", "a1"}         # everything chat A may see


class FakeEmbedder:
    """Stable text→vector mapping; enough for hybrid's dense arm."""

    def embed(self, text: str) -> list[float]:
        return _vec(text)


def _ids(results) -> set[str]:
    return {r.chunk_id for r in results}


# ------------------------------------------------- the dense librarian (Chroma)

class TestChromaScopeFilter:
    @pytest.fixture()
    def store(self, corpus) -> ChromaStore:
        s = ChromaStore(collection_name="scoped_test")   # ephemeral, in-memory
        s.add(corpus)
        return s

    def test_chat_a_sees_shared_plus_own_locker_only(self, store):
        hits = _ids(store.search(_vec("q"), k=10, scopes=scopes_for("A")))
        assert hits == SHELF_PLUS_A      # shelf + own locker; B + legacy invisible

    def test_no_guest_list_means_everything(self, store):
        hits = _ids(store.search(_vec("q"), k=10, scopes=None))
        assert hits == ALL_IDS           # old behaviour preserved

    def test_legacy_unstamped_chunk_invisible_once_scoped(self, store):
        # A book with NO stamp fails every guest list — this is why the seed
        # corpus must be re-ingested once after 4.2b ships.
        hits = _ids(store.search(_vec("q"), k=10, scopes=scopes_for("A")))
        assert "old" not in hits

    def test_empty_collection_returns_empty_not_error(self):
        # Regression (found live): a fresh/empty collection made search ask
        # Chroma for "top 0", which raises. Empty drawer → empty hands.
        empty = ChromaStore(collection_name="scoped_empty_test")
        assert empty.search(_vec("q"), k=5) == []
        assert empty.search(_vec("q"), k=5, scopes=scopes_for("A")) == []


# ------------------------------------------------ the card catalog (BM25)

class TestBM25ScopeFilter:
    @pytest.fixture()
    def bm25(self, corpus) -> BM25Retriever:
        r = BM25Retriever()
        r.add(corpus)
        return r

    def test_shared_token_visible_in_both_chats(self, bm25):
        for chat in ("A", "B"):
            hits = _ids(bm25.search("handbook", k=5, scopes=scopes_for(chat)))
            assert "pub" in hits

    def test_other_lockers_book_never_surfaces(self, bm25):
        # "secret" appears in BOTH lockers' books — the bouncer must seat
        # only the requesting chat's copy.
        hits = _ids(bm25.search("secret", k=5, scopes=scopes_for("A")))
        assert hits == {"a1"}

    def test_direct_keyword_hit_still_blocked_cross_chat(self, bm25):
        # Even asking for chat B's book BY its unique word yields nothing in
        # chat A — the strongest form of the isolation guarantee.
        assert bm25.search("barracuda", k=5, scopes=scopes_for("A")) == []

    def test_full_line_walk_beats_truncation(self, bm25):
        # k=1 with a guest list: the top RAW scorer for "secret" may be the
        # other locker's book, but the bouncer walks past it and still seats
        # OUR copy — no zero-survivors failure (full-line walk, not top-k slice).
        hits = _ids(bm25.search("secret", k=1, scopes=scopes_for("B")))
        assert hits == {"b1"}

    def test_no_guest_list_sees_both(self, bm25):
        hits = _ids(bm25.search("secret", k=5, scopes=None))
        assert hits == {"a1", "b1"}


# ------------------------------------------------ FAISS (dev backend, over-fetch)

class TestFAISSScopeFilter:
    @pytest.fixture()
    def store(self, corpus) -> FAISSStore:
        s = FAISSStore(dimension=DIM)
        s.add(corpus)
        return s

    def test_isolation_and_shared_visibility(self, store):
        hits = _ids(store.search(_vec("q"), k=10, scopes=scopes_for("A")))
        assert hits == SHELF_PLUS_A      # over-fetch + bouncer did their job

    def test_no_guest_list_means_everything(self, store):
        hits = _ids(store.search(_vec("q"), k=10, scopes=None))
        assert hits == ALL_IDS


# ------------------------------------------------ hybrid: both arms respect it

class TestHybridScopeFilter:
    @pytest.fixture()
    def hybrid(self, corpus) -> HybridRetriever:
        store = ChromaStore(collection_name="scoped_hybrid_test")
        store.add(corpus)
        bm25 = BM25Retriever()
        bm25.add(corpus)
        return HybridRetriever(vector_store=store, bm25=bm25, embedder=FakeEmbedder())

    def test_fused_results_never_leak_across_chats(self, hybrid):
        hits = _ids(hybrid.search("secret report", k=4, scopes=scopes_for("A")))
        assert "b1" not in hits and "old" not in hits
        assert "a1" in hits              # own locker present after RRF fusion

    def test_unscoped_hybrid_unchanged(self, hybrid):
        hits = _ids(hybrid.search("secret report", k=4, scopes=None))
        assert {"a1", "b1"} <= hits      # old behaviour: both lockers visible


# ------------------------------------------------ the clerk stamps at the door

class TestIngestStampsScope:
    """IngestPipeline must stamp every chunk it stores — default 'shared'."""

    @staticmethod
    def _pipeline_with_capture():
        from adaptiverag.ingest.pipeline import IngestPipeline
        from adaptiverag.ingest.models import Document

        class FakeLoader:
            def load_directory(self, path):
                return [Document(text="alpha beta gamma", metadata={"source": "a.md"})]

        class FakeChunker:
            def chunk(self, text, doc_id, metadata):
                return [SimpleNamespace(text=text, doc_id=doc_id,
                                        chunk_index=0, metadata=dict(metadata))]

        class FakeEmbedderBatch:
            def embed_batch(self, texts):
                return [_vec(t) for t in texts]

        class CaptureStore:
            def __init__(self):
                self.chunks = []

            def add(self, chunks):
                self.chunks.extend(chunks)

        store = CaptureStore()
        pipe = IngestPipeline(
            loader=FakeLoader(), chunker=FakeChunker(),
            embedder=FakeEmbedderBatch(), vector_store=store,
            summarizer=None, bm25=None,
        )
        return pipe, store

    def test_explicit_chat_scope_is_stamped(self, tmp_path):
        pipe, store = self._pipeline_with_capture()
        pipe.ingest(str(tmp_path), scope=chat_scope("conv-42"))
        assert store.chunks and all(
            c.metadata["scope"] == "chat:conv-42" for c in store.chunks
        )

    def test_default_stamp_is_shared(self, tmp_path):
        pipe, store = self._pipeline_with_capture()
        pipe.ingest(str(tmp_path))                       # no scope argument at all
        assert store.chunks and all(
            c.metadata["scope"] == SHARED_SCOPE for c in store.chunks
        )


# ------------------------------------------------ chains pass the list down

class RecordingRag:
    """Duck-typed rag_chain: records the guest list it was handed."""

    def __init__(self):
        self.seen_scopes = []

    def retrieve(self, query, scopes=None):
        self.seen_scopes.append(scopes)
        return []

    def query(self, question, expand=False, scopes=None):
        self.seen_scopes.append(scopes)
        return {"answer": "stub", "sources": []}


class TestScopesReachRetrieval:
    def test_multistep_hands_list_to_every_subquestion(self):
        from adaptiverag.reason.chain import MultiStepChain

        class FakeLLM:
            def generate(self, prompt):
                if "decomposition" in prompt:
                    return '["sub one?", "sub two?"]'
                return "synthesized"

        rag = RecordingRag()
        chain = MultiStepChain(rag_chain=rag, llm_client=FakeLLM())
        chain.query("complex question", scopes=scopes_for("A"))
        assert rag.seen_scopes == [scopes_for("A")] * 2   # both sub-questions scoped

    def test_search_documents_reads_ambient_guest_list(self):
        # The agent tool takes NO scope argument from the LLM — it reads the
        # list the API pinned to the request context.
        rag = RecordingRag()
        tool_fn = make_search_documents(rag)

        token = current_scopes.set(scopes_for("A"))
        try:
            tool_fn("anything")
        finally:
            current_scopes.reset(token)                   # never leak into other tests
        assert rag.seen_scopes == [scopes_for("A")]

    def test_search_documents_unscoped_stays_one_arg(self):
        # With no ambient list, the tool must call retrieve(query) the old
        # one-arg way, so pre-4.2b duck-typed fakes keep working.
        class OneArgRag:
            def __init__(self):
                self.called = False

            def retrieve(self, query):                    # NO scopes parameter at all
                self.called = True
                return []

        rag = OneArgRag()
        tool_fn = make_search_documents(rag)
        tool_fn("anything")                               # must not raise TypeError
        assert rag.called
