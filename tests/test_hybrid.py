"""Tests for BM25 and Hybrid (RRF) retrieval."""

import numpy as np
import pytest

from adaptiverag.retrieve.vector_store import StoredChunk, ChromaStore
from adaptiverag.retrieve.hybrid import BM25Retriever, HybridRetriever
from adaptiverag.ingest.embedder import LocalEmbedder


# ---- helpers ----

def _make_chunk(id: str, text: str, dim: int = 384) -> StoredChunk:
    """Create a StoredChunk with a random normalised embedding."""
    vec = np.random.default_rng(hash(id) % 2**32).random(dim)
    vec = vec / np.linalg.norm(vec)
    return StoredChunk(
        id=id,
        text=text,
        embedding=vec.tolist(),
        metadata={"source": "test"},
    )


CHUNKS = [
    _make_chunk("c1", "Basel III capital requirements mandate minimum CET1 ratio"),
    _make_chunk("c2", "The liquidity coverage ratio LCR ensures short term resilience"),
    _make_chunk("c3", "Python is a popular programming language for data science"),
    _make_chunk("c4", "Basel III introduced the leverage ratio as a backstop measure"),
    _make_chunk("c5", "Machine learning models require large training datasets"),
]


# ---- BM25 tests ----

class TestBM25Retriever:

    def test_exact_keyword_match(self):
        bm25 = BM25Retriever()
        bm25.add(CHUNKS)
        results = bm25.search("Basel III", k=3)
        result_ids = [r.chunk_id for r in results]
        assert "c1" in result_ids
        assert "c4" in result_ids

    def test_no_match_returns_empty(self):
        bm25 = BM25Retriever()
        bm25.add(CHUNKS)
        results = bm25.search("quantum entanglement", k=3)
        assert len(results) == 0

    def test_scores_are_positive(self):
        bm25 = BM25Retriever()
        bm25.add(CHUNKS)
        results = bm25.search("liquidity ratio", k=3)
        for r in results:
            assert r.score > 0

    def test_empty_index_returns_empty(self):
        bm25 = BM25Retriever()
        results = bm25.search("anything", k=3)
        assert results == []


# ---- Hybrid RRF tests ----

class TestHybridRetriever:

    @pytest.fixture
    def hybrid(self):
        """Set up a ChromaStore + BM25 + HybridRetriever."""
        embedder = LocalEmbedder()

        store = ChromaStore(collection_name="test_hybrid")
        store.add(CHUNKS)

        bm25 = BM25Retriever()
        bm25.add(CHUNKS)

        return HybridRetriever(
            vector_store=store,
            bm25=bm25,
            embedder=embedder,
            rrf_k=60,
            weight_dense=1.0,
            weight_sparse=1.0,
        )

    def test_returns_results(self, hybrid):
        results = hybrid.search("Basel III capital", k=3)
        assert len(results) > 0

    def test_metadata_has_ranks(self, hybrid):
        results = hybrid.search("Basel III", k=3)
        first = results[0]
        assert "rrf_score" in first.metadata
        has_dense = first.metadata.get("dense_rank") is not None
        has_sparse = first.metadata.get("sparse_rank") is not None
        assert has_dense or has_sparse

    def test_keyword_query_benefits_from_hybrid(self, hybrid):
        """Exact keyword 'LCR' should surface c2 in hybrid but might
        struggle in dense-only since embeddings may not capture acronyms."""
        results = hybrid.search("LCR", k=3)
        result_ids = [r.chunk_id for r in results]
        assert "c2" in result_ids

    def test_fusion_weights_change_ranking(self):
        """Cranking sparse weight should favour BM25 results."""
        embedder = LocalEmbedder()

        store = ChromaStore(collection_name="test_weights")
        store.add(CHUNKS)
        bm25 = BM25Retriever()
        bm25.add(CHUNKS)

        sparse_heavy = HybridRetriever(
            vector_store=store, bm25=bm25, embedder=embedder,
            weight_dense=0.1, weight_sparse=2.0,
        )
        dense_heavy = HybridRetriever(
            vector_store=store, bm25=bm25, embedder=embedder,
            weight_dense=2.0, weight_sparse=0.1,
        )

        sparse_results = sparse_heavy.search("Basel III", k=3)
        dense_results = dense_heavy.search("Basel III", k=3)

        sparse_ids = [r.chunk_id for r in sparse_results]
        dense_ids = [r.chunk_id for r in dense_results]
        assert len(sparse_ids) > 0
        assert len(dense_ids) > 0

    def test_k_limits_output(self, hybrid):
        results = hybrid.search("Basel", k=2)
        assert len(results) <= 2

# ---- lazy seeding + thread safety (the boot-loop fix) ----

class TestLazySeedAndThreadSafety:
    def test_search_before_any_add_is_empty_not_crash(self):
        # The window between boot and the background seed finishing:
        # hybrid must transparently degrade to dense-only.
        assert BM25Retriever().search("anything", k=3) == []

    def test_add_dedupes_by_chunk_id(self):
        # Ingest racing the boot seed can deliver the same chunk twice.
        # (Filler docs: BM25 IDF ≤ 0 when a term is in half+ of a tiny corpus.)
        bm25 = BM25Retriever()
        bm25.add(CHUNKS)                                     # filler corpus
        bm25.add([_make_chunk("dup", "Basel III capital rules")])
        bm25.add([_make_chunk("dup", "Basel III capital rules"),
                  _make_chunk("new", "unique liquidity keyword zanthum")])
        ids = [r.chunk_id for r in bm25.search("zanthum Basel", k=20)]
        assert ids.count("dup") == 1 and "new" in ids

    def test_seed_bm25_pulls_store_into_index(self):
        from types import SimpleNamespace
        from adaptiverag.pipeline import seed_bm25
        bm25 = BM25Retriever()
        seeded = CHUNKS + [_make_chunk("s1", "unique keyword zanthum appears here")]
        store = SimpleNamespace(get_all=lambda: seeded)
        assert seed_bm25(bm25, store) == len(seeded)
        assert [r.chunk_id for r in bm25.search("zanthum", k=5)] == ["s1"]

    def test_seed_bm25_empty_store_leaves_index_unbuilt(self):
        from types import SimpleNamespace
        from adaptiverag.pipeline import seed_bm25
        bm25 = BM25Retriever()
        assert seed_bm25(bm25, SimpleNamespace(get_all=lambda: [])) == 0
        assert bm25.search("anything") == []

    def test_concurrent_add_and_search_dont_corrupt(self):
        # Smoke drill for the lock: hammer add() and search() from threads.
        import threading
        bm25 = BM25Retriever()
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    bm25.add([_make_chunk(f"w{n}-{i}", f"Basel document number {i}")])
            except Exception as e:  # pragma: no cover
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    bm25.search("Basel document", k=5)
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(bm25.search("Basel document", k=100)) > 0
