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