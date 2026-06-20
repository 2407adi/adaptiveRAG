"""Retrieve/generate split + hybrid branch in RAGChain (offline, mocked)."""

from unittest.mock import MagicMock
from adaptiverag.reason.chain import RAGChain
from adaptiverag.retrieve.vector_store import SearchResult


def _results(*ids):
    return [
        SearchResult(chunk_id=c, text=f"text {c}", score=0.9,
                     metadata={"source": "doc.pdf", "chunk_index": 0})
        for c in ids
    ]


def _chain(**kwargs):
    vector_store, embedder, llm = MagicMock(), MagicMock(), MagicMock()
    embedder.embed.return_value = [0.0]
    llm.generate.return_value = "generated answer"
    chain = RAGChain(vector_store=vector_store, embedder=embedder,
                     llm_client=llm, top_k=5, **kwargs)
    return chain, vector_store, llm


class TestSplit:
    def test_retrieve_returns_raw_chunks_no_llm(self):
        chain, vs, llm = _chain()
        vs.search_by_text.return_value = _results("c1", "c2")
        out = chain.retrieve("revenue")
        assert [r.chunk_id for r in out] == ["c1", "c2"]
        llm.generate.assert_not_called()          # the whole point

    def test_query_unchanged_shape(self):
        chain, vs, llm = _chain()
        vs.search_by_text.return_value = _results("c1")
        out = chain.query("revenue")
        assert out["answer"] == "generated answer"
        assert out["sources"][0]["chunk_id"] == "c1"
        llm.generate.assert_called_once()


class TestHybridBranch:
    def test_dense_when_no_hybrid(self):
        chain, vs, _ = _chain()
        vs.search_by_text.return_value = _results("c1")
        chain.retrieve("q")
        vs.search_by_text.assert_called_once()

    def test_routes_through_hybrid_when_injected(self):
        hybrid = MagicMock()
        hybrid.search.return_value = _results("h1")
        chain, vs, _ = _chain(hybrid_retriever=hybrid)
        out = chain.retrieve("q")
        hybrid.search.assert_called_once()
        vs.search_by_text.assert_not_called()      # dense bypassed
        assert out[0].chunk_id == "h1"