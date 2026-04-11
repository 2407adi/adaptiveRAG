"""BM25 (sparse) retriever for keyword-based search."""

from __future__ import annotations

import re
from rank_bm25 import BM25Okapi

from adaptiverag.retrieve.vector_store import StoredChunk, SearchResult
from adaptiverag.retrieve.vector_store import StoredChunk, SearchResult, VectorStore
from adaptiverag.ingest.embedder import Embedder


class BM25Retriever:
    """Keyword-based retriever using BM25 (Okapi variant)."""

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []
        self._tokenised_corpus: list[list[str]] = []
        self._index: BM25Okapi | None = None

    # ---- simple whitespace tokeniser ----
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    # ---- build the BM25 index ----
    def add(self, chunks: list[StoredChunk]) -> None:
        self._chunks.extend(chunks)
        self._tokenised_corpus = [
            self._tokenise(c.text) for c in self._chunks
        ]
        self._index = BM25Okapi(self._tokenised_corpus)

    # ---- search ----
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._index is None or len(self._chunks) == 0:
            return []

        tokens = self._tokenise(query)
        scores = self._index.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )
        return results
    
class HybridRetriever:
    """Combines dense (vector) and sparse (BM25) retrieval using RRF."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        embedder: Embedder,
        rrf_k: int = 60,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.0,
    ) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        # 1. get results from both retrievers
        dense_results = self.vector_store.search_by_text(query, k=k * 2, embed_fn=self.embedder.embed)
        sparse_results = self.bm25.search(query, k=k * 2)

        # 2. build rank maps: chunk_id -> rank (1-based)
        dense_ranks = {
            r.chunk_id: rank
            for rank, r in enumerate(dense_results, start=1)
        }
        sparse_ranks = {
            r.chunk_id: rank
            for rank, r in enumerate(sparse_results, start=1)
        }

        # 3. collect all unique chunk IDs
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # 4. compute RRF score for each chunk
        fused: dict[str, float] = {}
        for chunk_id in all_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += self.weight_dense / (self.rrf_k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += self.weight_sparse / (self.rrf_k + sparse_ranks[chunk_id])
            fused[chunk_id] = score

        # 5. sort by fused score, take top k
        top_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]

        # 6. build final SearchResult list
        # keep a lookup of the original results for text/metadata
        all_results = {r.chunk_id: r for r in dense_results}
        all_results.update({r.chunk_id: r for r in sparse_results})

        return [
            SearchResult(
                chunk_id=cid,
                text=all_results[cid].text,
                score=fused[cid],
                metadata={
                    **all_results[cid].metadata,
                    "dense_rank": dense_ranks.get(cid),
                    "sparse_rank": sparse_ranks.get(cid),
                    "rrf_score": fused[cid],
                },
            )
            for cid in top_ids
        ]