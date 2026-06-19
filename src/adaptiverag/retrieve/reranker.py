"""Cross-encoder reranking — a precision second pass over retrieved chunks.

Stage 1 (cheap bi-encoder retrieval) fetches a wide net of candidates.
Stage 2 (this module) re-scores each (query, chunk) pair jointly and
keeps the best `top_n`. See CLAUDE.md Block 2.5 for the why.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from adaptiverag.retrieve.vector_store import SearchResult
from sentence_transformers import CrossEncoder



class Reranker(ABC):
    """Re-orders retrieved chunks by joint (query, chunk) relevance.

    Consumes and returns the same SearchResult dataclass the retrievers
    produce, so it drops into the pipeline between retrieval and the
    prompt without touching either side's shape.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """Score every (query, chunk) pair, sort desc, return top_n.

        Args:
            query:   The user's question.
            results: Stage-1 candidates (typically fetch_k of them, ~20).
            top_n:   How many survive into the prompt (typically 5).

        Returns:
            A new list of <= top_n SearchResults, ordered most- to
            least-relevant. The original bi-encoder/RRF score is moved
            into metadata['retrieval_score']; SearchResult.score now
            holds the reranker's score (note: for local cross-encoders
            this is a raw logit — meaningful only for ordering).
        """
        ...


class CrossEncoderReranker(Reranker):
    """Local cross-encoder reranker via sentence-transformers.

    Free, offline, no API key. Loads an MS MARCO-trained cross-encoder
    once at construction, then scores (query, chunk) pairs on CPU at
    ask time. Outputs raw logits — unbounded, can be negative — so the
    scores are meaningful only for *ordering*, not as probabilities.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        # Heavy: downloads (~80MB, first run only) and loads the model.
        # Build this ONCE and reuse it, same as the embedder.
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        # Nothing to do on an empty candidate list.
        if not results:
            return []

        # 1. Build one (query, chunk_text) pair per candidate.
        pairs = [(query, r.text) for r in results]

        # 2. One batched forward pass → one score per pair.
        scores = self._model.predict(pairs)

        # 3. Re-package each result: preserve the old retrieval score
        #    in metadata, overwrite .score with the reranker's score.
        reranked = []
        for r, new_score in zip(results, scores):
            reranked.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    score=float(new_score),
                    metadata={
                        **r.metadata,
                        "retrieval_score": r.score,
                        "rerank_score": float(new_score),
                    },
                )
            )

        # 4. Sort by the new score, highest first; keep top_n.
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_n]
    
import os


class CohereReranker(Reranker):
    """Cohere Rerank API backend (optional, paid tier).

    Higher quality than the local model, but needs a network hop and an
    API key (COHERE_API_KEY). Returns scores in [0, 1] — unlike the local
    cross-encoder's raw logits — but we still use them only for ordering
    to keep both backends interchangeable.
    """

    def __init__(
        self,
        model: str = "rerank-v3.5",
        api_key: str | None = None,
    ):
        import cohere  # local import: only needed if this backend is used

        self._client = cohere.ClientV2(
            api_key=api_key or os.getenv("COHERE_API_KEY")
        )
        self._model = model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        # 1. Send query + candidate texts to Cohere in one call.
        docs = [r.text for r in results]
        response = self._client.rerank(
            model=self._model,
            query=query,
            documents=docs,
            top_n=top_n,
        )

        # 2. Response gives back ranked entries: each has .index (into our
        #    original list) and .relevance_score. Already sorted best-first.
        reranked = []
        for item in response.results:
            original = results[item.index]
            reranked.append(
                SearchResult(
                    chunk_id=original.chunk_id,
                    text=original.text,
                    score=float(item.relevance_score),
                    metadata={
                        **original.metadata,
                        "retrieval_score": original.score,
                        "rerank_score": float(item.relevance_score),
                    },
                )
            )

        return reranked

def create_reranker(backend: str = "cross_encoder", **kwargs) -> Reranker:
    """Build a reranker from config.

    Args:
        backend: 'cross_encoder' (local, default) or 'cohere'.
        **kwargs: passed to the backend constructor
            cross_encoder: model_name
            cohere:        model, api_key

    Mirrors create_vector_store / create_embedder.
    """
    if backend == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    elif backend == "cohere":
        return CohereReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker backend: {backend}")
    
def build_reranker_from_settings(rerank_cfg):
    """Build a reranker from a RerankConfig, or None if disabled.

    Centralizes the enabled-check + backend dispatch so app.py and
    suite.py don't each reimplement it.
    """
    if not rerank_cfg.enabled:
        return None

    if rerank_cfg.backend == "cross_encoder":
        return create_reranker("cross_encoder", model_name=rerank_cfg.model)
    elif rerank_cfg.backend == "cohere":
        return create_reranker("cohere", model=rerank_cfg.model)
    else:
        raise ValueError(f"Unknown reranker backend: {rerank_cfg.backend}")