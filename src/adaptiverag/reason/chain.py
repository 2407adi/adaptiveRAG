from pathlib import Path

from ..retrieve.vector_store import VectorStore, SearchResult
from ..ingest.embedder import Embedder
from ..retrieve.query_expander import QueryExpander


class RAGChain:
    """Takes a query, retrieves relevant chunks, and generates an answer with citations."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder,
                 llm_client, top_k: int = 5,
                 query_expander: QueryExpander | None = None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.top_k = top_k
        self.query_expander = query_expander

    def query(self, question: str, expand: bool = False) -> dict:
        """Run the full retrieve → format → generate cycle.

        Args:
            question: The user's question.
            expand:   If True and a QueryExpander is configured, run
                      dual-search (original + expanded) and merge results.

        Returns:
            dict with 'answer' (str) and 'sources' (list of citation dicts)
        """
        # 1. Retrieve — always search with the original query
        original_results: list[SearchResult] = self.vector_store.search_by_text(
            query_text=question,
            k=self.top_k,
            embed_fn=self.embedder.embed,
        )

        # 1b. If expansion is on, also search with the rewritten query
        if expand and self.query_expander:
            expanded_query = self.query_expander.expand(question)
            expanded_results: list[SearchResult] = self.vector_store.search_by_text(
                query_text=expanded_query,
                k=self.top_k,
                embed_fn=self.embedder.embed,
            )
            results = self._merge_results(original_results, expanded_results)
        else:
            results = original_results

        # 2. Build prompt with retrieved context
        context = self._format_context(results)
        prompt = self._build_prompt(question, context)

        # 3. Call the LLM
        answer = self.llm_client.generate(prompt)

        # 4. Package sources for citations
        sources = [
                    {
                        "chunk_id": r.chunk_id,
                        "source": Path(r.metadata.get("source", "unknown")).name,
                        "page": r.metadata.get("page", ""),
                        "chunk_index": r.metadata.get("chunk_index", "unknown"),
                        "score": round(r.score, 4),
                        "text_preview": r.text[:200],
                        "full_text": r.text,
                    }
                    for r in results
                ]
        return {"answer": answer, "sources": sources}
    
    def _format_context(self, results: list[SearchResult]) -> str:
        """Format retrieved chunks into a numbered context string."""
        if not results:
            return "No relevant context found."

        blocks = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            chunk_idx = r.metadata.get("chunk_index", "?")
            blocks.append(
                f"[{i}] (source: {source}, chunk: {chunk_idx})\n{r.text}"
            )

        return "\n\n".join(blocks)
    
    def _merge_results(self, original: list[SearchResult],
                       expanded: list[SearchResult]) -> list[SearchResult]:
        """Merge two result sets, keeping the higher score for duplicates."""
        best: dict[str, SearchResult] = {}

        for r in original:
            best[r.chunk_id] = r

        for r in expanded:
            if r.chunk_id not in best or r.score > best[r.chunk_id].score:
                best[r.chunk_id] = r

        merged = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return merged[:self.top_k]
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the final prompt sent to the LLM."""
        return f"""You are a helpful assistant with access to the user's documents.
Use the context below as your primary source of information, especially for
domain-specific or proprietary details. You may also draw on your general
knowledge to provide complete, well-rounded answers — but when your general
knowledge conflicts with the provided context, prefer the context.

Always cite the provided context using bracket notation like [1], [2] when
referencing specific claims from the documents. If you use general knowledge
that is NOT from the context, do not add a citation — just state it naturally.

If the context doesn't contain enough information and you cannot supplement
with general knowledge, say so clearly.

Formatting rules:
- For mathematical formulas, use LaTeX with $...$ for inline math and $$...$$ for block equations.
- Use markdown formatting for structure (headers, bold, lists).

Context:
{context}

Question: {question}

Answer:"""