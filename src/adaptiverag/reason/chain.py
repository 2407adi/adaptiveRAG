from pathlib import Path
import json
import logging

from ..retrieve.vector_store import VectorStore, SearchResult
from ..ingest.embedder import Embedder
from ..retrieve.query_expander import QueryExpander

logger = logging.getLogger(__name__)


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
        return (f"""You are a helpful assistant with access to the user's documents.
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

Answer:""")


class MultiStepChain:
    """Handles complex queries by decomposing → answering sub-questions via RAG → synthesizing."""

    def __init__(self, rag_chain: RAGChain, llm_client, max_sub_questions: int = 4):
        self.rag_chain = rag_chain
        self.llm_client = llm_client
        self.max_sub_questions = max_sub_questions

    def _decompose(self, question: str) -> list[str]:
        """Phase 1: Break a complex question into independent sub-questions."""
        prompt = f"""You are a question decomposition assistant.

            Given a complex question, break it down into {self.max_sub_questions} or fewer
            simple, independent sub-questions that can each be answered by searching a
            document collection. Each sub-question should target ONE specific piece of
            information.

            Rules:
            - Output ONLY a JSON array of strings, nothing else.
            - Each sub-question must be self-contained (understandable without the original).
            - Do NOT include the original question as a sub-question.
            - Minimum 2 sub-questions, maximum {self.max_sub_questions}.

            Example:
            Question: "How does the vacation policy in the US compare to the UK, and which is more generous?"
            Output: ["What is the vacation policy for the US office?", "What is the vacation policy for the UK office?"]

            Now decompose this question:
            Question: "{question}"
            Output:"""

        raw = self.llm_client.generate(prompt)
        return self._parse_sub_questions(raw, question)

    def _parse_sub_questions(self, raw: str, original_question: str) -> list[str]:
        """Parse LLM output into a list of sub-questions, with fallback."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            sub_questions = json.loads(cleaned)
            if (isinstance(sub_questions, list)
                    and 2 <= len(sub_questions) <= self.max_sub_questions
                    and all(isinstance(q, str) for q in sub_questions)):
                logger.info("Decomposed into %d sub-questions", len(sub_questions))
                return sub_questions
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: treat the original as a single sub-question
        logger.warning("Decomposition failed, falling back to original question")
        return [original_question]
    
    def _answer_sub_questions(self, sub_questions: list[str],
                              expand: bool = False) -> list[dict]:
        """Phase 2: Answer each sub-question independently via RAG.

        Returns a list of step dicts, each containing:
            - sub_question: the sub-question text
            - answer: the RAG answer
            - sources: citation list from RAGChain
        """
        steps = []
        for i, sub_q in enumerate(sub_questions, 1):
            logger.info("Answering sub-question %d/%d: %s",
                        i, len(sub_questions), sub_q)

            result = self.rag_chain.query(sub_q, expand=expand)

            steps.append({
                "sub_question": sub_q,
                "answer": result["answer"],
                "sources": result["sources"],
            })

        return steps
    
    def _synthesize(self, question: str, steps: list[dict]) -> str:
        """Phase 3: Combine sub-answers into a coherent final response."""
        # Format the reasoning trace for the synthesis prompt
        evidence_blocks = []
        for i, step in enumerate(steps, 1):
            evidence_blocks.append(
                f"Sub-question {i}: {step['sub_question']}\n"
                f"Answer: {step['answer']}"
            )
        evidence = "\n\n".join(evidence_blocks)

        prompt = f"""You are a reasoning assistant that synthesizes information.

            You were given a complex question and it was broken into sub-questions.
            Each sub-question has been answered independently using the user's documents.

            Your job:
            1. Combine the sub-answers into ONE coherent response to the original question.
            2. Show your reasoning — explain how the sub-answers connect and what you conclude.
            3. If sub-answers contradict each other, acknowledge the conflict and explain.
            4. Use markdown formatting for clarity.
            5. Cite using [Sub-question N] notation when referencing a specific sub-answer.

            Original question: {question}

            Research findings:
            {evidence}

            Synthesized answer:"""

        return self.llm_client.generate(prompt)

    def query(self, question: str, expand: bool = False) -> dict:
        """Run the full decompose → answer → synthesize cycle.

        Args:
            question: The user's complex question.
            expand:   If True, use query expansion on each sub-question.

        Returns:
            dict with 'answer' (str), 'sources' (list), and 'reasoning_steps' (list)
        """
        # Phase 1: Decompose
        sub_questions = self._decompose(question)

        # Phase 2: Answer each sub-question via RAG
        steps = self._answer_sub_questions(sub_questions, expand=expand)

        # Phase 3: Synthesize
        answer = self._synthesize(question, steps)

        # Merge all sources, deduplicate by chunk_id
        seen_chunks = set()
        all_sources = []
        for step in steps:
            for source in step["sources"]:
                if source["chunk_id"] not in seen_chunks:
                    seen_chunks.add(source["chunk_id"])
                    all_sources.append(source)

        logger.info(
            "MultiStepChain complete: %d sub-questions, %d unique sources",
            len(steps), len(all_sources),
        )

        return {
            "answer": answer,
            "sources": all_sources,
            "reasoning_steps": steps,
        }