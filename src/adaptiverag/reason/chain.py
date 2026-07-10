from pathlib import Path
import json
import logging

from ..retrieve.vector_store import VectorStore, SearchResult
from ..ingest.embedder import Embedder
from ..retrieve.query_expander import QueryExpander
from collections.abc import Iterator

logger = logging.getLogger(__name__)


class RAGChain:
    """Takes a query, retrieves relevant chunks, and generates an answer with citations."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder,
                llm_client, top_k: int = 5,
                query_expander: QueryExpander | None = None,
                reranker=None, fetch_k: int = 20,
                hybrid_retriever=None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.top_k = top_k
        self.query_expander = query_expander
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.hybrid_retriever = hybrid_retriever   # None → dense-only


    def _search(self, query: str, k: int,
                scopes: list[str] | None = None) -> list[SearchResult]:
        """One retrieval call: hybrid if a HybridRetriever was injected,
        else dense-only. Keeps retrieve() agnostic to which is active —
        the factory picks based on settings.retrieval.mode.
        `scopes` = the guest list (Block 4.2b); None = no filter, old behaviour.
        """
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever.search(query, k=k, scopes=scopes)
        return self.vector_store.search_by_text(
            query_text=query, k=k, embed_fn=self.embedder.embed, scopes=scopes,
        )

    def retrieve(self, question: str, expand: bool = False,
                 scopes: list[str] | None = None) -> list[SearchResult]:
        """Retrieve the chunks for a question — no LLM, no answer.

        This is the whole evidence-gathering half of the pipeline:
        dense search → optional expansion-merge → optional rerank.
        query() calls this; the agent's search_documents tool will too.
        """
        # How wide to cast stage-1's net: if a reranker is attached,
        # fetch fetch_k candidates so it has a real shortlist to reorder.
        retrieve_k = self.fetch_k if self.reranker else self.top_k

        # 1. Always search with the original query (guest list rides along)
        original_results: list[SearchResult] = self._search(question, retrieve_k, scopes=scopes)

        # 1b. If expansion is on, also search with the rewritten query and merge
        if expand and self.query_expander:
            expanded_query = self.query_expander.expand(question)
            expanded_results: list[SearchResult] = self._search(expanded_query, retrieve_k, scopes=scopes)
            results = self._merge_results(original_results, expanded_results, limit=retrieve_k)
        else:
            results = original_results

        # 1c. Rerank — cross-encoder re-scores the wide net, cuts to top_k.
        if self.reranker:
            results = self.reranker.rerank(question, results, top_n=self.top_k)

        return results

    def query(self, question: str, expand: bool = False,
              scopes: list[str] | None = None) -> dict:
        # Evidence half — extracted so the agent can reuse it.
        results = self.retrieve(question, expand=expand, scopes=scopes)

        # Generation half — format → prompt → LLM → package sources.
        context = self._format_context(results)
        prompt = self._build_prompt(question, context)
        answer = self.llm_client.generate(prompt)

        sources = self._package_sources(results)

        return {"answer": answer, "sources": sources}
    
    def query_stream(self, question: str, expand: bool = False,
                     scopes: list[str] | None = None) -> Iterator[dict]:
        """Like query(), but narrates: stage notes → evidence → answer piece-by-piece."""
        yield {"type": "stage", "stage": "retrieving"}            # note 1: "detective is at the archive"
        results = self.retrieve(question, expand=expand, scopes=scopes)  # (blocking, but fast)
        sources = self._package_sources(results)
        yield {"type": "sources", "sources": sources}             # note 2: evidence slips, before the answer

        prompt = self._build_prompt(question, self._format_context(results))
        answer_parts: list[str] = []                              # keep a carbon copy as we dictate
        for piece in self.llm_client.generate_stream(prompt):     # bucket brigade from OpenAI
            answer_parts.append(piece)
            yield {"type": "token", "text": piece}                # note 3..n: a few characters each

        yield {"type": "done", "answer": "".join(answer_parts),   # final note: the assembled report,
               "sources": sources}                                #   so the caller needn't glue tokens
    
    def _package_sources(self, results: list[SearchResult]) -> list[dict]:
        """Turn SearchResults into the citation dicts the UI/API expect."""
        return [
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
                       expanded: list[SearchResult], limit=None) -> list[SearchResult]:
        """Merge two result sets, keeping the higher score for duplicates."""
        best: dict[str, SearchResult] = {}

        for r in original:
            best[r.chunk_id] = r

        for r in expanded:
            if r.chunk_id not in best or r.score > best[r.chunk_id].score:
                best[r.chunk_id] = r

        merged = sorted(best.values(), key=lambda r: r.score, reverse=True)
        limit = limit if limit is not None else self.top_k
        return merged[:limit]
    
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
with general knowledge, say so clearly. If sources disagree on a fact, present both versions and note the conflict.

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
                              expand: bool = False,
                              scopes: list[str] | None = None) -> list[dict]:
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

            result = self.rag_chain.query(sub_q, expand=expand, scopes=scopes)

            steps.append({
                "sub_question": sub_q,
                "answer": result["answer"],
                "sources": result["sources"],
            })

        return steps
    
    def _synthesis_prompt(self, question: str, steps: list[dict]) -> str:

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
        return prompt

    
    def _synthesize(self, question: str, steps: list[dict]) -> str:
        """Phase 3: Combine sub-answers into a coherent final response."""
        return self.llm_client.generate(self._synthesis_prompt(question, steps))
    
    @staticmethod
    def _merge_sources(steps: list[dict]) -> list[dict]:
        """All steps' sources, deduped by chunk_id, first-seen order kept."""
        seen, merged = set(), []
        for step in steps:
            for source in step["sources"]:
                if source["chunk_id"] not in seen:
                    seen.add(source["chunk_id"])
                    merged.append(source)
        return merged
    

    def query(self, question: str, expand: bool = False,
              scopes: list[str] | None = None) -> dict:
        """Run the full decompose → answer → synthesize cycle.

        Args:
            question: The user's complex question.
            expand:   If True, use query expansion on each sub-question.
            scopes:   Guest list (Block 4.2b); every sub-question inherits it.

        Returns:
            dict with 'answer' (str), 'sources' (list), and 'reasoning_steps' (list)
        """
        # Phase 1: Decompose
        sub_questions = self._decompose(question)

        # Phase 2: Answer each sub-question via RAG (same guest list each time)
        steps = self._answer_sub_questions(sub_questions, expand=expand, scopes=scopes)

        # Phase 3: Synthesize
        answer = self._synthesize(question, steps)

        # Merge all sources, deduplicate by chunk_id
        all_sources = self._merge_sources(steps)

        logger.info(
            "MultiStepChain complete: %d sub-questions, %d unique sources",
            len(steps), len(all_sources),
        )

        return {
            "answer": answer,
            "sources": all_sources,
            "reasoning_steps": steps,
        }
    
    def query_stream(self, question: str, expand: bool = False,
                     scopes: list[str] | None = None) -> Iterator[dict]:
        """Narrated version: the Analyst thinks out loud, then dictates the synthesis."""
        yield {"type": "stage", "stage": "decomposing"}               # "breaking your question apart..."
        sub_questions = self._decompose(question)

        steps = []
        for i, sub_q in enumerate(sub_questions, 1):
            yield {"type": "stage", "stage": "sub_question",          # "working on 2 of 3: ..."
                   "index": i, "total": len(sub_questions), "text": sub_q}
            result = self.rag_chain.query(sub_q, expand=expand, scopes=scopes)  # non-stream: internal legwork
            steps.append({"sub_question": sub_q, "answer": result["answer"],
                          "sources": result["sources"]})

        sources = self._merge_sources(steps)
        yield {"type": "sources", "sources": sources}

        yield {"type": "stage", "stage": "synthesizing"}              # "writing the final report..."
        parts: list[str] = []
        for piece in self.llm_client.generate_stream(self._synthesis_prompt(question, steps)):
            parts.append(piece)
            yield {"type": "token", "text": piece}                    # only the FINAL answer streams

        yield {"type": "done", "answer": "".join(parts),
               "sources": sources, "reasoning_steps": steps}