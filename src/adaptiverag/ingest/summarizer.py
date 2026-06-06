"""Generate a short topic summary of an ingested corpus.

Used by the QueryRouter to make corpus-aware routing decisions: 
questions about subjects not covered by the documents should
prefer DIRECT, not RAG.
"""

from __future__ import annotations
from pathlib import Path
import logging
from .models import Document

logger = logging.getLogger(__name__)


class CorpusSummarizer:
    """Builds and persists a summary of a document corpus.

    One LLM call for typical corpora ("stuff" mode). Falls back to a
    map-reduce two-pass approach when the concatenated documents exceed
    `char_budget`.
    """

    def __init__(
        self,
        llm_client,
        persist_path: str | Path,
        char_budget: int = 350_000,
    ):
        self.llm = llm_client
        self.persist_path = Path(persist_path)
        self.char_budget = char_budget

    def _build_stuff_prompt(self, documents: list[Document]) -> str:
        """Build the single-pass summarization prompt for small corpora."""

        doc_blocks = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            doc_blocks.append(f"[document: {source}]\n{doc.text}")
        corpus = "\n\n".join(doc_blocks)

        return f"""You are summarizing a corpus of documents a user has uploaded for question-answering. The summary will be shown to a query router that decides whether incoming questions can be answered from these documents or need to fall through to general knowledge.

    Read all documents below, then write ONE paragraph (~150 words) describing:
    - The main entities (companies, people, products, places)
    - The distinct topic areas covered
    - Specific facts likely to be queried (dates, numbers, names, key terms)

    Be concrete. Name things. Avoid generic phrases like "various topics" or "important information".

    Documents:

    {corpus}

    Topic summary (one paragraph, ~150 words):"""
        
    def _split_oversize_doc(self, doc: Document) -> list[Document]:
        """Break a single doc bigger than char_budget into sub-pieces.

        Each piece becomes a pseudo-Document with its source labeled
        'name (part k/N)' so the stuff prompt still has clean per-piece
        framing.
        """
        text = doc.text
        source = doc.metadata.get("source", "unknown")
        n_pieces = (len(text) + self.char_budget - 1) // self.char_budget  # ceiling div

        pieces = []
        for i in range(n_pieces):
            start = i * self.char_budget
            end = min(start + self.char_budget, len(text))
            pieces.append(Document(
                text=text[start:end],
                metadata={**doc.metadata, "source": f"{source} (part {i+1}/{n_pieces})"},
            ))
        return pieces
    
    def _batch_documents(self, documents: list[Document]) -> list[list[Document]]:
        """Greedy-pack documents into batches that each fit char_budget.

        Pass 1: any doc larger than the budget is pre-split into sub-pieces.
        Pass 2: greedy-pack the resulting (all <= budget) docs into batches.
        """
        # Pass 1: expand oversize docs into sub-pieces
        expanded: list[Document] = []
        for doc in documents:
            if len(doc.text) > self.char_budget:
                expanded.extend(self._split_oversize_doc(doc))
            else:
                expanded.append(doc)

        # Pass 2: greedy-pack
        batches: list[list[Document]] = []
        current: list[Document] = []
        current_chars = 0
        for doc in expanded:
            doc_chars = len(doc.text)
            if current and current_chars + doc_chars > self.char_budget:
                batches.append(current)
                current = [doc]
                current_chars = doc_chars
            else:
                current.append(doc)
                current_chars += doc_chars
        if current:
            batches.append(current)
        return batches
    
    def _build_reduce_prompt(self, summaries: list[str]) -> str:
        """Build the reduce-step prompt: merge partial summaries into one."""

        summary_blocks = []
        for i, s in enumerate(summaries, 1):
            summary_blocks.append(f"[batch {i} summary]\n{s}")
        combined = "\n\n".join(summary_blocks)

        return f"""You are merging multiple partial summaries of a single document corpus into ONE unified summary. Each partial summary covers a different subset of the documents.

    Read all partial summaries below, then write ONE paragraph (~150 words) describing:
    - The main entities (companies, people, products, places) across the whole corpus
    - The distinct topic areas covered
    - Specific facts likely to be queried (dates, numbers, names, key terms)

    Be concrete. Name things. Avoid generic phrases.

    Partial summaries:

    {combined}

    Unified topic summary (one paragraph, ~150 words):"""

    def _map_reduce(self, documents: list[Document]) -> str:
        """Two-pass summarization for over-budget corpora.

        Map: summarize each batch independently.
        Reduce: merge per-batch summaries into one unified paragraph.
        """
        batches = self._batch_documents(documents)
        logger.info("Map-reduce: %d batches", len(batches))

        batch_summaries: list[str] = []
        for i, batch in enumerate(batches, 1):
            logger.info("Summarizing batch %d/%d (%d docs)", i, len(batches), len(batch))
            prompt = self._build_stuff_prompt(batch)
            batch_summaries.append(self.llm.generate(prompt).strip())

        reduce_prompt = self._build_reduce_prompt(batch_summaries)
        return self.llm.generate(reduce_prompt).strip()
    
    def generate(self, documents: list[Document]) -> str | None:
        """Generate a topic summary of the corpus.

        Returns None for empty input or on LLM failure — never raises.
        """
        if not documents:
            return None

        try:
            total_chars = sum(len(doc.text) for doc in documents)
            if total_chars <= self.char_budget:
                prompt = self._build_stuff_prompt(documents)
                summary = self.llm.generate(prompt)
            else:
                logger.info(
                    "Corpus exceeds budget (%d > %d chars), using map-reduce",
                    total_chars, self.char_budget,
                )
                summary = self._map_reduce(documents)

            return summary.strip() or None
        except Exception as e:
            logger.warning("Corpus summarization failed: %s", e)
            return None
        
    def save(self, summary: str) -> None:
        """Write the summary to persist_path (creates parent dir)."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(summary, encoding="utf-8")

    def load(self) -> str | None:
        """Read the summary from persist_path, or None if not present."""
        if not self.persist_path.exists():
            return None
        text = self.persist_path.read_text(encoding="utf-8").strip()
        return text or None

    def generate_and_save(self, documents: list[Document]) -> str | None:
        """Generate a summary, save it if non-None, return it."""
        summary = self.generate(documents)
        if summary:
            self.save(summary)
        return summary