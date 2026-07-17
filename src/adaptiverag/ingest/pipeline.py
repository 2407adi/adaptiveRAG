
import time
from collections.abc import Callable

from .loader import DocumentLoader
from .chunker import BaseChunker
from .embedder import Embedder
from .exceptions import IngestTooLarge
from ..retrieve.vector_store import VectorStore, StoredChunk
from .summarizer import CorpusSummarizer
from ..retrieve.hybrid import BM25Retriever

# ProgressCallback(stage, done, total) — stage is one of:
# "loading", "chunking", "embedding", "storing", "summarizing".
# done/total are chunk counts during "embedding", 0/0 otherwise.
ProgressCallback = Callable[[str, int, int], None]


class IngestPipeline:
    """Orchestrates: load → chunk → embed → store.

    Embedding is done in small batches with a short sleep between them
    (throttling), so a big document can't monopolize the CPU of a small
    container — serving traffic keeps getting scheduler slots while a
    long ingest chews through its chunks in the background.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: BaseChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        summarizer: CorpusSummarizer | None = None,
        bm25: BM25Retriever | None = None,
        batch_size: int = 32,          # chunks embedded per batch
        throttle_seconds: float = 0.1,  # breather between batches (lets /health answer)
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.summarizer = summarizer
        self.bm25 = bm25       # None → dense-only pipeline, no keyword index
        self.batch_size = max(1, batch_size)
        self.throttle_seconds = throttle_seconds

    def ingest(
        self,
        source_path: str,
        scope: str = "shared",
        progress_cb: ProgressCallback | None = None,
        max_chunks: int | None = None,
    ) -> dict:
        """Process all documents in a directory and index them.

        Args:
            source_path: Directory of files to ingest.
            scope:       4.2b stamp — "shared" or "chat:<id>".
            progress_cb: Optional callback invoked as work advances; the API's
                         job runner uses it to publish live status for the UI.
            max_chunks:  Per-upload work cap. If chunking produces more than
                         this, IngestTooLarge is raised BEFORE any embedding —
                         cheap to check, protects the box from monster uploads.
                         None = uncapped (Streamlit harness, eval suite, tests).

        Raises:
            IngestTooLarge: chunk count exceeded max_chunks. No partial
                writes — the store is untouched when this is raised.
        """
        def report(stage: str, done: int = 0, total: int = 0) -> None:
            if progress_cb:
                progress_cb(stage, done, total)

        # ── Phase 1: load + chunk EVERYTHING first (fast, no heavy compute).
        # Knowing the total chunk count up front is what makes both the work
        # cap and a truthful progress bar possible.
        report("loading")
        documents = self.loader.load_directory(source_path)

        report("chunking")
        all_chunks = []
        for doc in documents:
            doc_id = doc.metadata.get("source", "unknown")
            all_chunks.extend(self.chunker.chunk(doc.text, doc_id, doc.metadata))

        total = len(all_chunks)
        if max_chunks is not None and total > max_chunks:
            raise IngestTooLarge(total_chunks=total, max_chunks=max_chunks)

        # ── Phase 2: embed in throttled batches (the expensive part).
        texts = [c.text for c in all_chunks]
        embeddings: list[list[float]] = []
        for start in range(0, total, self.batch_size):
            embeddings.extend(self.embedder.embed_batch(texts[start:start + self.batch_size]))
            done = min(start + self.batch_size, total)
            report("embedding", done, total)
            if self.throttle_seconds and done < total:   # no pointless tail sleep
                time.sleep(self.throttle_seconds)

        all_vector_chunks = []
        for chunk, embedding in zip(all_chunks, embeddings):
            page = chunk.metadata.get("page", "")
            page_suffix = f"::page-{page}" if page else ""
            all_vector_chunks.append(StoredChunk(
                id=f"{chunk.doc_id}{page_suffix}::chunk-{chunk.chunk_index}",
                text=chunk.text,
                embedding=embedding,
                metadata={
                    **chunk.metadata,
                    "source": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "scope": scope,
                },
            ))

        # ── Phase 3: store everything in one batch.
        report("storing", total, total)
        self.vector_store.add(all_vector_chunks)

        # Keep the keyword (BM25) index in lockstep with the dense store,
        # so a HybridRetriever sharing this same bm25 instance sees the
        # newly ingested chunks immediately.
        if self.bm25 is not None:
            self.bm25.add(all_vector_chunks)

        corpus_summary: str | None = None
        if self.summarizer is not None and documents:
            report("summarizing", total, total)
            corpus_summary = self.summarizer.generate_and_save(documents)

        return {
            "files_processed": len(set(
                c.metadata.get("source", "") for c in all_vector_chunks
            )),
            "total_chunks": len(all_vector_chunks),
            "corpus_summary": corpus_summary,
        }
