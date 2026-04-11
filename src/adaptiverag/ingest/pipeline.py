from pathlib import Path

from .loader import DocumentLoader
from .chunker import BaseChunker
from .embedder import Embedder
from .models import Document
from ..retrieve.vector_store import VectorStore, StoredChunk


class IngestPipeline:
    """Orchestrates: load → chunk → embed → store."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: BaseChunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self, source_path: str) -> dict:
        """Process all documents in a directory and index them."""
        # Fix 1: use load_directory — it handles unsupported files gracefully
        documents = self.loader.load_directory(source_path)

        all_vector_chunks = []

        for doc in documents:
            # Fix 2: chunker expects (text, doc_id, metadata), not a Document object
            doc_id = doc.metadata.get("source", "unknown")
            chunks = self.chunker.chunk(doc.text, doc_id, doc.metadata)

            # Fix 3: embed in batch, then build VectorChunk objects
            #         that carry the embedding inside them
            texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_batch(texts)

            for chunk, embedding in zip(chunks, embeddings):
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
                            },
                        ))

        # Store everything in one batch
        self.vector_store.add(all_vector_chunks)

        return {
            "files_processed": len(set(
                c.metadata.get("source", "") for c in all_vector_chunks
            )),
            "total_chunks": len(all_vector_chunks),
        }