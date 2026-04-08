from dataclasses import dataclass, field
from typing import Any
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


@dataclass
class Chunk:
    """A single chunk of text with metadata tracing it back to its source."""
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into chunks. Every subclass must implement this."""
        ...

class FixedChunker(BaseChunker):
    """Splits text into fixed-size character chunks with overlap."""

    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(
                text=text[start:end],
                doc_id=doc_id,
                chunk_index=len(chunks),
                metadata=metadata,
                start_char=start,
                end_char=end,
            ))
            start += step

        return chunks
    
class RecursiveChunker(BaseChunker):
    """Splits text using a hierarchy of separators, preserving natural boundaries."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 separators: list[str] | None = None):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        metadata = metadata or {}
        pieces = self._recursive_split(text, self.separators)
        return self._merge_pieces(pieces, doc_id, metadata)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text, trying each separator in order."""
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Last resort: hard cut at chunk_size
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        splits = text.split(separator)
        results = []
        for split in splits:
            if len(split) <= self.chunk_size:
                results.append(split)
            elif remaining_separators:
                results.extend(self._recursive_split(split, remaining_separators))
            else:
                results.extend(self._recursive_split(split, [""]))

        return results

    def _merge_pieces(self, pieces: list[str], doc_id: str,
                      metadata: dict[str, Any]) -> list[Chunk]:
        """Merge small pieces back together up to chunk_size."""
        chunks = []
        current = ""
        char_offset = 0

        for piece in pieces:
            if len(current) + len(piece) > self.chunk_size and current:
                chunks.append(Chunk(
                    text=current.strip(),
                    doc_id=doc_id,
                    chunk_index=len(chunks),
                    metadata=metadata,
                    start_char=char_offset,
                    end_char=char_offset + len(current),
                ))
                # Step back by overlap amount for the next chunk
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                char_offset += len(current) - len(overlap_text)
                current = overlap_text

            current += piece

        if current.strip():
            chunks.append(Chunk(
                text=current.strip(),
                doc_id=doc_id,
                chunk_index=len(chunks),
                metadata=metadata,
                start_char=char_offset,
                end_char=char_offset + len(current),
            ))

        return chunks
    


class SemanticChunker(BaseChunker):
    """Splits text at topic boundaries detected by embedding similarity."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
                 similarity_threshold: float = 0.5):
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        if not self.embedding_fn:
            raise ValueError("SemanticChunker requires an embedding_fn")

        metadata = metadata or {}
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [Chunk(text=text, doc_id=doc_id, chunk_index=0,
                          metadata=metadata, start_char=0, end_char=len(text))]

        embeddings = self.embedding_fn(sentences)
        breakpoints = self._find_breakpoints(embeddings)
        return self._build_chunks(sentences, breakpoints, doc_id, metadata)

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting on period, question mark, exclamation."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_breakpoints(self, embeddings: list[list[float]]) -> list[int]:
        """Find indices where topic shifts occur."""
        breakpoints = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.similarity_threshold:
                breakpoints.append(i + 1)
        return breakpoints

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        vec_a, vec_b = np.array(a), np.array(b)
        return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10))

    def _build_chunks(self, sentences: list[str], breakpoints: list[int],
                      doc_id: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Group sentences between breakpoints into chunks."""
        chunks = []
        boundaries = [0] + breakpoints + [len(sentences)]

        char_offset = 0
        for i in range(len(boundaries) - 1):
            group = sentences[boundaries[i]:boundaries[i + 1]]
            text = " ".join(group)
            chunks.append(Chunk(
                text=text,
                doc_id=doc_id,
                chunk_index=len(chunks),
                metadata=metadata,
                start_char=char_offset,
                end_char=char_offset + len(text),
            ))
            char_offset += len(text) + 1  # +1 for the space between groups

        return chunks
    
def get_chunker(config: dict[str, Any]) -> BaseChunker:
    """Create a chunker based on config. This is the strategy pattern wiring."""
    strategy = config.get("chunking", {}).get("strategy", "recursive")
    chunk_size = config.get("chunking", {}).get("chunk_size", 500)
    chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 50)

    if strategy == "fixed":
        return FixedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "semantic":
        embedding_fn = config.get("chunking", {}).get("embedding_fn")
        threshold = config.get("chunking", {}).get("similarity_threshold", 0.5)
        return SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_fn=embedding_fn,
            similarity_threshold=threshold,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")