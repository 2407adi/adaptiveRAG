"""src/adaptiverag/retrieve/vector_store.py — Base interface"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class StoredChunk:
    """A document chunk with its embedding, ready for vector storage."""
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result with similarity score."""
    chunk_id: str
    text: str
    score: float  # normalized: higher = more similar
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    def add(self, chunks: list[StoredChunk]) -> None:
        """Index a batch of chunks."""
        ...

    @abstractmethod
    def search(
        self, query_vector: list[float], k: int = 5
    ) -> list[SearchResult]:
        """Find k most similar chunks by vector."""
        ...

    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
    ) -> list[SearchResult]:
        """Convenience: embed text then search."""
        if embed_fn is None:
            raise ValueError("embed_fn required for text search")
        vector = embed_fn(query_text)
        return self.search(vector, k)

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove chunks by ID."""
        ...

    @abstractmethod
    def persist(self, path: str) -> None:
        """Save index to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> "VectorStore":
        """Reload a persisted index."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Number of indexed chunks."""
        ...


"""ChromaStore — append this to vector_store.py"""

import chromadb
from typing import cast

class ChromaStore(VectorStore):
    """ChromaDB-backed vector store. Zero-config, local."""

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
    ):
        if persist_directory:
            self._client = chromadb.PersistentClient(
                path=persist_directory
            )
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        self._persist_directory = persist_directory

    def add(self, chunks: list[StoredChunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata or {} for c in chunks],
        )

    def search(
        self, query_vector: list[float], k: int = 5
    ) -> list[SearchResult]:
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self._collection.count()),
            include=["distances", "documents", "metadatas"],
        )
        if not results["ids"][0]:
            return []

        distances = cast(list[list[float]], results["distances"])
        documents = cast(list[list[str]], results["documents"])
        metadatas = cast(list[list[dict]], results["metadatas"])

        search_results = []
        for i, chunk_id in enumerate(results["ids"][0]):
            distance = distances[0][i]
            score = 1.0 - distance
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=documents[0][i],
                    score=score,
                    metadata=metadatas[0][i] or {},
                )
            )
        return search_results

    def delete(self, ids: list[str]) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def persist(self, path: str) -> None:
        # PersistentClient auto-persists; this is a no-op if
        # already using persist_directory. For EphemeralClient,
        # you'd need to recreate with PersistentClient.
        pass

    @classmethod
    def load(cls, path: str, **kwargs) -> "ChromaStore":
        collection_name = kwargs.get("collection_name", "default")
        return cls(
            collection_name=collection_name,
            persist_directory=path,
        )

    def count(self) -> int:
        return self._collection.count()
    

"""FAISSStore — append this to vector_store.py"""

import json
from pathlib import Path

import faiss
import numpy as np


class FAISSStore(VectorStore):
    """FAISS-backed vector store. Fast, low-level."""

    def __init__(self, dimension: int):
        self._dimension = dimension
        # IndexFlatIP = inner product (cosine sim on normalized vectors)
        inner_index = faiss.IndexFlatIP(dimension)
        # IDMap lets us use our own int64 IDs
        self._index = faiss.IndexIDMap(inner_index)

        # FAISS only stores vectors — we track docs separately
        self._id_counter: int = 0
        self._str_to_int: dict[str, int] = {}   # "chunk-42" -> 7
        self._int_to_str: dict[int, str] = {}   # 7 -> "chunk-42"
        self._documents: dict[str, str] = {}     # id -> text
        self._metadatas: dict[str, dict] = {}    # id -> metadata

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize so inner product == cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid div by zero
        return vectors / norms

    def _get_int_id(self, str_id: str) -> int:
        """Map string ID to a unique int64 for FAISS."""
        if str_id not in self._str_to_int:
            self._str_to_int[str_id] = self._id_counter
            self._int_to_str[self._id_counter] = str_id
            self._id_counter += 1
        return self._str_to_int[str_id]

    def add(self, chunks: list[StoredChunk]) -> None:
        if not chunks:
            return

        vectors = np.array(
            [c.embedding for c in chunks], dtype=np.float32
        )
        vectors = self._normalize(vectors)

        int_ids = np.array(
            [self._get_int_id(c.id) for c in chunks], dtype=np.int64
        )

        self._index.add_with_ids(vectors, int_ids)  # type: ignore[call-arg]

        for c in chunks:
            self._documents[c.id] = c.text
            self._metadatas[c.id] = c.metadata or {}

    def search(
        self, query_vector: list[float], k: int = 5
    ) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []

        query = np.array([query_vector], dtype=np.float32)
        query = self._normalize(query)

        k = min(k, self._index.ntotal)
        scores, int_ids = self._index.search(query, k)  # type: ignore[call-arg]

        results = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id == -1:  # FAISS returns -1 for empty slots
                continue
            str_id = self._int_to_str[int(int_id)]
            results.append(
                SearchResult(
                    chunk_id=str_id,
                    text=self._documents[str_id],
                    score=float(score),  # already cosine sim
                    metadata=self._metadatas.get(str_id, {}),
                )
            )
        return results

    def delete(self, ids: list[str]) -> None:
        int_ids = []
        for str_id in ids:
            if str_id in self._str_to_int:
                int_ids.append(self._str_to_int[str_id])
                # Clean up mappings
                int_id = self._str_to_int.pop(str_id)
                self._int_to_str.pop(int_id, None)
                self._documents.pop(str_id, None)
                self._metadatas.pop(str_id, None)

        if int_ids:
            id_array = np.array(int_ids, dtype=np.int64)
            self._index.remove_ids(id_array)

    def persist(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(p / "index.faiss"))

        metadata = {
            "dimension": self._dimension,
            "id_counter": self._id_counter,
            "str_to_int": self._str_to_int,
            "int_to_str": {
                str(k): v for k, v in self._int_to_str.items()
            },
            "documents": self._documents,
            "metadatas": self._metadatas,
        }
        with open(p / "metadata.json", "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str, **kwargs) -> "FAISSStore":
        p = Path(path)

        with open(p / "metadata.json") as f:
            metadata = json.load(f)

        store = cls(dimension=metadata["dimension"])
        store._index = faiss.read_index(str(p / "index.faiss"))
        store._id_counter = metadata["id_counter"]
        store._str_to_int = metadata["str_to_int"]
        store._int_to_str = {
            int(k): v for k, v in metadata["int_to_str"].items()
        }
        store._documents = metadata["documents"]
        store._metadatas = metadata["metadatas"]
        return store

    def count(self) -> int:
        return self._index.ntotal
    
"""Factory — append to vector_store.py"""


def create_vector_store(
    backend: str = "chroma", **kwargs
) -> VectorStore:
    """Create a vector store from config.

    Args:
        backend: 'chroma' or 'faiss'
        **kwargs: passed to the backend constructor
            chroma: collection_name, persist_directory
            faiss: dimension (required)
    """
    if backend == "chroma":
        return ChromaStore(**kwargs)
    elif backend == "faiss":
        if "dimension" not in kwargs:
            raise ValueError("FAISSStore requires 'dimension'")
        return FAISSStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
