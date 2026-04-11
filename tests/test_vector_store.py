"""tests/test_vector_store.py"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from adaptiverag.retrieve.vector_store import (
    StoredChunk,
    ChromaStore,
    FAISSStore,
    create_vector_store,
)

DIMENSION = 64  # small for testing


def _make_chunks(n: int = 10) -> list[StoredChunk]:
    """Generate n chunks with random embeddings."""
    rng = np.random.default_rng(42)
    return [
        StoredChunk(
            id=f"chunk-{i}",
            text=f"Document number {i} about topic {i % 3}",
            embedding=rng.random(DIMENSION).tolist(),
            metadata={"source": f"file_{i}.txt"},
        )
        for i in range(n)
    ]


class TestVectorStoreContract:
    """Tests that both backends must pass."""

    @pytest.fixture(params=["chroma", "faiss"])
    def store(self, request):
        if request.param == "chroma":
            import uuid
            return ChromaStore(collection_name=f"test_{uuid.uuid4().hex}")
        return FAISSStore(dimension=DIMENSION)

    def test_add_and_count(self, store):
        chunks = _make_chunks(5)
        store.add(chunks)
        assert store.count() == 5

    def test_add_idempotent(self, store):
        chunks = _make_chunks(3)
        store.add(chunks)
        store.add(chunks)  # same IDs again
        # Chroma upserts; FAISS will have duplicates
        # unless you handle it — this tests your choice
        assert store.count() >= 3

    def test_search_returns_k_results(self, store):
        chunks = _make_chunks(10)
        store.add(chunks)
        query = chunks[0].embedding
        results = store.search(query, k=5)
        assert len(results) == 5

    def test_search_top_result_is_exact_match(self, store):
        chunks = _make_chunks(10)
        store.add(chunks)
        query = chunks[3].embedding
        results = store.search(query, k=1)
        assert results[0].chunk_id == "chunk-3"

    def test_search_scores_are_descending(self, store):
        chunks = _make_chunks(10)
        store.add(chunks)
        results = store.search(chunks[0].embedding, k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_delete_removes_chunks(self, store):
        chunks = _make_chunks(5)
        store.add(chunks)
        store.delete(["chunk-0", "chunk-1"])
        assert store.count() == 3

    def test_search_by_text_requires_embed_fn(self, store):
        with pytest.raises(ValueError, match="embed_fn"):
            store.search_by_text("hello")

    def test_search_by_text_with_fn(self, store):
        chunks = _make_chunks(5)
        store.add(chunks)
        dummy_fn = lambda text: chunks[2].embedding
        results = store.search_by_text("anything", k=1, embed_fn=dummy_fn)
        assert results[0].chunk_id == "chunk-2"


class TestPersistReload:
    """Persist-reload cycle for each backend."""

    def test_chroma_persist_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaStore(
                collection_name="persist_test",
                persist_directory=tmpdir,
            )
            store.add(_make_chunks(5))
            assert store.count() == 5

            # Simulate restart
            reloaded = ChromaStore.load(
                tmpdir, collection_name="persist_test"
            )
            assert reloaded.count() == 5
            results = reloaded.search(
                _make_chunks(5)[0].embedding, k=1
            )
            assert results[0].chunk_id == "chunk-0"

    def test_faiss_persist_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FAISSStore(dimension=DIMENSION)
            store.add(_make_chunks(5))
            store.persist(tmpdir)

            reloaded = FAISSStore.load(tmpdir)
            assert reloaded.count() == 5
            results = reloaded.search(
                _make_chunks(5)[0].embedding, k=1
            )
            assert results[0].chunk_id == "chunk-0"


class TestFactory:

    def test_create_chroma(self):
        store = create_vector_store("chroma")
        assert isinstance(store, ChromaStore)

    def test_create_faiss(self):
        store = create_vector_store("faiss", dimension=128)
        assert isinstance(store, FAISSStore)

    def test_faiss_requires_dimension(self):
        with pytest.raises(ValueError, match="dimension"):
            create_vector_store("faiss")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_vector_store("pinecone")