import pytest
from adaptiverag.ingest.chunker import (
    Chunk, FixedChunker, RecursiveChunker, SemanticChunker, get_chunker
)


def make_text(word_count: int) -> str:
    """Generate dummy text of approximately word_count words."""
    sentence = "The quick brown fox jumps over the lazy dog. "
    repeats = word_count // 9 + 1
    return (sentence * repeats).strip()


class TestFixedChunker:
    def test_chunk_count(self):
        text = make_text(5000)
        chunker = FixedChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(text, doc_id="doc_1")
        # Each step is 450 chars, so roughly len(text)/450 chunks
        assert 10 <= len(chunks) <= 120  # broad range, depends on char count

    def test_overlap_exists(self):
        text = make_text(500)
        chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(text, doc_id="doc_1")
        # End of chunk 0 should overlap with start of chunk 1
        assert chunks[0].text[-20:] == chunks[1].text[:20]

    def test_metadata_preserved(self):
        text = make_text(200)
        meta = {"source": "report.pdf", "author": "Aditya"}
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text, doc_id="doc_1", metadata=meta)
        for c in chunks:
            assert c.metadata["source"] == "report.pdf"
            assert c.doc_id == "doc_1"

    def test_chunk_index_sequential(self):
        text = make_text(500)
        chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text, doc_id="doc_1")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


class TestRecursiveChunker:
    def test_respects_paragraph_boundaries(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=0)
        chunks = chunker.chunk(text, doc_id="doc_1")
        # Entire text fits in one chunk since it's under 500 chars
        assert len(chunks) == 1

    def test_splits_large_text(self):
        text = make_text(5000)
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(text, doc_id="doc_1")
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.text) <= 550  # some tolerance

    def test_metadata_preserved(self):
        text = make_text(500)
        meta = {"source": "notes.txt"}
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text, doc_id="doc_2", metadata=meta)
        for c in chunks:
            assert c.metadata["source"] == "notes.txt"


class TestSemanticChunker:
    def fake_embedding_fn(self, texts: list[str]) -> list[list[float]]:
        """Return fake embeddings — similar for all except index 2."""
        embeddings = []
        for i, _ in enumerate(texts):
            if i == 2:
                embeddings.append([0.0, 1.0, 0.0])  # different topic
            else:
                embeddings.append([1.0, 0.0, 0.0])  # same topic
        return embeddings

    def test_splits_at_topic_boundary(self):
        text = "Sentence one. Sentence two. Totally different topic. Back to normal. More normal."
        chunker = SemanticChunker(
            embedding_fn=self.fake_embedding_fn,
            similarity_threshold=0.5,
        )
        chunks = chunker.chunk(text, doc_id="doc_1")
        assert len(chunks) >= 2  # should split at the topic shift

    def test_raises_without_embedding_fn(self):
        chunker = SemanticChunker()
        with pytest.raises(ValueError):
            chunker.chunk("some text", doc_id="doc_1")


class TestGetChunker:
    def test_returns_fixed(self):
        config = {"chunking": {"strategy": "fixed"}}
        assert isinstance(get_chunker(config), FixedChunker)

    def test_returns_recursive(self):
        config = {"chunking": {"strategy": "recursive"}}
        assert isinstance(get_chunker(config), RecursiveChunker)

    def test_default_is_recursive(self):
        assert isinstance(get_chunker({}), RecursiveChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            get_chunker({"chunking": {"strategy": "unknown"}})