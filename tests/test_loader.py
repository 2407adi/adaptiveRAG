import pytest
from pathlib import Path
from adaptiverag.ingest import Document, DocumentLoader, UnsupportedFileType

SAMPLE_DIR = Path("data/sample")


@pytest.fixture
def loader():
    return DocumentLoader()


class TestLoadFile:
    """Test individual file loading."""

    def test_load_txt(self, loader):
        docs = loader.load_file(SAMPLE_DIR / "Basel_Introduction.txt")
        assert len(docs) == 1
        assert "This document, together with" in docs[0].text
        assert docs[0].metadata["source"].endswith("Basel_Introduction.txt")
        assert docs[0].metadata["filetype"] == "txt"

    def test_load_markdown(self, loader):
        docs = loader.load_file(SAMPLE_DIR / "sample.md")
        assert len(docs) == 1
        assert "# Sample Markdown" in docs[0].text
        assert docs[0].metadata["filetype"] == "md"

    def test_load_csv(self, loader):
        docs = loader.load_file(SAMPLE_DIR / "sample.csv")
        assert len(docs) == 1
        assert "Alice" in docs[0].text
        assert "name:" in docs[0].text  # key:value format
        assert docs[0].metadata["row_count"] == 3

    def test_load_html(self, loader):
        docs = loader.load_file(SAMPLE_DIR / "sample.html")
        assert len(docs) == 1
        assert "Hello World" in docs[0].text
        assert "<h1>" not in docs[0].text  # tags stripped
        assert docs[0].metadata["title"] == "Sample Page"

    def test_load_pdf(self, loader):
        docs = loader.load_file(SAMPLE_DIR / "BaselThree.pdf")
        assert "This document" in docs[0].text
        assert docs[0].metadata["page"] == 1
        assert docs[1].metadata["page"] == 2

    def test_load_docx(self, loader):
        docs = loader.load_file(SAMPLE_DIR / "Basel_Introduction.docx")
        assert len(docs) == 1
        assert "This document" in docs[0].text


class TestLoadDirectory:
    """Test batch loading."""

    def test_loads_all_supported(self, loader):
        docs = loader.load_directory(SAMPLE_DIR)
        # At least one doc from each of the 6 file types
        assert len(docs) >= 6

        sources = [d.metadata["source"] for d in docs]
        for ext in ["txt", "md", "csv", "html", "pdf", "docx"]:
            assert any(s.endswith(f".{ext}") for s in sources), \
                f"No document from .{ext} file"

    def test_all_docs_have_text(self, loader):
        docs = loader.load_directory(SAMPLE_DIR)
        for doc in docs:
            assert isinstance(doc.text, str)
            assert len(doc.text.strip()) > 0

    def test_all_docs_have_source(self, loader):
        docs = loader.load_directory(SAMPLE_DIR)
        for doc in docs:
            assert "source" in doc.metadata


class TestErrorHandling:
    """Test edge cases."""

    def test_unsupported_file_type(self, loader):
        with pytest.raises(UnsupportedFileType) as exc_info:
            loader.load_file("fake.xyz")
        assert ".xyz" in str(exc_info.value)

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_file("nonexistent.pdf")

    def test_not_a_directory(self, loader):
        with pytest.raises(NotADirectoryError):
            loader.load_directory("not_a_dir")