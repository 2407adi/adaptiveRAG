import pytest
from pathlib import Path
from adaptiverag.ingest import DocumentLoader, UnsupportedFileType

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
        assert docs[0].metadata["filetype"] == "csv"
        assert docs[0].metadata["row_count"] > 0
        # structure, not contents: one "Header: value | ..." line per data row,
        # so the test survives sample.csv being swapped for a different file
        assert len(docs[0].text.splitlines()) == docs[0].metadata["row_count"]

    def test_load_html(self, loader):
        docs = loader.load_file(SAMPLE_DIR / "sample.html")
        assert len(docs) == 1
        assert "Hello World" in docs[0].text
        assert "<h1>" not in docs[0].text  # tags stripped
        assert docs[0].metadata["title"] == "Sample Page"

    def test_load_pdf(self, loader):
        docs = loader.load_file(SAMPLE_DIR / "BaselThree.pdf")
        # structure, not contents. NOTE: the extractor SKIPS empty pages
        # (extractors.py: `if text.strip()`), so page numbers may have gaps —
        # assert they only ever increase, never that they're consecutive.
        pages = [d.metadata["page"] for d in docs]
        assert pages[0] == 1
        assert pages == sorted(pages)                        # ascending order
        assert len(docs) <= docs[0].metadata["total_pages"]  # skips allowed
        assert all(d.text.strip() for d in docs)             # no empty docs survive

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

    def test_unsupported_file_type(self, loader, tmp_path):
        # the file must EXIST — otherwise the loader's file-not-found check
        # fires first and we never reach the unsupported-type check under test
        bad_file = tmp_path / "fake.xyz"
        bad_file.write_text("some content")
        with pytest.raises(UnsupportedFileType) as exc_info:
            loader.load_file(bad_file)
        assert ".xyz" in str(exc_info.value)

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_file("nonexistent.pdf")

    def test_not_a_directory(self, loader):
        with pytest.raises(NotADirectoryError):
            loader.load_directory("not_a_dir")