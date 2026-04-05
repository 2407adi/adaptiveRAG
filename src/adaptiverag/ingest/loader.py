from pathlib import Path
from .models import Document
from .exceptions import UnsupportedFileType
from .extractors import (
    extract_text,
    extract_markdown,
    extract_csv,
    extract_html,
    extract_pdf,
    extract_docx,
)


class DocumentLoader:
    """Universal document loader with extension-based dispatch."""

    EXTRACTORS = {
        ".txt": extract_text,
        ".md": extract_markdown,
        ".csv": extract_csv,
        ".html": extract_html,
        ".htm": extract_html,
        ".pdf": extract_pdf,
        ".docx": extract_docx,
    }

    def load_file(self, filepath: str | Path) -> list[Document]:
        """Load a single file and return Documents."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = filepath.suffix.lower()
        extractor = self.EXTRACTORS.get(ext)

        if extractor is None:
            raise UnsupportedFileType(str(filepath), ext)

        return extractor(filepath)

    def load_directory(self, dirpath: str | Path) -> list[Document]:
        """Load all supported files from a directory."""
        dirpath = Path(dirpath)

        if not dirpath.is_dir():
            raise NotADirectoryError(f"Not a directory: {dirpath}")

        documents = []
        skipped = []

        for filepath in sorted(dirpath.iterdir()):
            if filepath.is_file():
                try:
                    documents.extend(self.load_file(filepath))
                except UnsupportedFileType:
                    skipped.append(filepath.name)

        if skipped:
            print(f"Skipped unsupported files: {skipped}")

        return documents