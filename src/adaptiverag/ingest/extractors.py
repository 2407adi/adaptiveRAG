from pathlib import Path
from .models import Document
import csv
import io

from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document as DocxDocument


def extract_text(filepath: Path) -> list[Document]:
    """Extract from .txt files."""
    text = filepath.read_text(encoding="utf-8")
    return [Document(
        text=text,
        metadata={"source": str(filepath), "filetype": "txt"},
    )]


def extract_markdown(filepath: Path) -> list[Document]:
    """Extract from .md files."""
    text = filepath.read_text(encoding="utf-8")
    return [Document(
        text=text,
        metadata={"source": str(filepath), "filetype": "md"},
    )]


def extract_csv(filepath: Path) -> list[Document]:
    """Extract from .csv files. Each row becomes readable text."""
    text = filepath.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text))

    rows = []
    for i, row in enumerate(reader, start=1):
        row_text = " | ".join(f"{k}: {v}" for k, v in row.items())
        rows.append(row_text)

    return [Document(
        text="\n".join(rows),
        metadata={
            "source": str(filepath),
            "filetype": "csv",
            "row_count": len(rows),
        },
    )]


def extract_html(filepath: Path) -> list[Document]:
    """Extract visible text from .html files, stripping tags."""
    raw = filepath.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return [Document(
        text=text,
        metadata={
            "source": str(filepath),
            "filetype": "html",
            "title": soup.title.string if soup.title else None,
        },
    )]


def extract_pdf(filepath: Path) -> list[Document]:
    """Extract from .pdf files. One Document per page."""
    reader = PdfReader(str(filepath))
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():  # skip blank pages
            documents.append(Document(
                text=text,
                metadata={
                    "source": str(filepath),
                    "filetype": "pdf",
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                },
            ))

    return documents


def extract_docx(filepath: Path) -> list[Document]:
    """Extract from .docx files."""
    doc = DocxDocument(str(filepath))

    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)

    return [Document(
        text="\n\n".join(paragraphs),
        metadata={
            "source": str(filepath),
            "filetype": "docx",
            "paragraph_count": len(paragraphs),
        },
    )]