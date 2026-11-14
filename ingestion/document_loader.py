from pathlib import Path
import pymupdf
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted text."""
    return " ".join(text.split())


def load_pdf(file_path: Path) -> str:
    """Extract text from a PDF file."""
    text_parts = []

    with pymupdf.open(file_path) as pdf:
        for page in pdf:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


def load_docx(file_path: Path) -> str:
    """Extract paragraphs and table content from a DOCX file."""
    document = Document(file_path)

    paragraphs = [p.text for p in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            paragraphs.append(" | ".join(cell.text for cell in row.cells))

    return "\n".join(paragraphs)


def load_txt(file_path: Path) -> str:
    """Read a UTF-8 text file."""
    return file_path.read_text(encoding="utf-8-sig")


def load_document(file_path: str) -> dict:
    """Load a supported document and return its text and metadata."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        text = load_pdf(path)
    elif extension == ".docx":
        text = load_docx(path)
    else:
        text = load_txt(path)

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError(f"No readable text found in: {path.name}")

    return {
        "text": cleaned_text,
        "metadata": {
            "source": path.name,
            "file_type": extension.lstrip("."),
            "file_path": str(path.resolve())
        }
    }

