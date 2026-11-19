"""Validated document loading shared by ingestion and graph workflows."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import unicodedata

import pymupdf
from docx import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024


def clean_text(text: str) -> str:
    """Normalize Unicode and whitespace while removing extraction artifacts."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        char for char in text
        if char.isspace() or unicodedata.category(char) not in {"Cc", "Cf"}
    )
    return " ".join(text.split())


def load_pdf(file_path: Path) -> str:
    """Extract selectable text; scanned images require a separate OCR stage."""
    with pymupdf.open(file_path) as pdf:
        if pdf.needs_pass:
            raise ValueError("Password-protected PDFs are not supported")
        return "\n".join(page.get_text() for page in pdf)


def load_docx(file_path: Path) -> str:
    document = Document(file_path)
    # iter_inner_content preserves the order of paragraphs and tables.
    parts = []
    for item in document.iter_inner_content():
        if hasattr(item, "rows"):
            parts.extend(" | ".join(cell.text for cell in row.cells) for row in item.rows)
        else:
            parts.append(item.text)
    return "\n".join(parts)


def load_txt(file_path: Path) -> str:
    data = file_path.read_bytes()
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    text = data.decode(encoding)
    if any(unicodedata.category(c) == "Cc" and not c.isspace() for c in text):
        raise ValueError("TXT document contains binary or control characters")
    return text


def load_document(file_path: str, *, max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> dict:
    """Return cleaned text and metadata; reject invalid input before ingestion.

    max_file_size is a byte limit, defaulting to the API's 10 MiB limit.
    TXT supports UTF-8 (with or without BOM) and BOM-marked UTF-16.
    """
    if isinstance(max_file_size, bool) or not isinstance(max_file_size, int) or max_file_size <= 0:
        raise ValueError("max_file_size must be a positive integer")
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    stat = path.stat()
    if stat.st_size == 0:
        raise ValueError(f"Document is empty: {path.name}")
    if stat.st_size > max_file_size:
        raise ValueError(f"Document exceeds the {max_file_size} byte size limit: {path.name}")
    try:
        loader = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_txt}[extension]
        text = clean_text(loader(path))
    except Exception as error:
        raise ValueError(f"Unable to read {extension[1:].upper()} document {path.name}: {error}") from error
    if not text:
        raise ValueError(f"No readable text found in: {path.name}")
    with path.open("rb") as stream:
        file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "text": text,
        "metadata": {
            "source": path.name,
            "file_type": extension[1:],
            "file_path": str(path.resolve()),
            "file_size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "file_hash": file_hash,
            "content_hash": content_hash,
            "document_id": content_hash,
            "word_count": len(text.split()),
            "character_count": len(text),
        },
    }
