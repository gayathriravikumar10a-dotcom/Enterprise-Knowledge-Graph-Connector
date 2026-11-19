"""Predictable word-based chunks with overlap and source offsets."""


def chunk_spans(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Return bounded chunks with exclusive offsets into text.split()."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({"text": " ".join(words[start:end]), "word_start": start, "word_end": end})
        if end == len(words):
            break
        start = end - overlap
    return chunks


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Split into overlapping strings, retaining the existing public interface."""
    return [chunk["text"] for chunk in chunk_spans(text, chunk_size, overlap)]
