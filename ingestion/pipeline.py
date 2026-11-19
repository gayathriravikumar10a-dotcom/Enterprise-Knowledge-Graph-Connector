from ingestion.document_loader import load_document, DEFAULT_MAX_FILE_SIZE
from ingestion.chunker import chunk_spans


def process_document(file_path: str, chunk_size: int = 500, overlap: int = 100,
                     *,
                     max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> dict:
    """Load a document and attach source metadata and offsets to every chunk."""
    document = load_document(file_path, max_file_size=max_file_size)
    spans = chunk_spans(document["text"], chunk_size, overlap)
    chunks = []
    for index, span in enumerate(spans):
        chunks.append({
            "chunk_id": f"{document['metadata']['source']}_chunk_{index}",
            "text": span["text"],
            "metadata": {**document["metadata"], "chunk_index": index,
                         "word_start": span["word_start"], "word_end": span["word_end"],
                         "chunk_strategy": "word", "total_chunks": len(spans)},
        })
    return {"source": document["metadata"]["source"], "metadata": document["metadata"],
            "total_chunks": len(chunks), "chunks": chunks}
