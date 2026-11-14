from ingestion.document_loader import load_document
from ingestion.chunker import chunk_text


def process_document(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> dict:
    """Load a document and convert it into structured chunks."""

    document = load_document(file_path)

    chunks = chunk_text(
        document["text"],
        chunk_size=chunk_size,
        overlap=overlap
    )

    processed_chunks = []

    for index, chunk in enumerate(chunks):
        processed_chunks.append({
            "chunk_id": f"{document['metadata']['source']}_chunk_{index}",
            "text": chunk,
            "metadata": {
                **document["metadata"],
                "chunk_index": index
            }
        })

    return {
        "source": document["metadata"]["source"],
        "total_chunks": len(processed_chunks),
        "chunks": processed_chunks
    }
