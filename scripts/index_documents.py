import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.workflow import build_knowledge_graph_from_document
from vector_store.chroma_store import ChromaVectorStore


DOCUMENT_PATH = (
    PROJECT_ROOT / "data" / "sample_documents" / "company_info.txt"
)


def main():
    print("ENTERPRISE DOCUMENT VECTOR INDEXING")
    print("=" * 50)

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Document not found: {DOCUMENT_PATH}"
        )

    print(f"\nProcessing: {DOCUMENT_PATH.name}")

    # Reuse the existing document ingestion pipeline
    result = build_knowledge_graph_from_document(
        str(DOCUMENT_PATH)
    )

    chunks = result["chunks"]

    print(f"Document chunks generated: {len(chunks)}")

    # Store actual document chunks in ChromaDB
    vector_store = ChromaVectorStore(
        collection_name="enterprise_documents"
    )

    inserted = vector_store.add_documents(chunks)

    print(f"\nChunks indexed: {inserted}")
    print(f"Total vectors in collection: {vector_store.count()}")

    print("\nVECTOR SEARCH TEST")
    print("-" * 50)

    query = "Which employee manages Project Atlas?"

    print(f"Query: {query}")

    results = vector_store.search(query, top_k=3)

    for index, item in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Content: {item['text']}")
        print(f"Distance: {item['distance']:.4f}")

    print("\nDOCUMENT INDEXING COMPLETED")


if __name__ == "__main__":
    main()
