import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vector_store.chroma_store import ChromaVectorStore


def main():
    print("CHROMADB VECTOR STORE TEST")
    print("=" * 45)

    store = ChromaVectorStore(
        collection_name="enterprise_test"
    )

    sample_chunks = [
        {
            "chunk_id": "test_001",
            "text": "Asha works in the Artificial Intelligence department.",
            "metadata": {"source": "company_info.txt"}
        },
        {
            "chunk_id": "test_002",
            "text": "Ravi is responsible for managing Project Atlas.",
            "metadata": {"source": "company_info.txt"}
        },
        {
            "chunk_id": "test_003",
            "text": "Project Atlas is an enterprise AI knowledge management initiative.",
            "metadata": {"source": "company_info.txt"}
        },
        {
            "chunk_id": "test_004",
            "text": "The engineering team develops software applications.",
            "metadata": {"source": "company_info.txt"}
        }
    ]

    inserted = store.add_documents(sample_chunks)

    print(f"\nDocuments inserted: {inserted}")
    print(f"Total documents: {store.count()}")

    query = "Who is responsible for Project Atlas?"

    print(f"\nSEARCH QUERY: {query}")
    print("-" * 45)

    results = store.search(query, top_k=3)

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Text: {result['text']}")
        print(f"Distance: {result['distance']:.4f}")

    store.client.delete_collection(name="enterprise_test")

    print("\nVECTOR STORE TEST COMPLETED")


if __name__ == "__main__":
    main()
