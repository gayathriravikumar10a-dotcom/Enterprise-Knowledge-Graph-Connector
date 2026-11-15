
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_retriever import HybridRetriever


def main():
    retriever = HybridRetriever()

    try:
        query = input("Enter a question: ").strip()

        results = retriever.retrieve(
            query,
            vector_top_k=5,
            graph_limit=5
        )

        print("\nVECTOR RESULTS")

        for index, item in enumerate(
            results["vector_results"], start=1
        ):
            metadata = item.get("metadata") or {}

            print(f"\nResult {index}")
            print("Chunk ID:", item.get("chunk_id"))
            print("Document ID:", metadata.get("document_id"))
            print(
                "Filename:",
                metadata.get("original_filename", metadata.get("source"))
            )
            print("Distance:", item.get("distance"))
            print("Text:", item.get("text", "")[:400])

        print("\nGRAPH RESULTS:", results["graph_result_count"])

    finally:
        retriever.close()


if __name__ == "__main__":
    main()

