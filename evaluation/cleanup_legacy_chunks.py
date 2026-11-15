
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vector_store.chroma_store import ChromaVectorStore


def main():
    store = ChromaVectorStore()

    try:
        results = store.collection.get(
            include=["metadatas", "documents"]
        )

        legacy_ids = []

        for index, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][index] or {}

            if (
                not metadata.get("document_id")
                and metadata.get("source") == "enterprise_document"
            ):
                legacy_ids.append(chunk_id)

        print("Legacy chunks found:", len(legacy_ids))
        print("Legacy chunk IDs:", legacy_ids)

        if "--delete" not in sys.argv:
            print("\nDRY RUN: Nothing was deleted.")
            print("If these are the expected legacy chunks, run:")
            print("python evaluation/cleanup_legacy_chunks.py --delete")
            return

        if legacy_ids:
            store.collection.delete(ids=legacy_ids)

        print("\nDeleted legacy chunks:", len(legacy_ids))
        print("Remaining Chroma chunks:", store.count())

    finally:
        store.close()


if __name__ == "__main__":
    main()
