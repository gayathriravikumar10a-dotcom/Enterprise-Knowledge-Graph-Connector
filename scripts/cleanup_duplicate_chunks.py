
import sys
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vector_store.chroma_store import ChromaVectorStore


def main():

    store = ChromaVectorStore(
        collection_name="enterprise_documents"
    )

    try:
        collection = store.collection

        records = collection.get(
            include=["documents", "metadatas"]
        )

        ids = records["ids"]
        documents = records["documents"]
        metadatas = records["metadatas"]

        groups = defaultdict(list)

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas
        ):
            metadata = metadata or {}

            source = (
                metadata.get("original_filename")
                or metadata.get("source")
                or "unknown"
            )

            # Preserve separate chunk positions.
            match = re.search(r"_chunk_(\d+)$", chunk_id)

            if not match:
                continue

            chunk_index = match.group(1)

            key = (
                source,
                chunk_index,
                document.strip()
            )

            groups[key].append(chunk_id)

        duplicate_ids = []

        print("\nDUPLICATE CLEANUP")
        print("=" * 55)

        for (source, chunk_index, document), chunk_ids in groups.items():

            if len(chunk_ids) <= 1:
                continue

            print(f"\nDocument: {source}")
            print(f"Chunk index: {chunk_index}")
            print(f"Keeping: {chunk_ids[0]}")

            for duplicate_id in chunk_ids[1:]:
                print(f"Deleting: {duplicate_id}")
                duplicate_ids.append(duplicate_id)

        if not duplicate_ids:
            print("\nNo duplicate chunks found.")
            return

        print("\n" + "=" * 55)
        print(f"Total records before cleanup: {len(ids)}")
        print(f"Duplicate records to delete: {len(duplicate_ids)}")

        confirmation = input(
            "\nType DELETE DUPLICATES to confirm: "
        )

        if confirmation != "DELETE DUPLICATES":
            print("\nCleanup cancelled. No records were deleted.")
            return

        collection.delete(ids=duplicate_ids)

        remaining_count = collection.count()

        print("\nCLEANUP COMPLETED")
        print("=" * 55)
        print(f"Deleted records: {len(duplicate_ids)}")
        print(f"Remaining records: {remaining_count}")

    finally:
        if hasattr(store, "close"):
            store.close()


if __name__ == "__main__":
    main()
