
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_retriever import HybridRetriever

DATASET_PATH = PROJECT_ROOT / "evaluation" / "retrieval_dataset.json"
K = 5


def get_filename(result):
    metadata = result.get("metadata") or {}
    return metadata.get("original_filename") or metadata.get("source")


def recall_at_k(retrieved, relevant):
    if not relevant:
        return 0.0
    return len(set(retrieved[:K]) & relevant) / len(relevant)


def document_precision_at_k(retrieved, relevant):
    unique_retrieved = list(dict.fromkeys(retrieved[:K]))

    if not unique_retrieved:
        return 0.0

    return len(set(unique_retrieved) & relevant) / len(unique_retrieved)


def reciprocal_rank(retrieved, relevant):
    for rank, filename in enumerate(retrieved, start=1):
        if filename in relevant:
            return 1.0 / rank
    return 0.0


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    retriever = HybridRetriever()
    all_results = []
    latencies = []

    try:
        for case in test_cases:
            start = time.perf_counter()

            results = retriever.vector_search(
                case["query"],
                top_k=K
            )

            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            retrieved = []
            for item in results:
                filename = get_filename(item)

                if filename and filename not in retrieved:
                    retrieved.append(filename)

            relevant = set(case["relevant_filenames"])

            all_results.append({
                "query": case["query"],
                "retrieved": retrieved,
                "recall": recall_at_k(retrieved, relevant),
                "precision": document_precision_at_k(retrieved, relevant),
                "mrr": reciprocal_rank(retrieved, relevant),
                "latency": latency
            })

    finally:
        retriever.close()

    count = len(all_results)

    if not count:
        print("No evaluation queries found.")
        return

    print("\n========== VECTOR RETRIEVAL EVALUATION ==========")

    for index, result in enumerate(all_results, start=1):
        print(f"\nQuery {index}: {result['query']}")
        print("Retrieved documents:", result["retrieved"])
        print("Recall@5:", round(result["recall"], 4))
        print("Document Precision@5:", round(result["precision"], 4))
        print("Reciprocal Rank:", round(result["mrr"], 4))
        print("Latency:", round(result["latency"], 2), "ms")

    print("\n========== FINAL METRICS ==========")
    print("Mean Recall@5:", round(sum(r["recall"] for r in all_results) / count, 4))
    print("Mean Document Precision@5:", round(sum(r["precision"] for r in all_results) / count, 4))
    print("MRR:", round(sum(r["mrr"] for r in all_results) / count, 4))
    print("Average Latency:", round(sum(latencies) / len(latencies), 2), "ms")


if __name__ == "__main__":
    main()
