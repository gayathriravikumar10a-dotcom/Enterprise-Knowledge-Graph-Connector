
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
    return (
        metadata.get("original_filename")
        or metadata.get("source")
    )


def precision_at_k(results, relevant_filenames, k=K):
    """
    Chunk-level Precision@K.
    A retrieved chunk is relevant when its source filename
    belongs to the relevant filename set.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant_count = sum(
        1
        for item in results[:k]
        if get_filename(item) in relevant_filenames
    )

    return relevant_count / k


def recall_at_k(results, relevant_filenames, k=K):
    """
    Document-level Recall@K.
    Measures how many relevant documents appear among
    the top K retrieved chunks.
    """
    if not relevant_filenames:
        return 0.0

    retrieved_filenames = {
        get_filename(item)
        for item in results[:k]
        if get_filename(item)
    }

    return (
        len(retrieved_filenames & relevant_filenames)
        / len(relevant_filenames)
    )


def reciprocal_rank(results, relevant_filenames):
    """Return the reciprocal rank of the first relevant chunk."""
    for rank, item in enumerate(results, start=1):
        if get_filename(item) in relevant_filenames:
            return 1.0 / rank

    return 0.0


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("Evaluation dataset must contain test cases.")

    for index, case in enumerate(test_cases, start=1):
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("query"), str)
            or not case["query"].strip()
            or not isinstance(case.get("relevant_filenames"), list)
            or not case["relevant_filenames"]
        ):
            raise ValueError(
                f"Invalid evaluation test case at index {index}."
            )

    retriever = HybridRetriever()
    evaluation_results = []
    latencies = []

    try:
        for case in test_cases:
            query = case["query"]
            relevant = set(case["relevant_filenames"])

            start = time.perf_counter()

            results = retriever.vector_search(
                query,
                top_k=K
            )

            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            precision = precision_at_k(results, relevant)
            recall = recall_at_k(results, relevant)
            mrr = reciprocal_rank(results, relevant)

            retrieved_documents = [
                get_filename(item)
                for item in results
                if get_filename(item)
            ]

            evaluation_results.append({
                "query": query,
                "retrieved_documents": retrieved_documents,
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
                "latency": latency
            })

    finally:
        retriever.close()

    count = len(evaluation_results)

    print("\n========== VECTOR RETRIEVAL EVALUATION ==========")
    print("Evaluation queries:", count)
    print("K:", K)

    for index, result in enumerate(evaluation_results, start=1):
        print(f"\nQuery {index}: {result['query']}")
        print("Retrieved chunks' documents:",
              result["retrieved_documents"])
        print("Precision@5:", round(result["precision"], 4))
        print("Recall@5:", round(result["recall"], 4))
        print("MRR:", round(result["mrr"], 4))
        print("Retrieval latency:",
              round(result["latency"], 2), "ms")

    mean_precision = sum(
        result["precision"] for result in evaluation_results
    ) / count

    mean_recall = sum(
        result["recall"] for result in evaluation_results
    ) / count

    mean_mrr = sum(
        result["mrr"] for result in evaluation_results
    ) / count

    average_latency = sum(latencies) / len(latencies)

    print("\n========== FINAL METRICS ==========")
    print("Mean Precision@5:", round(mean_precision, 4))
    print("Mean Recall@5:", round(mean_recall, 4))
    print("Mean MRR:", round(mean_mrr, 4))
    print("Average Retrieval Latency:",
          round(average_latency, 2), "ms")


if __name__ == "__main__":
    main()
