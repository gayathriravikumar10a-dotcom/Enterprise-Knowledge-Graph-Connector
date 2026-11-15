
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_retriever import HybridRetriever


TEST_CASES = [
    {
        "query": "Which employee manages Project Atlas?",
        "expected": ("Ravi", "MANAGES", "Project Atlas")
    },
    {
        "query": "Who works on Project Atlas?",
        "expected": ("Asha", "WORKS_ON", "Project Atlas")
    }
]


def matches(result, expected):
    actual = (
        result.get("subject"),
        result.get("relationship"),
        result.get("object")
    )
    return actual == expected


def main():
    retriever = HybridRetriever()

    total_relevant = len(TEST_CASES)
    hits = 0
    reciprocal_ranks = []

    try:
        print("\n========== GRAPH RETRIEVAL EVALUATION ==========")

        for index, case in enumerate(TEST_CASES, start=1):
            results = retriever.retrieve(
                case["query"],
                vector_top_k=5,
                graph_limit=5
            )

            graph_results = results["graph_results"]
            expected = case["expected"]

            matching_ranks = [
                rank
                for rank, result in enumerate(graph_results, start=1)
                if matches(result, expected)
            ]

            found = bool(matching_ranks)

            if found:
                hits += 1
                reciprocal_ranks.append(1 / matching_ranks[0])
            else:
                reciprocal_ranks.append(0.0)

            print(f"\nQuery {index}: {case['query']}")
            print("Expected triple:", expected)
            print("Retrieved triples:")

            for result in graph_results:
                print(
                    result.get("subject"),
                    "--",
                    result.get("relationship"),
                    "-->",
                    result.get("object")
                )

            print("Expected triple found:", found)

    finally:
        retriever.close()

    print("\n========== FINAL GRAPH METRICS ==========")
    print("Recall@5:", round(hits / total_relevant, 4))
    print(
        "MRR:",
        round(sum(reciprocal_ranks) / total_relevant, 4)
    )


if __name__ == "__main__":
    main()

