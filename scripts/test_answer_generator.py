import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.answer_generator import GroundedAnswerGenerator


def main():
    generator = GroundedAnswerGenerator()

    retrieval_results = {
        "vector_results": [
            {
                "text": "Ravi manages Project Atlas.",
                "chunk_id": "chunk_0"
            }
        ],
        "graph_results": [
            {
                "subject": "Ravi",
                "relationship": "MANAGES",
                "object": "Project Atlas",
                "evidence": "Ravi manages Project Atlas."
            }
        ]
    }

    query = "Who manages Project Atlas?"

    result = generator.generate_answer(
        query,
        retrieval_results
    )

    print("\nGROUNDED ANSWER")
    print("=" * 50)
    print(result["answer"])

    print("\nSOURCE INFORMATION")
    print(result["sources"])

    print("\nGrounded:", result["grounded"])


if __name__ == "__main__":
    main()
