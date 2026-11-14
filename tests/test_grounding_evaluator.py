import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.grounding_evaluator import GroundingEvaluator


def test_supported_answer():
    evaluator = GroundingEvaluator()

    retrieval_results = {
        "vector_results": [
            {"text": "Ravi manages Project Atlas."}
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

    result = evaluator.evaluate_answer(
        "Ravi manages Project Atlas.",
        retrieval_results
    )

    assert result["grounded"] is True
    assert result["support_ratio"] == 1.0


def test_unsupported_answer():
    evaluator = GroundingEvaluator()

    retrieval_results = {
        "vector_results": [
            {"text": "Ravi manages Project Atlas."}
        ],
        "graph_results": []
    }

    result = evaluator.evaluate_answer(
        "Asha manages Project Phoenix in Mumbai.",
        retrieval_results
    )

    assert result["grounded"] is False
    assert result["unsupported_terms"]


def test_empty_answer():
    evaluator = GroundingEvaluator()

    result = evaluator.evaluate_answer(
        "",
        {"vector_results": [], "graph_results": []}
    )

    assert result["grounded"] is False
    assert result["support_ratio"] == 0.0


def test_no_evidence():
    evaluator = GroundingEvaluator()

    result = evaluator.evaluate_answer(
        "Ravi manages Project Atlas.",
        {"vector_results": [], "graph_results": []}
    )

    assert result["grounded"] is False
    assert result["support_ratio"] == 0.0
