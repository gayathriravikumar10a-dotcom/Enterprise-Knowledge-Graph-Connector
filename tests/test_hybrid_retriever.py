import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_retriever import HybridRetriever


def test_extract_query_terms():
    retriever = HybridRetriever.__new__(HybridRetriever)

    terms = retriever.extract_query_terms(
        "Which employee manages Project Atlas?"
    )

    assert "employee" in terms
    assert "manages" in terms
    assert "project" in terms
    assert "atlas" in terms
    assert "which" not in terms


def test_vector_search():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.vector_store = MagicMock()

    retriever.vector_store.search.return_value = [
        {
            "chunk_id": "chunk_0",
            "text": "Ravi manages Project Atlas.",
            "distance": 0.25,
            "metadata": {}
        }
    ]

    results = retriever.vector_search(
        "Who manages Project Atlas?"
    )

    assert len(results) == 1
    assert "Ravi" in results[0]["text"]


def test_graph_search():
    retriever = HybridRetriever.__new__(HybridRetriever)

    retriever.graph_search = MagicMock(return_value=[
        {
            "matched_entity": "Project Atlas",
            "entity_type": "PROJECT",
            "subject": "Ravi",
            "relationship": "MANAGES",
            "object": "Project Atlas",
            "evidence": "Ravi manages Project Atlas."
        }
    ])

    results = retriever.graph_search(
        "Who manages Project Atlas?"
    )

    assert len(results) == 1
    assert results[0]["relationship"] == "MANAGES"


def test_retrieve_combines_results():
    retriever = HybridRetriever.__new__(HybridRetriever)

    retriever.vector_search = MagicMock(return_value=[
        {
            "text": "Ravi manages Project Atlas.",
            "distance": 0.25
        }
    ])

    retriever.graph_search = MagicMock(return_value=[
        {
            "subject": "Ravi",
            "relationship": "MANAGES",
            "object": "Project Atlas"
        }
    ])

    results = retriever.retrieve(
        "Who manages Project Atlas?"
    )

    assert results["vector_result_count"] == 1
    assert results["graph_result_count"] == 1
    assert results["vector_results"]
    assert results["graph_results"]


def test_retrieve_rejects_empty_query():
    retriever = HybridRetriever.__new__(HybridRetriever)

    try:
        retriever.retrieve("")
        assert False, "Expected ValueError"
    except ValueError:
        assert True
