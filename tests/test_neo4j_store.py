from unittest.mock import MagicMock
import pytest

from knowledge_graph.graph_builder import KnowledgeGraph
from knowledge_graph.neo4j_store import Neo4jStore


def _make_mock_store():
    store = Neo4jStore.__new__(Neo4jStore)
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    store.driver = mock_driver
    return store, mock_session


def test_neo4j_add_entity_stores_clean_aliases():
    store, session = _make_mock_store()
    session.run.return_value.single.return_value = {"name": "Ravi Sharma"}

    stored = store.add_entity(
        "Ravi Sharma",
        "PERSON",
        document_id="doc-1",
        aliases=["Ravi", "R. Sharma", "Ravi Sharma", ""],
    )

    assert stored == "Ravi Sharma"
    _, kwargs = session.run.call_args
    assert kwargs["name"] == "Ravi Sharma"
    assert kwargs["entity_type"] == "PERSON"
    assert kwargs["document_id"] == "doc-1"
    assert kwargs["aliases"] == ["R. Sharma", "Ravi"]


def test_neo4j_add_relationship_success_and_missing_node():
    store, session = _make_mock_store()
    session.run.return_value.single.return_value = {
        "subject": "Ravi Sharma",
        "object": "Project Atlas",
    }

    rel = store.add_relationship(
        "Ravi Sharma",
        "MANAGES",
        "Project Atlas",
        evidence="Ravi manages Project Atlas.",
        document_id="doc-1",
    )
    assert rel == {"subject": "Ravi Sharma", "object": "Project Atlas"}

    session.run.return_value.single.return_value = None
    with pytest.raises(ValueError, match="Check that both entities exist"):
        store.add_relationship("Unknown", "MANAGES", "Project Atlas")


def test_neo4j_store_graph_persists_entities_aliases_and_edges():
    store, session = _make_mock_store()
    session.run.return_value.single.side_effect = [
        {"name": "Ravi Sharma"},
        {"name": "Project Atlas"},
        {"subject": "Ravi Sharma", "object": "Project Atlas"},
    ]

    kg = KnowledgeGraph()
    kg.build_from_extractions(
        entities=[
            {"text": "Ravi Sharma", "label": "PERSON", "aliases": ["Ravi"]},
            {"text": "Project Atlas", "label": "PROJECT", "aliases": []},
        ],
        relationships=[
            {
                "subject": "Ravi Sharma",
                "predicate": "MANAGES",
                "object": "Project Atlas",
                "evidence": "Ravi manages Project Atlas.",
            }
        ],
    )

    summary = store.store_graph(kg, document_id="doc-42")
    assert summary == {"entities_stored": 2, "relationships_stored": 1}
    assert session.run.call_count == 3


def test_neo4j_find_entity_by_name_or_alias():
    store, session = _make_mock_store()
    session.run.return_value.single.return_value = {
        "name": "Artificial Intelligence",
        "entity_type": "DEPARTMENT",
        "aliases": ["AI"],
    }

    match = store.find_entity_by_name_or_alias("AI")
    assert match is not None
    assert match["name"] == "Artificial Intelligence"
    assert "AI" in match["aliases"]