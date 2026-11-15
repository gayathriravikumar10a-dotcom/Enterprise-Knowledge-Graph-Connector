
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extraction.entity_resolver import EntityResolver


def test_resolves_unambiguous_first_name():
    entities = [
        {"text": "Ravi Sharma", "label": "PERSON"},
        {"text": "Ravi", "label": "PERSON"},
    ]

    result = EntityResolver().resolve(entities, [])

    assert len(result["entities"]) == 1
    assert result["entities"][0]["text"] == "Ravi Sharma"


def test_does_not_merge_ambiguous_names():
    entities = [
        {"text": "Ravi Sharma", "label": "PERSON"},
        {"text": "Ravi Kumar", "label": "PERSON"},
        {"text": "Ravi", "label": "PERSON"},
    ]

    result = EntityResolver().resolve(entities, [])

    assert len(result["entities"]) == 3


def test_updates_relationship_endpoints():
    entities = [
        {"text": "Ravi Sharma", "label": "PERSON"},
        {"text": "Ravi", "label": "PERSON"},
        {"text": "Project Atlas", "label": "PROJECT"},
    ]

    relationships = [
        {
            "subject": "Ravi",
            "predicate": "MANAGES",
            "object": "Project Atlas",
        }
    ]

    result = EntityResolver().resolve(entities, relationships)

    assert result["relationships"][0]["subject"] == "Ravi Sharma"


def test_does_not_merge_different_entity_types():
    entities = [
        {"text": "Atlas", "label": "PERSON"},
        {"text": "Atlas", "label": "PROJECT"},
    ]

    result = EntityResolver().resolve(entities, [])

    assert len(result["entities"]) == 2


def test_normalizes_case_and_whitespace():
    entities = [
        {"text": "Ravi Sharma", "label": "PERSON"},
        {"text": "  ravi   sharma ", "label": "PERSON"},
    ]

    result = EntityResolver().resolve(entities, [])

    assert len(result["entities"]) == 1
