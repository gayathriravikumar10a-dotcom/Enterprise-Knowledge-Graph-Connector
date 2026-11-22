from extraction.relationship_extractor import extract_relationships


def test_extract_works_on_relationship():
    text = "Asha works on Project Atlas."

    relationships = extract_relationships(text)

    assert len(relationships) == 1
    assert relationships[0]["subject"] == "Asha"
    assert relationships[0]["predicate"] == "WORKS_ON"
    assert relationships[0]["object"] == "Project Atlas"


def test_extract_manages_relationship():
    text = "Ravi manages Project Atlas."

    relationships = extract_relationships(text)

    assert len(relationships) == 1
    assert relationships[0]["subject"] == "Ravi"
    assert relationships[0]["predicate"] == "MANAGES"
    assert relationships[0]["object"] == "Project Atlas"


def test_extract_department_relationship():
    text = "Asha works in Engineering."

    relationships = extract_relationships(text)

    assert len(relationships) == 1
    assert relationships[0]["subject"] == "Asha"
    assert relationships[0]["predicate"] == "BELONGS_TO"
    assert relationships[0]["object"] == "Engineering"


def test_empty_relationship_input():
    assert extract_relationships("") == []


def test_unrelated_sentence():
    text = "The weather is sunny today."

    assert extract_relationships(text) == []
def test_extract_passive_and_team_relationships():
    from extraction.entity_extractor import register_entity_patterns

    register_entity_patterns([
        {"label": "TEAM", "pattern": "AI Development Team"},
        {"label": "DEPARTMENT", "pattern": "Technology Department"},
        {"label": "PROJECT", "pattern": "Project Alpha"},
    ])

    text = (
        "Gagan works in the AI Development Team. "
        "The AI Development Team belongs to the Technology Department "
        "and works on Project Alpha. "
        "Project Atlas is managed by Ravi."
    )

    relationships = extract_relationships(text)
    triples = {(r["subject"], r["predicate"], r["object"]) for r in relationships}

    assert ("Gagan", "BELONGS_TO", "AI Development Team") in triples
    assert ("AI Development Team", "BELONGS_TO", "Technology Department") in triples
    assert ("AI Development Team", "WORKS_ON", "Project Alpha") in triples
    assert ("Ravi", "MANAGES", "Project Atlas") in triples