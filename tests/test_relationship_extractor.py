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
