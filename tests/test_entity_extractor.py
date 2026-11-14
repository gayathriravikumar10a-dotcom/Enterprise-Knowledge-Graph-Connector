from extraction.entity_extractor import extract_entities


def test_extract_person_entities():
    text = "Asha works with Ravi on the project."

    entities = extract_entities(text)

    people = [
        entity["text"]
        for entity in entities
        if entity["label"] == "PERSON"
    ]

    assert "Asha" in people
    assert "Ravi" in people


def test_extract_custom_enterprise_entities():
    text = (
        "Asha works in Artificial Intelligence "
        "on Project Atlas."
    )

    entities = extract_entities(text)

    entity_pairs = [
        (entity["text"], entity["label"])
        for entity in entities
    ]

    assert ("Project Atlas", "PROJECT") in entity_pairs
    assert ("Artificial Intelligence", "DEPARTMENT") in entity_pairs


def test_empty_text():
    assert extract_entities("") == []
