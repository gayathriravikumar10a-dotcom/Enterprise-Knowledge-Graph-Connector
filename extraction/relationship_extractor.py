import re
from extraction.entity_extractor import nlp


RELATION_PATTERNS = {
    "WORKS_ON": [
        r"\bworks?\s+on\b",
        r"\bworking\s+on\b",
        r"\bassigned\s+to\b",
        r"\bcontributes?\s+to\b"
    ],
    "MANAGES": [
        r"\bmanages?\b",
        r"\bleads?\b",
        r"\b oversees?\b"
    ],
    "BELONGS_TO": [
        r"\bbelongs?\s+to\b",
        r"\bworks?\s+in\b",
        r"\bpart\s+of\b"
    ]
}


def identify_relationship(sentence: str) -> str | None:
    """Identify a relationship using predefined linguistic patterns."""

    for relation, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return relation

    return None


def extract_relationships(text: str) -> list:
    """Extract subject-predicate-object relationships from text."""

    if not text or not text.strip():
        return []

    doc = nlp(text)
    relationships = []

    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        relation = identify_relationship(sentence_text)

        if not relation:
            continue

        entities = [
            entity for entity in sentence.ents
            if entity.label_ in {"PERSON", "PROJECT", "DEPARTMENT", "ORG"}
        ]

        if len(entities) < 2:
            continue

        subject = entities[0]
        object_entity = entities[-1]

        if subject.label_ not in {"PERSON", "ORG"}:
            continue

        if relation == "WORKS_ON" and object_entity.label_ != "PROJECT":
            continue

        if relation == "MANAGES" and object_entity.label_ != "PROJECT":
            continue

        if relation == "BELONGS_TO" and object_entity.label_ not in {
            "DEPARTMENT", "ORG"
        }:
            continue

        relationships.append({
            "subject": subject.text,
            "subject_type": subject.label_,
            "predicate": relation,
            "object": object_entity.text,
            "object_type": object_entity.label_,
            "evidence": sentence_text
        })

    return relationships
