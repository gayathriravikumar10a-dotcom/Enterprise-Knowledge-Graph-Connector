import re
from extraction.entity_extractor import ALLOWED_LABELS, nlp


RELATION_PATTERNS = {
    "WORKS_ON": [
        r"\bworks?\s+on\b",
        r"\bworking\s+on\b",
        r"\bassigned\s+to\b",
        r"\bcontributes?\s+to\b",
        r"\bbuilding\b",
        r"\bdevelops?\b",
        r"\bdeveloping\b",
    ],
    "MANAGES": [
        r"\bmanages?\b",
        r"\bmanaging\b",
        r"\bleads?\b",
        r"\bleading\b",
        r"\boversees?\b",
        r"\bheads?\b",
    ],
    "BELONGS_TO": [
        r"\bbelongs?\s+to\b",
        r"\bworks?\s+in\b",
        r"\bpart\s+of\b",
        r"\bmember\s+of\b",
        r"\bwithin\b",
        r"\bunder\b",
    ],
    "OWNS": [
        r"\bowns?\b",
        r"\bmaintains?\b",
        r"\bresponsible\s+for\b",
    ],
}

PASSIVE_PATTERNS = {
    "MANAGES": [
        r"\b(?:is|was|are|were)\s+(?:managed|led|overseen|headed)\s+by\b",
    ],
    "WORKS_ON": [
        r"\b(?:is|was|are|were)\s+(?:developed|built|worked\s+on)\s+by\b",
    ],
    "OWNS": [
        r"\b(?:is|was|are|were)\s+(?:owned|maintained)\s+by\b",
    ],
}

VALID_SCHEMA = {
    "WORKS_ON": {
        ("PERSON", "PROJECT"),
        ("PERSON", "PRODUCT"),
        ("TEAM", "PROJECT"),
        ("TEAM", "PRODUCT"),
    },
    "MANAGES": {
        ("PERSON", "PROJECT"),
        ("PERSON", "TEAM"),
        ("PERSON", "DEPARTMENT"),
        ("PERSON", "PRODUCT"),
        ("TEAM", "PROJECT"),
        ("DEPARTMENT", "PROJECT"),
    },
    "BELONGS_TO": {
        ("PERSON", "TEAM"),
        ("PERSON", "DEPARTMENT"),
        ("PERSON", "ORG"),
        ("TEAM", "DEPARTMENT"),
        ("TEAM", "ORG"),
        ("DEPARTMENT", "ORG"),
        ("PROJECT", "DEPARTMENT"),
        ("PRODUCT", "PROJECT"),
        ("PRODUCT", "DEPARTMENT"),
    },
    "OWNS": {
        ("PERSON", "PRODUCT"),
        ("PERSON", "PROJECT"),
        ("TEAM", "PRODUCT"),
        ("TEAM", "PROJECT"),
        ("DEPARTMENT", "PRODUCT"),
        ("DEPARTMENT", "PROJECT"),
        ("ORG", "PRODUCT"),
    },
}


def identify_relationship(sentence: str) -> str | None:
    """Identify the primary relationship in a sentence (kept for backward compatibility)."""
    for relation, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return relation
    for relation, patterns in PASSIVE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return relation
    return None


def _find_relation_matches(sentence_text: str) -> list[dict]:
    """Find all active and passive predicate matches with character spans."""
    matches = []

    for relation, patterns in PASSIVE_PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, sentence_text, re.IGNORECASE):
                matches.append({
                    "predicate": relation,
                    "start": m.start(),
                    "end": m.end(),
                    "passive": True,
                })

    for relation, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, sentence_text, re.IGNORECASE):
                if any(
                    existing["start"] <= m.start() and m.end() <= existing["end"]
                    for existing in matches
                ):
                    continue
                matches.append({
                    "predicate": relation,
                    "start": m.start(),
                    "end": m.end(),
                    "passive": False,
                })

    matches.sort(key=lambda item: item["start"])
    return matches


def extract_relationships(text: str) -> list:
    """Extract schema-validated subject-predicate-object relationships from text."""
    if not text or not text.strip():
        return []

    doc = nlp(text)
    relationships = []
    seen_triples = set()
    last_active_subject = None

    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        sent_start = sentence.start_char

        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char - sent_start,
                "end": ent.end_char - sent_start,
            }
            for ent in sentence.ents
            if ent.label_ in ALLOWED_LABELS
        ]

        if not entities:
            last_active_subject = None
            continue

        matches = _find_relation_matches(sentence_text)
        if not matches:
            last_active_subject = None
            continue

        consumed_objects = set()

        for idx, match in enumerate(matches):
            prev_end = matches[idx - 1]["end"] if idx > 0 else 0
            next_start = (
                matches[idx + 1]["start"]
                if idx + 1 < len(matches)
                else len(sentence_text)
            )

            before_ents = [
                e for e in entities
                if prev_end <= e["start"]
                and e["end"] <= match["start"]
                and (e["start"], e["end"]) not in consumed_objects
            ]
            after_ents = [
                e for e in entities
                if match["end"] <= e["start"]
                and e["end"] <= next_start
            ]

            if not after_ents:
                continue

            right_ent = after_ents[0]
            consumed_objects.add((right_ent["start"], right_ent["end"]))

            if before_ents:
                left_ent = before_ents[-1]
            elif not match["passive"] and last_active_subject is not None:
                left_ent = last_active_subject
            else:
                continue

            subject, obj = (
                (right_ent, left_ent) if match["passive"] else (left_ent, right_ent)
            )
            predicate = match["predicate"]

            if not match["passive"]:
                last_active_subject = subject

            if (subject["label"], obj["label"]) not in VALID_SCHEMA.get(predicate, set()):
                continue

            triple_key = (subject["text"], predicate, obj["text"])
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)

            relationships.append({
                "subject": subject["text"],
                "subject_type": subject["label"],
                "predicate": predicate,
                "object": obj["text"],
                "object_type": obj["label"],
                "evidence": sentence_text,
            })

    return relationships