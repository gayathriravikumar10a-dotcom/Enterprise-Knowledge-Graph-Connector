
import spacy


# Entity categories supported by our knowledge graph.
ALLOWED_LABELS = {
    "PERSON",
    "PROJECT",
    "DEPARTMENT",
    "TEAM",
    "ORG",
    "PRODUCT",
}


# Default patterns from the original project.
DEFAULT_PATTERNS = [
    # Projects
    {"label": "PROJECT", "pattern": "Project Atlas"},

    # Departments
    {"label": "DEPARTMENT", "pattern": "Artificial Intelligence"},
    {"label": "DEPARTMENT", "pattern": "Engineering"},

    # Known people
    {"label": "PERSON", "pattern": "Ravi Sharma"},
    {"label": "PERSON", "pattern": "Asha"},
    {"label": "PERSON", "pattern": "Ravi"},
]


def validate_patterns(patterns):
    """Validate enterprise entity patterns before adding them."""

    if not isinstance(patterns, list):
        raise ValueError("Patterns must be provided as a list.")

    for item in patterns:
        if not isinstance(item, dict):
            raise ValueError("Each pattern must be a dictionary.")

        label = item.get("label")
        pattern = item.get("pattern")

        if label not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported entity label: {label}")

        if not isinstance(pattern, (str, list)) or not pattern:
            raise ValueError(
                "Each pattern must contain a non-empty string "
                "or a valid spaCy token pattern."
            )


def create_nlp_pipeline(extra_patterns=None):
    """Create the spaCy NLP pipeline with enterprise patterns."""

    nlp = spacy.load("en_core_web_sm")

    ruler = nlp.add_pipe("entity_ruler", before="ner")

    patterns = list(DEFAULT_PATTERNS)

    if extra_patterns:
        validate_patterns(extra_patterns)
        patterns.extend(extra_patterns)

    ruler.add_patterns(patterns)

    return nlp


# Shared NLP pipeline used by other project modules.
nlp = create_nlp_pipeline()


def register_entity_patterns(patterns):
    """
    Register additional enterprise entities at runtime.

    Example:
    {"label": "TEAM", "pattern": "AI Development Team"}
    """

    validate_patterns(patterns)

    ruler = nlp.get_pipe("entity_ruler")
    ruler.add_patterns(patterns)


def extract_entities(text: str) -> list:
    """Extract named entities from enterprise text."""

    if not text or not text.strip():
        return []

    doc = nlp(text)

    entities = []

    for entity in doc.ents:
        entities.append({
            "text": entity.text,
            "label": entity.label_,
            "start": entity.start_char,
            "end": entity.end_char
        })

    return entities
