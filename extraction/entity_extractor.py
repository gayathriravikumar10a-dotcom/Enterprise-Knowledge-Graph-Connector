
import spacy


def create_nlp_pipeline():
    nlp = spacy.load("en_core_web_sm")

    ruler = nlp.add_pipe("entity_ruler", before="ner")

    patterns = [
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

    ruler.add_patterns(patterns)

    return nlp


nlp = create_nlp_pipeline()


def extract_entities(text: str) -> list:
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

