from extraction.entity_extractor import extract_entities, register_entity_patterns
from extraction.entity_resolver import EntityResolver
from extraction.relationship_extractor import extract_relationships


DEFAULT_BENCHMARK = [
    {
        "text": "Ravi Sharma manages Project Atlas.",
        "entities": {("Ravi Sharma", "PERSON"), ("Project Atlas", "PROJECT")},
        "relationships": {("Ravi Sharma", "MANAGES", "Project Atlas")},
    },
    {
        "text": "Asha works in Engineering and works on Project Atlas.",
        "entities": {("Asha", "PERSON"), ("Engineering", "DEPARTMENT"), ("Project Atlas", "PROJECT")},
        "relationships": {
            ("Asha", "BELONGS_TO", "Engineering"),
            ("Asha", "WORKS_ON", "Project Atlas"),
        },
    },
    {
        "text": "Gagan works in the AI Development Team. The AI Development Team belongs to the Technology Department.",
        "entities": {
            ("Gagan", "PERSON"),
            ("AI Development Team", "TEAM"),
            ("Technology Department", "DEPARTMENT"),
        },
        "relationships": {
            ("Gagan", "BELONGS_TO", "AI Development Team"),
            ("AI Development Team", "BELONGS_TO", "Technology Department"),
        },
    },
    {
        "text": "Project Alpha is managed by R. Sharma and Ravi Sharma owns Knowledge Connector.",
        "entities": {
            ("Project Alpha", "PROJECT"),
            ("Ravi Sharma", "PERSON"),
            ("Knowledge Connector", "PRODUCT"),
        },
        "relationships": {
            ("Ravi Sharma", "MANAGES", "Project Alpha"),
            ("Ravi Sharma", "OWNS", "Knowledge Connector"),
        },
    },
]


def _compute_prf(predicted: set, gold: set) -> dict:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_extraction_quality(dataset: list[dict] | None = None) -> dict:
    """Evaluate entity recognition, entity resolution, and relationship extraction."""
    register_entity_patterns([
        {"label": "TEAM", "pattern": "AI Development Team"},
        {"label": "DEPARTMENT", "pattern": "Technology Department"},
        {"label": "PROJECT", "pattern": "Project Alpha"},
        {"label": "PRODUCT", "pattern": "Knowledge Connector"},
    ])

    samples = dataset or DEFAULT_BENCHMARK
    resolver = EntityResolver()

    pred_entities_all = set()
    gold_entities_all = set()
    pred_rels_all = set()
    gold_rels_all = set()

    for idx, sample in enumerate(samples):
        raw_entities = extract_entities(sample["text"])
        raw_rels = extract_relationships(sample["text"])
        resolved = resolver.resolve(raw_entities, raw_rels)

        for ent in resolved["entities"]:
            pred_entities_all.add((idx, ent["text"], ent["label"]))
        for text_val, label_val in sample["entities"]:
            gold_entities_all.add((idx, text_val, label_val))

        for rel in resolved["relationships"]:
            pred_rels_all.add((idx, rel["subject"], rel["predicate"], rel["object"]))
        for subj, pred, obj in sample["relationships"]:
            gold_rels_all.add((idx, subj, pred, obj))

    return {
        "sample_count": len(samples),
        "entity_metrics": _compute_prf(pred_entities_all, gold_entities_all),
        "relationship_metrics": _compute_prf(pred_rels_all, gold_rels_all),
    }


if __name__ == "__main__":
    report = evaluate_extraction_quality()
    print("Entity Extraction Quality:", report["entity_metrics"])
    print("Relationship Extraction Quality:", report["relationship_metrics"])