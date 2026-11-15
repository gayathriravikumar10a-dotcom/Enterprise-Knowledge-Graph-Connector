
from ingestion.document_loader import load_document
from ingestion.chunker import chunk_text
from extraction.entity_extractor import extract_entities
from extraction.relationship_extractor import extract_relationships
from extraction.entity_resolver import EntityResolver
from knowledge_graph.graph_builder import KnowledgeGraph
from knowledge_graph.reasoning_engine import GraphReasoner


def build_knowledge_graph_from_document(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> dict:
    """Convert an enterprise document into a resolved knowledge graph."""

    document = load_document(file_path)

    chunks = chunk_text(
        document["text"],
        chunk_size=chunk_size,
        overlap=overlap
    )

    extracted_entities = extract_entities(document["text"])
    extracted_relationships = extract_relationships(document["text"])

    # Resolve duplicate entities and relationship aliases.
    resolution_result = EntityResolver().resolve(
        extracted_entities,
        extracted_relationships
    )

    entities = resolution_result["entities"]
    relationships = resolution_result["relationships"]

    # Build graph using canonical entity names.
    kg = KnowledgeGraph()
    kg.build_from_extractions(entities, relationships)

    return {
        "source": document["metadata"]["source"],
        "chunks": chunks,
        "entities": entities,
        "relationships": relationships,
        "knowledge_graph": kg,
        "reasoner": GraphReasoner(kg),
        "entity_resolution": {
            "entities_before": resolution_result["entities_before"],
            "entities_after": resolution_result["entities_after"],
            "aliases_resolved": resolution_result["alias_map"]
        }
    }
