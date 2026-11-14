import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.workflow import build_knowledge_graph_from_document
from knowledge_graph.neo4j_store import Neo4jStore

DOCUMENT_PATH = PROJECT_ROOT / "data" / "sample_documents" / "graph_demo.txt"


def main():
    print("BUILDING KNOWLEDGE GRAPH")
    print("-" * 40)

    result = build_knowledge_graph_from_document(str(DOCUMENT_PATH))
    knowledge_graph = result["knowledge_graph"]

    entity_count = len(knowledge_graph.graph.nodes)
    relationship_count = len(knowledge_graph.graph.edges)

    print(f"Entities extracted: {entity_count}")
    print(f"Relationships extracted: {relationship_count}")

    print("\nCONNECTING TO NEO4J")

    store = Neo4jStore()

    try:
        store.verify_connection()
        store.create_constraints()
        store.store_graph(knowledge_graph)

        print("\nKNOWLEDGE GRAPH STORED SUCCESSFULLY")
        print(f"Entities submitted: {entity_count}")
        print(f"Relationships submitted: {relationship_count}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
