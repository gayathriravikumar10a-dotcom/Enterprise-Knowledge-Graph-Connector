import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vector_store.chroma_store import ChromaVectorStore
from knowledge_graph.neo4j_store import Neo4jStore


class HybridRetriever:

    STOP_WORDS = {
        "who", "what", "which", "where", "when", "why",
        "how", "is", "are", "the", "a", "an", "of",
        "to", "in", "on", "for", "and", "does", "do"
    }

    def __init__(self):
        self.vector_store = ChromaVectorStore(
            collection_name="enterprise_documents"
        )
        self.graph_store = Neo4jStore()

    def extract_query_terms(self, query):
        words = re.findall(r"[a-zA-Z0-9]+", query.lower())

        return [
            word for word in words
            if word not in self.STOP_WORDS and len(word) > 1
        ]

    def vector_search(self, query, top_k=5):
        return self.vector_store.search(
            query=query,
            top_k=top_k
        )

    def graph_search(self, query, limit=10):
        terms = self.extract_query_terms(query)

        if not terms:
            return []

        cypher_query = """
        MATCH (e:Entity)
        WHERE any(
            term IN $terms
            WHERE toLower(e.name) CONTAINS term
               OR term CONTAINS toLower(e.name)
        )
        OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
        RETURN
            e.name AS matched_entity,
            e.entity_type AS entity_type,
            startNode(r).name AS subject,
            r.predicate AS relationship,
            endNode(r).name AS object,
            r.evidence AS evidence
        LIMIT $limit
        """

        with self.graph_store.driver.session() as session:
            result = session.run(
                cypher_query,
                terms=terms,
                limit=limit
            )

            return [dict(record) for record in result]

    def retrieve(self, query, vector_top_k=5, graph_limit=10):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        vector_results = self.vector_search(
            query,
            top_k=vector_top_k
        )

        graph_results = self.graph_search(
            query,
            limit=graph_limit
        )

        return {
            "query": query,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "vector_result_count": len(vector_results),
            "graph_result_count": len(graph_results)
        }

    def close(self):
        self.graph_store.close()


if __name__ == "__main__":
    retriever = HybridRetriever()

    try:
        query = "Which employee manages Project Atlas?"

        results = retriever.retrieve(query)

        print("\nHYBRID RETRIEVAL RESULTS")
        print("=" * 55)
        print(f"Query: {results['query']}")

        print("\nVECTOR SEARCH RESULTS")
        print("-" * 55)

        for index, item in enumerate(
            results["vector_results"], start=1
        ):
            print(f"\nResult {index}")
            print(f"Text: {item['text']}")
            print(f"Distance: {item['distance']:.4f}")

        print("\nKNOWLEDGE GRAPH RESULTS")
        print("-" * 55)

        for index, item in enumerate(
            results["graph_results"], start=1
        ):
            print(f"\nResult {index}")
            print(f"Matched entity: {item['matched_entity']}")
            print(f"Entity type: {item['entity_type']}")
            print(f"Subject: {item['subject']}")
            print(f"Relationship: {item['relationship']}")
            print(f"Object: {item['object']}")
            print(f"Evidence: {item['evidence']}")

        print("\nHYBRID RETRIEVAL COMPLETED")

    finally:
        retriever.close()
