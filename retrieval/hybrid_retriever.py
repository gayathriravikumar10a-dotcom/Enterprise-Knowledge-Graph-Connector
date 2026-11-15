
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
        "to", "in", "on", "for", "and", "does", "do",
        "did", "can", "could", "would", "should",
        "please", "tell", "me", "about", "find", "show"
    }

    RELATIONSHIP_INTENTS = {
        "MANAGES": {
            "manage", "manages", "managed", "manager",
            "managing", "responsible", "leads", "lead"
        },
        "WORKS_ON": {
            "work", "works", "working", "worked",
            "contributes", "contribute", "develops"
        },
        "BELONGS_TO": {
            "belong", "belongs", "member", "part",
            "department", "team"
        }
    }

    def __init__(self):
        self.vector_store = ChromaVectorStore(
            collection_name="enterprise_documents"
        )
        self.graph_store = Neo4jStore()

    def extract_query_terms(self, query):

        if not query or not query.strip():
            return []

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

    def extract_relationship_intent(self, query):
        """Identify the relationship requested by the user."""

        query_words = set(
            re.findall(r"[a-zA-Z0-9]+", query.lower())
        )

        for relationship, keywords in self.RELATIONSHIP_INTENTS.items():
            if query_words.intersection(keywords):
                return relationship

        return None

    def calculate_graph_score(self, result, query_terms, intent):
        """Calculate graph relevance using entity and intent matching."""

        matched_terms = set(
            result.get("matched_terms") or []
        )

        query_term_set = set(query_terms)

        if not query_term_set:
            return 0.0

        entity_score = (
            len(matched_terms.intersection(query_term_set))
            / len(query_term_set)
        )

        relationship_text = (
            f"{result.get('relationship') or ''} "
            f"{result.get('evidence') or ''}"
        )

        relationship_terms = set(
            re.findall(
                r"[a-zA-Z0-9]+",
                relationship_text.lower()
            )
        )

        relationship_overlap = (
            len(query_term_set.intersection(relationship_terms))
            / len(query_term_set)
        )

        score = (
            0.75 * entity_score
            + 0.25 * relationship_overlap
        )

        # Strongly prioritize the relationship explicitly requested.
        if intent and result.get("relationship") == intent:
            score += 0.35

        return round(min(score, 1.0), 4)

    def deduplicate_graph_results(self, results):

        unique_results = {}

        for result in results:

            key = (
                result.get("subject"),
                result.get("relationship"),
                result.get("object")
            )

            existing = unique_results.get(key)

            if (
                existing is None
                or result.get("relevance_score", 0)
                > existing.get("relevance_score", 0)
            ):
                unique_results[key] = result

        return list(unique_results.values())

    def graph_search(self, query, limit=10):

        if limit < 1:
            raise ValueError("limit must be at least 1")

        terms = self.extract_query_terms(query)

        if not terms:
            return []

        intent = self.extract_relationship_intent(query)

        cypher_query = """
        MATCH (e:Entity)-[r:RELATED_TO]-(other:Entity)

        WITH
            e,
            r,
            other,
            [
                term IN $terms
                WHERE term IN split(toLower(e.name), ' ')
            ] AS matched_terms

        WHERE size(matched_terms) > 0

        RETURN
            e.name AS matched_entity,
            e.entity_type AS entity_type,
            startNode(r).name AS subject,
            r.predicate AS relationship,
            endNode(r).name AS object,
            r.evidence AS evidence,
            matched_terms AS matched_terms

        LIMIT $candidate_limit
        """

        with self.graph_store.driver.session() as session:

            result = session.run(
                cypher_query,
                terms=terms,
                candidate_limit=max(limit * 5, limit)
            )

            graph_results = [
                dict(record)
                for record in result
            ]

        for item in graph_results:

            item["relevance_score"] = self.calculate_graph_score(
                item,
                terms,
                intent
            )

        graph_results = self.deduplicate_graph_results(
            graph_results
        )

        graph_results.sort(
            key=lambda item: item["relevance_score"],
            reverse=True
        )

        return graph_results[:limit]

    def retrieve(
        self,
        query,
        vector_top_k=5,
        graph_limit=10
    ):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if vector_top_k < 1:
            raise ValueError("vector_top_k must be at least 1")

        if graph_limit < 1:
            raise ValueError("graph_limit must be at least 1")

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

        queries = [
            "Which employee manages Project Atlas?",
            "Who works on Project Atlas?"
        ]

        for query in queries:

            results = retriever.retrieve(query)

            print("\n" + "=" * 60)
            print("QUERY:", query)
            print("=" * 60)

            print("\nGRAPH RESULTS")

            for index, item in enumerate(
                results["graph_results"],
                start=1
            ):

                print(f"\nResult {index}")
                print("Subject:", item["subject"])
                print("Relationship:", item["relationship"])
                print("Object:", item["object"])
                print("Evidence:", item["evidence"])
                print("Relevance:", item["relevance_score"])

    finally:
        retriever.close()
