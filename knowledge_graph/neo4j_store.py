import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class Neo4jStore:

    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        if not all([uri, username, password]):
            raise ValueError(
                "Neo4j configuration missing. Check your .env file."
            )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

    def verify_connection(self):
        self.driver.verify_connectivity()
        return True

    def create_constraints(self):
        query = """
        CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
        FOR (e:Entity)
        REQUIRE e.name IS UNIQUE
        """

        with self.driver.session() as session:
            session.run(query).consume()

    def add_entity(self, name, entity_type, document_id=None, aliases=None):
        clean_aliases = sorted({
            a.strip() for a in (aliases or [])
            if a and a.strip() and a.strip() != name
        })

        query = """
        MERGE (e:Entity {name: $name})

        ON CREATE SET
            e.source_document_ids = [],
            e.aliases = [],
            e.provenance_complete = true

        ON MATCH SET
            e.provenance_complete =
                CASE
                    WHEN e.source_document_ids IS NULL THEN false
                    ELSE coalesce(e.provenance_complete, true)
                END

        SET e.entity_type = $entity_type

        SET e.aliases = reduce(
            acc = coalesce(e.aliases, []),
            alias IN $aliases |
            CASE WHEN alias IN acc THEN acc ELSE acc + alias END
        )

        SET e.source_document_ids =
            CASE
                WHEN $document_id IS NULL
                    THEN coalesce(e.source_document_ids, [])
                WHEN $document_id IN coalesce(e.source_document_ids, [])
                    THEN coalesce(e.source_document_ids, [])
                ELSE coalesce(e.source_document_ids, []) + $document_id
            END

        RETURN e.name AS name
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                name=name,
                entity_type=entity_type,
                document_id=document_id,
                aliases=clean_aliases,
            )

            record = result.single()

            if record is None:
                raise RuntimeError(
                    f"Failed to store entity: {name}"
                )

            return record["name"]

    def find_entity_by_name_or_alias(self, query_name: str) -> dict | None:
        """Find a canonical Entity node matching either e.name or an entry in e.aliases."""
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) = toLower($q)
           OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) = toLower($q))
        RETURN e.name AS name,
               e.entity_type AS entity_type,
               coalesce(e.aliases, []) AS aliases
        LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(query, q=query_name.strip()).single()
            return dict(record) if record is not None else None

    def get_graph_snapshot(self, limit: int = 200) -> dict:
        """Return nodes and relationships stored in Neo4j for visualization and inspection."""
        query = """
        MATCH (e:Entity)
        WITH e LIMIT $limit
        WITH collect({
            id: e.name,
            label: e.name,
            entity_type: coalesce(e.entity_type, 'UNKNOWN'),
            aliases: coalesce(e.aliases, [])
        }) AS nodes
        OPTIONAL MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
        WITH nodes, collect({
            source: s.name,
            target: o.name,
            predicate: coalesce(r.predicate, 'RELATED_TO'),
            evidence: coalesce(r.evidence, '')
        }) AS raw_edges
        RETURN nodes, [edge IN raw_edges WHERE edge.source IS NOT NULL] AS edges
        """
        with self.driver.session() as session:
            record = session.run(query, limit=limit).single()
            if record is None:
                return {"nodes": [], "edges": []}
            return {
                "nodes": record["nodes"],
                "edges": record["edges"],
            }

    def add_relationship(
        self,
        subject,
        predicate,
        object_name,
        evidence="",
        document_id=None
    ):
        query = """
        MATCH (s:Entity {name: $subject})
        MATCH (o:Entity {name: $object_name})

        MERGE (s)-[r:RELATED_TO {predicate: $predicate}]->(o)

        ON CREATE SET
            r.source_document_ids = [],
            r.provenance_complete = true

        ON MATCH SET
            r.provenance_complete =
                CASE
                    WHEN r.source_document_ids IS NULL THEN false
                    ELSE coalesce(r.provenance_complete, true)
                END

        SET r.evidence = $evidence

        SET r.source_document_ids =
            CASE
                WHEN $document_id IS NULL
                    THEN coalesce(r.source_document_ids, [])
                WHEN $document_id IN coalesce(r.source_document_ids, [])
                    THEN coalesce(r.source_document_ids, [])
                ELSE coalesce(r.source_document_ids, []) + $document_id
            END

        RETURN s.name AS subject, o.name AS object
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                subject=subject,
                predicate=predicate,
                object_name=object_name,
                evidence=evidence,
                document_id=document_id
            )

            record = result.single()

            if record is None:
                raise ValueError(
                    f"Could not create relationship: "
                    f"{subject} -{predicate}-> {object_name}. "
                    "Check that both entities exist."
                )

            return dict(record)

    def store_graph(self, knowledge_graph, document_id=None):
        graph = knowledge_graph.graph

        entity_count = 0
        relationship_count = 0

        for name, attributes in graph.nodes(data=True):
            entity_type = attributes.get(
                "entity_type",
                attributes.get("label", "UNKNOWN")
            )
            aliases = attributes.get("aliases", [])

            self.add_entity(
                name,
                entity_type,
                document_id=document_id,
                aliases=aliases,
            )

            entity_count += 1

        for subject, object_name, attributes in graph.edges(data=True):
            predicate = attributes.get(
                "relation",
                attributes.get("predicate", "RELATED_TO")
            )

            evidence = attributes.get("evidence", "")

            self.add_relationship(
                subject,
                predicate,
                object_name,
                evidence,
                document_id=document_id
            )

            relationship_count += 1

        print(f"Neo4j entities stored: {entity_count}")
        print(f"Neo4j relationships stored: {relationship_count}")

        return {
            "entities_stored": entity_count,
            "relationships_stored": relationship_count
        }

    def delete_document_data(self, document_id):
        """
        Remove a document's provenance from the graph.

        Delete only graph records whose provenance is complete
        and which are no longer supported by another document.
        Legacy records with unknown provenance are preserved.
        """

        def delete_data_transaction(tx):

            relationship_result = tx.run(
                """
                MATCH ()-[r:RELATED_TO]->()
                WHERE $document_id IN coalesce(
                    r.source_document_ids, []
                )

                WITH r,
                     [
                         id IN coalesce(r.source_document_ids, [])
                         WHERE id <> $document_id
                     ] AS remaining

                SET r.source_document_ids = remaining

                WITH r, remaining
                WHERE size(remaining) = 0
                  AND coalesce(r.provenance_complete, false) = true

                DELETE r
                RETURN count(r) AS deleted_relationships
                """,
                document_id=document_id
            ).single()

            entity_result = tx.run(
                """
                MATCH (e:Entity)
                WHERE $document_id IN coalesce(
                    e.source_document_ids, []
                )

                WITH e,
                     [
                         id IN coalesce(e.source_document_ids, [])
                         WHERE id <> $document_id
                     ] AS remaining

                SET e.source_document_ids = remaining

                WITH e, remaining
                OPTIONAL MATCH (e)-[r]-()

                WITH e, remaining, count(r) AS degree

                WHERE size(remaining) = 0
                  AND degree = 0
                  AND coalesce(e.provenance_complete, false) = true

                DELETE e
                RETURN count(e) AS deleted_entities
                """,
                document_id=document_id
            ).single()

            return {
                "relationships_deleted":
                    relationship_result["deleted_relationships"],
                "entities_deleted":
                    entity_result["deleted_entities"]
            }

        with self.driver.session() as session:
            return session.execute_write(delete_data_transaction)

    def close(self):
        self.driver.close()