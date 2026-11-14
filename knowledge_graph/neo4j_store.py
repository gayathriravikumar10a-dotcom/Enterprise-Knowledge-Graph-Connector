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

    def add_entity(self, name, entity_type):
        query = """
        MERGE (e:Entity {name: $name})
        SET e.entity_type = $entity_type
        RETURN e.name AS name
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                name=name,
                entity_type=entity_type
            )

            record = result.single()

            if record is None:
                raise RuntimeError(
                    f"Failed to store entity: {name}"
                )

            return record["name"]

    def add_relationship(
        self,
        subject,
        predicate,
        object_name,
        evidence=""
    ):
        query = """
        MATCH (s:Entity {name: $subject})
        MATCH (o:Entity {name: $object_name})
        MERGE (s)-[r:RELATED_TO {predicate: $predicate}]->(o)
        SET r.evidence = $evidence
        RETURN s.name AS subject, o.name AS object
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                subject=subject,
                predicate=predicate,
                object_name=object_name,
                evidence=evidence
            )

            record = result.single()

            if record is None:
                raise ValueError(
                    f"Could not create relationship: "
                    f"{subject} -{predicate}-> {object_name}. "
                    "Check that both entities exist."
                )

            return dict(record)

    def store_graph(self, knowledge_graph):
        graph = knowledge_graph.graph

        entity_count = 0
        relationship_count = 0

        # Store all entities first
        for name, attributes in graph.nodes(data=True):
            entity_type = attributes.get(
                "entity_type",
                attributes.get("label", "UNKNOWN")
            )

            self.add_entity(name, entity_type)
            entity_count += 1

        # Store relationships after entities
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
                evidence
            )

            relationship_count += 1

        print(f"Neo4j entities stored: {entity_count}")
        print(f"Neo4j relationships stored: {relationship_count}")

        return {
            "entities_stored": entity_count,
            "relationships_stored": relationship_count
        }

    def close(self):
        self.driver.close()
