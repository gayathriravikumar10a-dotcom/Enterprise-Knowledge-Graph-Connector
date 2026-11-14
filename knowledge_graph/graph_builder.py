import networkx as nx


class KnowledgeGraph:

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entity(self, name: str, entity_type: str):
        self.graph.add_node(
            name,
            entity_type=entity_type
        )

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        evidence: str = ""
    ):
        self.graph.add_edge(
            subject,
            object_name,
            relation=predicate,
            evidence=evidence
        )

    def build_from_extractions(self, entities, relationships):
        for entity in entities:
            self.add_entity(
                entity["text"],
                entity["label"]
            )

        for relationship in relationships:
            self.add_relationship(
                relationship["subject"],
                relationship["predicate"],
                relationship["object"],
                relationship.get("evidence", "")
            )

    def get_neighbors(self, entity_name: str):
        if entity_name not in self.graph:
            return []

        return list(self.graph.neighbors(entity_name))

    def get_relationships(self):
        relationships = []

        for source, target, data in self.graph.edges(data=True):
            relationships.append({
                "subject": source,
                "predicate": data["relation"],
                "object": target,
                "evidence": data.get("evidence", "")
            })

        return relationships
