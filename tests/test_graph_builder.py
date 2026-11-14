from knowledge_graph.graph_builder import KnowledgeGraph


def test_add_entities():
    kg = KnowledgeGraph()

    kg.add_entity("Asha", "PERSON")
    kg.add_entity("Project Atlas", "PROJECT")

    assert "Asha" in kg.graph
    assert "Project Atlas" in kg.graph


def test_add_relationship():
    kg = KnowledgeGraph()

    kg.add_relationship(
        "Asha",
        "WORKS_ON",
        "Project Atlas"
    )

    assert kg.graph.has_edge("Asha", "Project Atlas")


def test_build_graph_from_extractions():
    kg = KnowledgeGraph()

    entities = [
        {"text": "Asha", "label": "PERSON"},
        {"text": "Project Atlas", "label": "PROJECT"}
    ]

    relationships = [
        {
            "subject": "Asha",
            "predicate": "WORKS_ON",
            "object": "Project Atlas",
            "evidence": "Asha works on Project Atlas."
        }
    ]

    kg.build_from_extractions(entities, relationships)

    assert kg.graph.number_of_nodes() == 2
    assert kg.graph.number_of_edges() == 1


def test_get_neighbors():
    kg = KnowledgeGraph()

    kg.add_relationship("Asha", "WORKS_ON", "Project Atlas")

    assert "Project Atlas" in kg.get_neighbors("Asha")


def test_get_relationships():
    kg = KnowledgeGraph()

    kg.add_relationship("Ravi", "MANAGES", "Project Atlas")

    relationships = kg.get_relationships()

    assert relationships[0]["predicate"] == "MANAGES"
