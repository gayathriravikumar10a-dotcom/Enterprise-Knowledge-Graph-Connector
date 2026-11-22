from knowledge_graph.graph_builder import KnowledgeGraph
from knowledge_graph.reasoning_engine import GraphReasoner


def create_test_graph():
    kg = KnowledgeGraph()

    kg.add_relationship(
        "Asha", "WORKS_ON", "Project Atlas",
        "Asha works on Project Atlas."
    )

    kg.add_relationship(
        "Ravi", "MANAGES", "Project Atlas",
        "Ravi manages Project Atlas."
    )

    return kg


def test_find_projects():
    kg = create_test_graph()
    reasoner = GraphReasoner(kg)

    assert reasoner.find_objects(
        "Asha", "WORKS_ON"
    ) == ["Project Atlas"]


def test_find_project_managers():
    kg = create_test_graph()
    reasoner = GraphReasoner(kg)

    assert reasoner.find_subjects(
        "MANAGES", "Project Atlas"
    ) == ["Ravi"]


def test_multi_hop_reasoning():
    kg = create_test_graph()
    reasoner = GraphReasoner(kg)

    answers = reasoner.who_manages_persons_project("Asha")

    assert len(answers) == 1
    assert answers[0]["manager"] == "Ravi"
    assert answers[0]["project"] == "Project Atlas"


def test_unknown_person():
    kg = create_test_graph()
    reasoner = GraphReasoner(kg)

    assert reasoner.who_manages_persons_project(
        "Unknown"
    ) == []


def test_explain_relationship():
    kg = create_test_graph()
    reasoner = GraphReasoner(kg)

    evidence = reasoner.explain_relationship(
        "Asha", "WORKS_ON", "Project Atlas"
    )

    assert evidence == "Asha works on Project Atlas."
def test_find_paths_and_indirect_departments():
    kg = KnowledgeGraph()
    kg.add_entity("Gagan", "PERSON")
    kg.add_entity("AI Development Team", "TEAM")
    kg.add_entity("Technology Department", "DEPARTMENT")
    kg.add_entity("Project Alpha", "PROJECT")

    kg.add_relationship("Gagan", "BELONGS_TO", "AI Development Team", "Gagan works in AI Development Team.")
    kg.add_relationship("AI Development Team", "BELONGS_TO", "Technology Department", "Team belongs to Technology Department.")
    kg.add_relationship("AI Development Team", "WORKS_ON", "Project Alpha", "Team works on Project Alpha.")

    reasoner = GraphReasoner(kg)

    paths = reasoner.find_paths("Gagan", "Technology Department", max_hops=3)
    assert len(paths) == 1
    assert paths[0]["hops"] == 2
    assert "AI Development Team" in paths[0]["reasoning"]

    depts = reasoner.infer_indirect_departments("Gagan")
    assert len(depts) == 1
    assert depts[0]["department"] == "Technology Department"
    assert depts[0]["via"] == "AI Development Team"