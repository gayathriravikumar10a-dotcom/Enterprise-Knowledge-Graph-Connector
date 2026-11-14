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
