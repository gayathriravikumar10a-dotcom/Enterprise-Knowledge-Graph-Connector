from knowledge_graph.workflow import build_knowledge_graph_from_document


def test_document_to_knowledge_graph():
    result = build_knowledge_graph_from_document(
        "data/sample_documents/graph_demo.txt"
    )

    assert result["source"] == "graph_demo.txt"
    assert len(result["entities"]) >= 3
    assert len(result["relationships"]) >= 3


def test_integrated_multi_hop_reasoning():
    result = build_knowledge_graph_from_document(
        "data/sample_documents/graph_demo.txt"
    )

    answers = result["reasoner"].who_manages_persons_project("Asha")

    assert len(answers) == 1
    assert answers[0]["manager"] == "Ravi"
    assert answers[0]["project"] == "Project Atlas"


def test_integrated_graph_structure():
    result = build_knowledge_graph_from_document(
        "data/sample_documents/graph_demo.txt"
    )

    graph = result["knowledge_graph"].graph

    assert graph.has_edge("Asha", "Project Atlas")
    assert graph.has_edge("Ravi", "Project Atlas")
