from knowledge_graph.visualizer import build_graph_data, render_knowledge_graph_html


def test_visualizer_normalizes_and_renders_html():
    nodes = [
        {"id": "Ravi Sharma", "entity_type": "PERSON", "aliases": ["Ravi"]},
        {"id": "Project Atlas", "entity_type": "PROJECT", "aliases": []},
    ]
    edges = [
        {"source": "Ravi Sharma", "target": "Project Atlas", "predicate": "MANAGES"}
    ]

    normalized = build_graph_data(nodes, edges)
    assert len(normalized["nodes"]) == 2
    assert len(normalized["edges"]) == 1

    html_output = render_knowledge_graph_html(nodes, edges)
    assert "Ravi Sharma" in html_output
    assert "Project Atlas" in html_output
    assert "MANAGES" in html_output