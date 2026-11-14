from rag.rag_pipeline import EnterpriseRAGPipeline


class MockRetriever:

    def retrieve(self, query):
        return {
            "query": query,
            "vector_results": [
                {
                    "text": "Ravi manages Project Atlas."
                }
            ],
            "graph_results": [
                {
                    "subject": "Ravi",
                    "relationship": "MANAGES",
                    "object": "Project Atlas",
                    "evidence": "Ravi manages Project Atlas."
                }
            ],
            "vector_result_count": 1,
            "graph_result_count": 1
        }

    def close(self):
        pass


class MockAnswerGenerator:

    def generate_answer(self, query, retrieval_results):
        return {
            "answer": "Ravi manages Project Atlas.",
            "sources": {
                "document_chunks": 1,
                "graph_relationships": 1
            },
            "model": "mock-model"
        }


def create_pipeline(monkeypatch):

    monkeypatch.setattr(
        "rag.rag_pipeline.HybridRetriever",
        MockRetriever
    )

    monkeypatch.setattr(
        "rag.rag_pipeline.GroundedAnswerGenerator",
        MockAnswerGenerator
    )

    return EnterpriseRAGPipeline()


def test_pipeline_returns_grounded_answer(monkeypatch):

    pipeline = create_pipeline(monkeypatch)

    result = pipeline.ask("Who manages Project Atlas?")

    assert result["answer"] == "Ravi manages Project Atlas."
    assert result["answer_status"] == "grounded"
    assert result["grounding_evaluation"]["grounded"] is True

    pipeline.close()


def test_pipeline_rejects_empty_query(monkeypatch):

    pipeline = create_pipeline(monkeypatch)

    import pytest

    with pytest.raises(ValueError):
        pipeline.ask("")

    pipeline.close()


def test_pipeline_abstains_on_unsupported_answer(monkeypatch):

    pipeline = create_pipeline(monkeypatch)

    class UnsupportedGenerator:

        def generate_answer(self, query, retrieval_results):
            return {
                "answer": "Asha manages Project Phoenix in Mumbai.",
                "sources": {
                    "document_chunks": 1,
                    "graph_relationships": 1
                },
                "model": "mock-model"
            }

    pipeline.answer_generator = UnsupportedGenerator()

    result = pipeline.ask("Who manages Project Atlas?")

    assert result["answer_status"] == "insufficient_grounding"
    assert result["grounding_evaluation"]["grounded"] is False

    pipeline.close()


def test_pipeline_returns_source_information(monkeypatch):

    pipeline = create_pipeline(monkeypatch)

    result = pipeline.ask("Who manages Project Atlas?")

    assert result["sources"]["document_chunks"] == 1
    assert result["sources"]["graph_relationships"] == 1

    pipeline.close()
