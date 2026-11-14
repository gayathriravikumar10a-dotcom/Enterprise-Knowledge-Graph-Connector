import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.hybrid_retriever import HybridRetriever
from rag.answer_generator import GroundedAnswerGenerator
from evaluation.grounding_evaluator import GroundingEvaluator


class EnterpriseRAGPipeline:

    def __init__(self):
        self.retriever = HybridRetriever()
        self.answer_generator = GroundedAnswerGenerator()
        self.grounding_evaluator = GroundingEvaluator()

    def ask(self, query):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        retrieval_results = self.retriever.retrieve(query)

        answer_result = self.answer_generator.generate_answer(
            query,
            retrieval_results
        )

        grounding_result = self.grounding_evaluator.evaluate_answer(
            answer_result["answer"],
            retrieval_results
        )

        if not grounding_result["grounded"]:
            final_answer = (
                "I could not verify a sufficiently supported answer "
                "from the available enterprise documents. "
                "Please refine your question or provide additional documents."
            )
            status = "insufficient_grounding"
        else:
            final_answer = answer_result["answer"]
            status = "grounded"

        return {
            "query": query,
            "answer": final_answer,
            "answer_status": status,
            "grounding_evaluation": grounding_result,
            "sources": answer_result["sources"],
            "model": answer_result.get("model"),
            "retrieval": retrieval_results
        }

    def close(self):
        self.retriever.close()


if __name__ == "__main__":

    pipeline = EnterpriseRAGPipeline()

    try:
        question = "Who manages Project Atlas?"
        result = pipeline.ask(question)

        print("\nQUESTION:")
        print(result["query"])

        print("\nANSWER:")
        print(result["answer"])

        print("\nANSWER STATUS:")
        print(result["answer_status"])

        print("\nGROUNDING EVALUATION:")
        print(result["grounding_evaluation"])

        print("\nSOURCE INFORMATION:")
        print(result["sources"])

        print("\nMODEL:")
        print(result["model"])

    finally:
        pipeline.close()
