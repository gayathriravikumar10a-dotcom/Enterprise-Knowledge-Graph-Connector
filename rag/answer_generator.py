import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class GroundedAnswerGenerator:

    def __init__(self, model_name=None):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from the .env file."
            )

        self.client = genai.Client(api_key=api_key)

        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        )

    @staticmethod
    def safe_text(value, default="Unknown"):
        if value is None:
            return default

        text = str(value).strip()
        return text if text else default

    def build_context(self, retrieval_results):
        context_parts = ["DOCUMENT EVIDENCE:"]

        for index, item in enumerate(
            retrieval_results.get("vector_results", []),
            start=1
        ):
            if not isinstance(item, dict):
                continue

            document_text = self.safe_text(
                item.get("text"),
                "No document text available."
            )

            context_parts.append(
                f"Document Source {index}:\n{document_text}"
            )

        context_parts.append("KNOWLEDGE GRAPH EVIDENCE:")

        for index, item in enumerate(
            retrieval_results.get("graph_results", []),
            start=1
        ):
            if not isinstance(item, dict):
                continue

            subject = self.safe_text(item.get("subject"))
            relationship = self.safe_text(
                item.get("relationship")
            )
            object_name = self.safe_text(item.get("object"))
            evidence = self.safe_text(
                item.get("evidence"),
                "No supporting text available."
            )

            context_parts.append(
                f"Graph Evidence {index}: "
                f"{subject} -[{relationship}]-> "
                f"{object_name}. "
                f"Supporting text: {evidence}"
            )

        return "\n".join(
            part for part in context_parts
            if isinstance(part, str) and part.strip()
        )

    def generate_answer(self, query, retrieval_results):
        vector_results = retrieval_results.get(
            "vector_results", []
        ) or []

        graph_results = retrieval_results.get(
            "graph_results", []
        ) or []

        if not vector_results and not graph_results:
            return {
                "answer": (
                    "I could not find sufficient evidence "
                    "in the available enterprise documents."
                ),
                "sources": [],
                "grounded": False,
                "model": self.model_name
            }

        context = self.build_context(retrieval_results)

        prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the evidence below.

STRICT RULES:
1. Do not invent facts, names, or relationships.
2. Do not use outside knowledge.
3. If the evidence does not answer the question,
   clearly state that insufficient evidence is available.
4. Keep the answer concise and direct.
5. Use the supporting evidence when explaining the answer.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

ANSWER:
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            answer = response.text

            if not answer:
                answer = (
                    "The model returned an empty response. "
                    "Please try again."
                )

            return {
                "answer": answer.strip(),
                "sources": {
                    "document_chunks": len(vector_results),
                    "graph_relationships": len(graph_results)
                },
                "grounded": True,
                "model": self.model_name
            }

        except Exception as error:
            raise RuntimeError(
                f"Gemini generation failed using "
                f"'{self.model_name}': {error}"
            ) from error
