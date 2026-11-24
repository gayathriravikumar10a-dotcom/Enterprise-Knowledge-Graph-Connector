
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class GroundedAnswerGenerator:

    def __init__(self, model_name=None):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is missing from the .env file."
            )

        self.model_name = (
            model_name
            or os.getenv("OPENROUTER_MODEL", "openrouter/free")
        )

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.app_name = os.getenv(
            "OPENROUTER_APP_NAME",
            "Enterprise Knowledge Graph Connector"
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
6. Clearly distinguish between separate projects.
7. Do not infer missing information.

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

ANSWER:
"""

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url or "http://localhost",
            "X-Title": self.app_name
        }

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=90
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            choices = result.get("choices", [])

            if not choices:
                raise RuntimeError(
                    "OpenRouter returned no answer choices."
                )

            answer = (
                choices[0]
                .get("message", {})
                .get("content", "")
            )

            if isinstance(answer, list):
                answer = " ".join(
                    item.get("text", "")
                    for item in answer
                    if isinstance(item, dict)
                )

            if not answer or not str(answer).strip():
                answer = (
                    "The model returned an empty response. "
                    "Please try again."
                )

            return {
                "answer": str(answer).strip(),
                "sources": {
                    "document_chunks": len(vector_results),
                    "graph_relationships": len(graph_results)
                },
                "grounded": True,
                "model": self.model_name
            }

        except urllib.error.HTTPError as error:
            error_body = error.read().decode(
                "utf-8", errors="replace"
            )

            raise RuntimeError(
                f"OpenRouter API failed with HTTP "
                f"{error.code}: {error_body}"
            ) from error

        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not connect to OpenRouter: {error.reason}"
            ) from error

        except Exception as error:
            raise RuntimeError(
                f"OpenRouter generation failed using "
                f"'{self.model_name}': {error}"
            ) from error
