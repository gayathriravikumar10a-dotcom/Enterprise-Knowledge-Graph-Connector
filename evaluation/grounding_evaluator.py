class GroundingEvaluator:

    STOP_WORDS = {
        "the", "is", "a", "an", "and", "or", "to", "of",
        "in", "on", "for", "with", "this", "that", "it",
        "as", "by", "from", "be", "are", "was", "were",
        "based", "provided", "evidence", "source",
        "document", "graph", "states", "indicates",
        "explicitly", "according", "information",
        "supporting", "text", "answer", "available"
    }

    def __init__(self, threshold=0.60):
        self.threshold = threshold

    @staticmethod
    def safe_text(value):
        if value is None:
            return ""

        if isinstance(value, (list, tuple, set)):
            return " ".join(
                str(item) for item in value if item is not None
            )

        return str(value)

    def tokenize(self, text):
        text = self.safe_text(text).lower()

        cleaned = "".join(
            character if character.isalnum() else " "
            for character in text
        )

        return {
            word
            for word in cleaned.split()
            if word and word not in self.STOP_WORDS
        }

    def evaluate_answer(self, answer, retrieval_results):

        answer_terms = self.tokenize(answer)

        evidence_parts = []

        for item in retrieval_results.get("vector_results", []) or []:
            if isinstance(item, str):
                evidence_parts.append(item)

            elif isinstance(item, dict):
                for key in ("text", "document", "content"):
                    value = item.get(key)

                    if value is not None:
                        evidence_parts.append(self.safe_text(value))

        for item in retrieval_results.get("graph_results", []) or []:
            if isinstance(item, str):
                evidence_parts.append(item)

            elif isinstance(item, dict):
                for key in (
                    "subject",
                    "relationship",
                    "object",
                    "evidence"
                ):
                    value = item.get(key)

                    if value is not None:
                        evidence_parts.append(self.safe_text(value))

        evidence_text = " ".join(
            part for part in evidence_parts
            if isinstance(part, str) and part.strip()
        )

        evidence_terms = self.tokenize(evidence_text)

        if not answer_terms or not evidence_terms:
            return {
                "grounded": False,
                "support_ratio": 0.0,
                "supported_terms": [],
                "unsupported_terms": sorted(answer_terms)
            }

        supported_terms = answer_terms.intersection(evidence_terms)
        unsupported_terms = answer_terms.difference(evidence_terms)

        support_ratio = round(
            len(supported_terms) / len(answer_terms),
            4
        )

        return {
            "grounded": support_ratio >= self.threshold,
            "support_ratio": support_ratio,
            "supported_terms": sorted(supported_terms),
            "unsupported_terms": sorted(unsupported_terms)
        }
