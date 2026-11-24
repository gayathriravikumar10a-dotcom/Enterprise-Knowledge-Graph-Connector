
import re


class GroundingEvaluator:
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "has", "have", "had",
        "of", "to", "in", "on", "at", "by", "for", "from", "with",
        "and", "or", "but", "as", "that", "this", "these", "those",
        "who", "what", "which", "when", "where", "how",
        "based", "provided", "evidence", "available",
        "according", "answer", "information","section", "states", "explicitly", "while",
    }

    NORMALIZATION = {
        "manages": "manage",
        "managed": "manage",
        "manager": "manage",
        "managing": "manage",
        "works": "work",
        "working": "work",
        "worked": "work",
        "employees": "employee",
        "projects": "project",
        "teams": "team",
        "departments": "department",
        "belongs": "belong",
        "belonged": "belong",
        "leading": "lead",
        "leads": "lead",
        "led": "lead",
    }

    def __init__(self, threshold=0.80):
        if not 0 < threshold <= 1:
            raise ValueError(
                "Threshold must be greater than 0 and at most 1."
            )
        self.threshold = threshold

    @staticmethod
    def safe_text(value):
        return "" if value is None else str(value)

    def clean_answer(self, answer):
        """
        Evaluate factual answer content rather than source labels,
        evidence annotations, or explanatory wording.
        """
        text = self.safe_text(answer).strip()

        # Preserve visible text from Markdown links.
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove Markdown formatting.
        text = re.sub(r"[*_`~]+", "", text)

        # Remove parenthetical source references.
        text = re.sub(
            r"\([^)]*(?:document\s+sources?|graph\s+evidence|source\s+\d)[^)]*\)",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        # Remove appended evidence and metadata sections.
        metadata_patterns = [
            r"\bSupporting\s+Evidence\s*:",
            r"\bKnowledge\s+(?:Graph\s+)?Evidence\s*:",
            r"\bGraph\s+Evidence\s*:",
            r"\bRetrieved\s+Evidence\s*:",
            r"\bSource\s+Details\s*:",
            r"\bKnowledge\s*&\s*\d+\s*state\s*:",
            r"\b&\s*\d+\s*state\s*:",
        ]

        positions = []

        for pattern in metadata_patterns:
            match = re.search(
                pattern, text, flags=re.IGNORECASE
            )
            if match:
                positions.append(match.start())

        if positions:
            text = text[:min(positions)]

        # Remove source-attribution wording, not factual content.
        text = re.sub(
            r"\baccording\s+to\s+document\s+source\s+\d+\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\baccording\s+to\s+(?:the\s+)?(?:provided\s+)?sources?\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # "Explicitly listed as separate" expresses the same
        # factual relationship as "are separate".
        text = re.sub(
            r"\bexplicitly\s+listed\s+as\b",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Normalize a leading "No," because it is an answer
        # introduction, not an independently supported fact.
        text = re.sub(
            r"^\s*no\s*[,.:]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Normalize the factual meaning:
        # "are not the same project" -> "are separate project".
        # This allows comparison with evidence saying "separate projects".
        text = re.sub(
            r"\b(?:is|are|was|were)\s+not\s+the\s+same\b",
            lambda match: match.group(0).split()[0] + " separate",
            text,
            flags=re.IGNORECASE,
        )

        # Handle the shorter wording "not the same".
        text = re.sub(
            r"\bnot\s+the\s+same\b",
            "separate",
            text,
            flags=re.IGNORECASE,
        )

        # Existing introductory wording cleanup.
        text = re.sub(
            r"\bbased on the provided evidence\b[:,]?",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\balso\s+identified\s+as\b",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text):
        words = re.findall(
            r"[a-zA-Z0-9]+",
            self.safe_text(text).lower()
        )

        normalized = []

        for word in words:
            word = self.NORMALIZATION.get(word, word)

            if word not in self.STOP_WORDS:
                normalized.append(word)

        return set(normalized)

    def split_claims(self, answer):
        cleaned = self.clean_answer(answer)

        if not cleaned:
            return []

        parts = re.split(
            r"(?<=[.!?])\s+|[\n;]+",
            cleaned
        )

        return [
            part.strip()
            for part in parts
            if part.strip() and self.tokenize(part)
        ]

    def collect_evidence(self, retrieval_results):
        evidence_units = []
        retrieval_results = retrieval_results or {}

        for item in retrieval_results.get("vector_results", []):
            text = self.safe_text(item.get("text", "")).strip()

            if text:
                evidence_units.append(text)

        for item in retrieval_results.get("graph_results", []):
            subject = self.safe_text(item.get("subject", ""))
            relationship = self.safe_text(
                item.get("relationship", "")
            )
            obj = self.safe_text(item.get("object", ""))
            evidence = self.safe_text(item.get("evidence", ""))

            graph_text = " ".join(
                part
                for part in [
                    subject,
                    relationship,
                    obj,
                    evidence,
                ]
                if part
            ).strip()

            if graph_text:
                evidence_units.append(graph_text)

        return evidence_units

    def evaluate_claim(self, claim, evidence_units):
        claim_terms = self.tokenize(claim)

        if not claim_terms or not evidence_units:
            return {
                "claim": claim,
                "supported": False,
                "support_ratio": 0.0,
                "supported_terms": [],
                "unsupported_terms": sorted(claim_terms),
            }

        best_supported_terms = set()
        best_ratio = 0.0

        for evidence in evidence_units:
            evidence_terms = self.tokenize(evidence)
            matched = claim_terms.intersection(evidence_terms)
            ratio = len(matched) / len(claim_terms)

            if ratio > best_ratio:
                best_ratio = ratio
                best_supported_terms = matched

        unsupported_terms = claim_terms - best_supported_terms

        return {
            "claim": claim,
            "supported": best_ratio >= self.threshold,
            "support_ratio": round(best_ratio, 4),
            "supported_terms": sorted(best_supported_terms),
            "unsupported_terms": sorted(unsupported_terms),
        }

    def evaluate_answer(self, answer, retrieval_results):
        claims = self.split_claims(answer)
        evidence_units = self.collect_evidence(retrieval_results)

        if not claims or not evidence_units:
            return {
                "grounded": False,
                "support_ratio": 0.0,
                "supported_terms": [],
                "unsupported_terms": [],
                "claims": [],
                "unsupported_claims": claims,
            }

        claim_results = [
            self.evaluate_claim(claim, evidence_units)
            for claim in claims
        ]

        ratios = [
            result["support_ratio"]
            for result in claim_results
        ]

        supported_terms = set()
        unsupported_terms = set()

        for result in claim_results:
            supported_terms.update(result["supported_terms"])
            unsupported_terms.update(result["unsupported_terms"])

        grounded = all(
            result["supported"]
            for result in claim_results
        )

        return {
            "grounded": grounded,
            "support_ratio": round(
                sum(ratios) / len(ratios), 4
            ),
            "supported_terms": sorted(supported_terms),
            "unsupported_terms": sorted(unsupported_terms),
            "claims": claim_results,
            "unsupported_claims": [
                result["claim"]
                for result in claim_results
                if not result["supported"]
            ],
        }
