import re
from dataclasses import dataclass
from typing import List, Sequence

from .text import tokenize


EN_EXCLUDE_PATTERNS = [
    re.compile(r"\bnot\s+(?:include|including|contain|containing|have|with)\s+([a-z0-9_-]+)"),
    re.compile(r"\bwithout\s+([a-z0-9_-]+)"),
    re.compile(r"\bexclude(?:ing)?\s+([a-z0-9_-]+)"),
    re.compile(r"\bno\s+([a-z0-9_-]+)"),
    re.compile(r"\bnot\s+([a-z0-9_-]+)"),
]
ZH_EXCLUDE_PATTERNS = [
    re.compile(r"(?:不含|不包括|不要|不需要)([\u4e00-\u9fffA-Za-z0-9_]+)"),
    re.compile(r"无([\u4e00-\u9fffA-Za-z0-9_]+)"),
]

EN_INCLUDE_PATTERNS = [
    re.compile(r"\bwith\s+([a-z0-9_-]+)"),
    re.compile(r"\binclude(?:s|d|ing)?\s+([a-z0-9_-]+)"),
    re.compile(r"\bcontain(?:s|ed|ing)?\s+([a-z0-9_-]+)"),
]
ZH_INCLUDE_PATTERNS = [
    re.compile(r"(?:包含|包括|带)([\u4e00-\u9fffA-Za-z0-9_]+)"),
]

EN_MAX_PATTERNS = [
    re.compile(r"\b(?:under|below|less than|at most)\s*\$?\s*(\d+(?:\.\d+)?)"),
]
EN_MIN_PATTERNS = [
    re.compile(r"\b(?:over|above|more than|at least)\s*\$?\s*(\d+(?:\.\d+)?)"),
]
ZH_MAX_PATTERNS = [
    re.compile(r"(?:低于|不高于|小于|最多)\s*\$?\s*(\d+(?:\.\d+)?)"),
]
ZH_MIN_PATTERNS = [
    re.compile(r"(?:高于|不少于|大于|至少)\s*\$?\s*(\d+(?:\.\d+)?)"),
]

DOC_NUMBER_PATTERN = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")
IGNORED_TERMS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "any",
    "very",
    "too",
    "include",
    "contains",
    "containing",
    "with",
}


@dataclass(frozen=True)
class ConstraintSpec:
    exclude_terms: Sequence[str]
    include_terms: Sequence[str]
    max_value: float | None
    min_value: float | None

    def has_constraints(self) -> bool:
        return bool(self.exclude_terms or self.include_terms or self.max_value is not None or self.min_value is not None)


class RuleBasedConstraintScorer:
    """
    Fast, model-free scorer for PoC.
    It captures lexical negation/include constraints and simple numeric bounds.
    """

    def parse(self, query: str) -> ConstraintSpec:
        lowered = query.lower()
        exclude_terms = self._extract_terms(lowered, EN_EXCLUDE_PATTERNS)
        include_terms = self._extract_terms(lowered, EN_INCLUDE_PATTERNS)
        exclude_terms += self._extract_terms(query, ZH_EXCLUDE_PATTERNS)
        include_terms += self._extract_terms(query, ZH_INCLUDE_PATTERNS)

        max_value = self._extract_numeric(lowered, EN_MAX_PATTERNS)
        min_value = self._extract_numeric(lowered, EN_MIN_PATTERNS)
        max_value = self._coalesce(max_value, self._extract_numeric(query, ZH_MAX_PATTERNS))
        min_value = self._coalesce(min_value, self._extract_numeric(query, ZH_MIN_PATTERNS))

        return ConstraintSpec(
            exclude_terms=self._unique(exclude_terms),
            include_terms=self._unique(include_terms),
            max_value=max_value,
            min_value=min_value,
        )

    def score(self, query: str, doc_text: str) -> float:
        spec = self.parse(query)
        return self.score_from_spec(spec, doc_text)

    def score_from_spec(self, spec: ConstraintSpec, doc_text: str) -> float:
        if not spec.has_constraints():
            return 1.0

        tokens = self._expanded_tokens(tokenize(doc_text))
        score = 1.0

        for term in spec.exclude_terms:
            if term in tokens and not self._matches_negated_context(doc_text, term):
                score *= 0.05

        for term in spec.include_terms:
            if term not in tokens:
                score *= 0.6

        doc_numbers = self._extract_doc_numbers(doc_text)
        if spec.max_value is not None:
            if not doc_numbers:
                score *= 0.7
            elif min(doc_numbers) > spec.max_value:
                score *= 0.1

        if spec.min_value is not None:
            if not doc_numbers:
                score *= 0.7
            elif max(doc_numbers) < spec.min_value:
                score *= 0.1

        return max(0.0, min(score, 1.0))

    def _extract_terms(self, text: str, patterns: List[re.Pattern[str]]) -> List[str]:
        matches: List[str] = []
        for pattern in patterns:
            for term in pattern.findall(text):
                normalized = self._canonicalize(term.lower().strip())
                if normalized and normalized not in IGNORED_TERMS:
                    matches.append(normalized)
        return matches

    def _extract_numeric(self, text: str, patterns: List[re.Pattern[str]]) -> float | None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        return None

    def _extract_doc_numbers(self, text: str) -> List[float]:
        values = []
        for raw in DOC_NUMBER_PATTERN.findall(text):
            try:
                values.append(float(raw))
            except ValueError:
                continue
        return values

    def _coalesce(self, lhs: float | None, rhs: float | None) -> float | None:
        return lhs if lhs is not None else rhs

    def _unique(self, values: Sequence[str]) -> List[str]:
        seen = set()
        output = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            output.append(value)
        return output

    def _expanded_tokens(self, tokens: Sequence[str]) -> set[str]:
        expanded = set()
        for token in tokens:
            base = self._canonicalize(token)
            if base:
                expanded.add(base)
        return expanded

    def _canonicalize(self, token: str) -> str:
        token = token.strip().lower()
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 3 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _matches_negated_context(self, text: str, term: str) -> bool:
        lowered = text.lower()
        variants = {term}
        if not term.endswith("s"):
            variants.add(f"{term}s")
        for variant in variants:
            escaped = re.escape(variant)
            patterns = [
                rf"\bwithout\s+{escaped}\b",
                rf"\bno\s+{escaped}\b",
                rf"\bnot\s+{escaped}\b",
                rf"\b{escaped}\s*-\s*free\b",
                rf"\bfree of\s+{escaped}\b",
                rf"不含{escaped}",
                rf"无{escaped}",
            ]
            if any(re.search(pattern, lowered) for pattern in patterns):
                return True
        return False
