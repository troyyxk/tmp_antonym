from typing import Sequence

from .constraints import RuleBasedConstraintScorer
from .types import RetrievalResult


def constraint_compliance_rate(
    results: Sequence[RetrievalResult],
    min_constraint_score: float = 0.45,
    query: str | None = None,
    scorer: RuleBasedConstraintScorer | None = None,
) -> float:
    if not results:
        return 0.0

    if query is None:
        hit = sum(1 for result in results if result.constraint_score >= min_constraint_score)
        return hit / len(results)

    scorer = scorer or RuleBasedConstraintScorer()
    hit = sum(1 for result in results if scorer.score(query, result.document.text) >= min_constraint_score)
    return hit / len(results)
