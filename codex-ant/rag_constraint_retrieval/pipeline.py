from dataclasses import dataclass
from typing import List, Sequence

from .constraints import RuleBasedConstraintScorer
from .topical import LexicalTopicalRetriever, TopicalRetrieverProtocol
from .types import Document, RetrievalResult


@dataclass(frozen=True)
class RetrievalConfig:
    first_stage_k: int = 50
    final_k: int = 10
    topical_weight: float = 0.65
    constraint_weight: float = 0.35
    hard_filter: bool = True
    min_constraint_score: float = 0.45


class ConstraintAwareRetriever:
    """
    Two-stage retrieval pipeline:
    1) topical recall
    2) constraint-aware rerank/filter
    """

    def __init__(
        self,
        documents: Sequence[Document],
        topical_retriever: TopicalRetrieverProtocol | None = None,
        constraint_scorer: RuleBasedConstraintScorer | None = None,
    ):
        self.documents = list(documents)
        self.topical_retriever = topical_retriever or LexicalTopicalRetriever(self.documents)
        self.constraint_scorer = constraint_scorer or RuleBasedConstraintScorer()

    def baseline_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        stage1 = self.topical_retriever.retrieve(query=query, top_k=top_k)
        if not stage1:
            return []

        max_topical = max(score for _, score in stage1) or 1.0
        results = []
        for doc, topical_score in stage1:
            normalized_topical = topical_score / max_topical
            results.append(
                RetrievalResult(
                    document=doc,
                    topical_score=topical_score,
                    constraint_score=1.0,
                    final_score=normalized_topical,
                )
            )
        return results

    def search(self, query: str, config: RetrievalConfig | None = None) -> List[RetrievalResult]:
        cfg = config or RetrievalConfig()
        stage1 = self.topical_retriever.retrieve(query=query, top_k=cfg.first_stage_k)
        if not stage1:
            return []

        max_topical = max(score for _, score in stage1) or 1.0
        weight_sum = cfg.topical_weight + cfg.constraint_weight
        if weight_sum <= 0.0:
            topical_weight = 0.5
            constraint_weight = 0.5
        else:
            topical_weight = cfg.topical_weight / weight_sum
            constraint_weight = cfg.constraint_weight / weight_sum

        reranked: List[RetrievalResult] = []
        for doc, topical_score in stage1:
            constraint_score = self.constraint_scorer.score(query, doc.text)
            if cfg.hard_filter and constraint_score < cfg.min_constraint_score:
                continue

            normalized_topical = topical_score / max_topical
            final_score = topical_weight * normalized_topical + constraint_weight * constraint_score
            reranked.append(
                RetrievalResult(
                    document=doc,
                    topical_score=topical_score,
                    constraint_score=constraint_score,
                    final_score=final_score,
                )
            )

        reranked.sort(key=lambda item: item.final_score, reverse=True)
        return reranked[: max(cfg.final_k, 0)]
