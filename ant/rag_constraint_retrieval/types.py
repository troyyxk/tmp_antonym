from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    topical_score: float
    constraint_score: float
    final_score: float
