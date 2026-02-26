import math
from typing import List, Protocol, Sequence, Tuple

from .embeddings import Embedder
from .text import cosine_similarity, term_frequency, tokenize
from .types import Document


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "without",
    "was",
    "were",
    "with",
    "include",
    "includes",
    "including",
    "contain",
    "contains",
    "containing",
}


class TopicalRetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[Document, float]]:
        ...


class LexicalTopicalRetriever:
    """
    Lightweight lexical retriever used for stage-1 recall in the PoC.
    Replace this class with a dense retriever in production.
    """

    def __init__(self, documents: Sequence[Document]):
        self.documents: List[Document] = list(documents)
        self._doc_vectors = [term_frequency(self._content_tokens(doc.text)) for doc in self.documents]

    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[Document, float]]:
        query_vec = term_frequency(self._content_tokens(query))
        scored = []
        for doc, doc_vec in zip(self.documents, self._doc_vectors):
            score = cosine_similarity(query_vec, doc_vec)
            if score > 0.0:
                scored.append((doc, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(top_k, 0)]

    def _content_tokens(self, text: str) -> List[str]:
        return [token for token in tokenize(text) if token not in STOPWORDS]


class DenseTopicalRetriever:
    """
    Dense stage-1 retriever driven by a pluggable embedding backend.
    """

    def __init__(self, documents: Sequence[Document], embedder: Embedder):
        self.documents: List[Document] = list(documents)
        self.embedder = embedder
        self._doc_vectors = self._normalize_batch(self.embedder.embed([doc.text for doc in self.documents]))

    def retrieve(self, query: str, top_k: int = 50) -> List[Tuple[Document, float]]:
        query_vector = self._normalize(self.embedder.embed([query])[0])
        scored = []
        for doc, doc_vector in zip(self.documents, self._doc_vectors):
            score = self._dot(query_vector, doc_vector)
            scored.append((doc, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(top_k, 0)]

    def _normalize_batch(self, vectors: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self._normalize(vector) for vector in vectors]

    def _normalize(self, vector: Sequence[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return [0.0 for _ in vector]
        return [value / norm for value in vector]

    def _dot(self, vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        return sum(value_a * value_b for value_a, value_b in zip(vec_a, vec_b))


# Backward-compatible alias so previous imports still work.
TopicalRetriever = LexicalTopicalRetriever
