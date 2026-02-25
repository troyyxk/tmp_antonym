import math
import re
from collections import Counter
from typing import Dict, Iterable, List


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [t.strip() for t in tokens if t.strip()]


def term_frequency(tokens: Iterable[str]) -> Dict[str, float]:
    counter = Counter(tokens)
    total = sum(counter.values())
    if total == 0:
        return {}
    return {token: count / total for token, count in counter.items()}


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    # Iterate over the smaller vector for lower cost.
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def contains_term(text: str, term: str) -> bool:
    tokens = set(tokenize(text))
    return term.lower() in tokens
