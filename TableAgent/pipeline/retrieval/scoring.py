from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = np.asarray(left, dtype=np.float32)
    right_vector = np.asarray(right, dtype=np.float32)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


def normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    score_range = max(scores) - minimum
    if score_range == 0:
        return [1.0] * len(scores)
    return [(score - minimum) / score_range for score in scores]


def hybrid_score(
    lexical_score: float,
    embedding_score: float,
    *,
    lexical_weight: float,
    embedding_weight: float,
) -> float:
    return lexical_weight * lexical_score + embedding_weight * embedding_score
