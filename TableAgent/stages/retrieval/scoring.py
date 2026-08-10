from __future__ import annotations

from collections.abc import Sequence
import math
import re

import numpy as np


_BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "of", "on", "or", "the", "to", "what", "which", "with", "bảng",
    "các", "cho", "của", "là", "nào", "những", "trong", "và", "với",
}


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


def bm25_scores(
    query: str,
    documents: Sequence[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Score a bounded candidate set without requiring an external search service."""
    tokenized_documents = [_bm25_tokens(document) for document in documents]
    query_tokens = _bm25_tokens(query)
    if not tokenized_documents or not query_tokens:
        return [0.0] * len(tokenized_documents)

    document_count = len(tokenized_documents)
    average_length = max(
        sum(len(tokens) for tokens in tokenized_documents) / document_count,
        1.0,
    )
    frequencies = {
        token: sum(token in tokens for tokens in tokenized_documents)
        for token in set(query_tokens)
    }
    scores = []
    for tokens in tokenized_documents:
        term_frequency = {token: tokens.count(token) for token in set(tokens)}
        score = 0.0
        for token in query_tokens:
            frequency = term_frequency.get(token, 0)
            if not frequency:
                continue
            document_frequency = frequencies[token]
            inverse_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * len(tokens) / average_length
            )
            score += inverse_frequency * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return scores


def _bm25_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[\w]+", str(value).casefold(), flags=re.UNICODE)
        if token not in _BM25_STOPWORDS and (len(token) >= 2 or token.isdigit())
    ]
