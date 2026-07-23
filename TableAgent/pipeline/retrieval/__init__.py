"""Retrieval APIs with compatibility exports for the existing SourceRetriever."""

from .cards import (
    _build_retrieval_card,
    _extract_headers_text,
    build_metadata_retrieval_card,
    build_sheet_metadata_payload,
    build_source_retrieval_card,
    build_table_retrieval_cards,
    extract_columns,
    extract_headers_text,
)
from .contracts import (
    TableRetriever,
    TableRetrieverContract,
)
from .embeddings import MockEmbeddingModel, OpenAICompatibleEmbeddingClient
from .models import SourceCandidate, TableCandidate, TableSearchRequest
from .reranking import choose_from_reranker
from .scoring import cosine_similarity, hybrid_score, normalize_scores
from .source_retriever import SourceRetriever


__all__ = [
    "MockEmbeddingModel",
    "OpenAICompatibleEmbeddingClient",
    "SourceCandidate",
    "SourceRetriever",
    "TableCandidate",
    "TableRetriever",
    "TableRetrieverContract",
    "TableSearchRequest",
    "build_metadata_retrieval_card",
    "build_sheet_metadata_payload",
    "build_source_retrieval_card",
    "build_table_retrieval_cards",
    "choose_from_reranker",
    "cosine_similarity",
    "extract_columns",
    "extract_headers_text",
    "hybrid_score",
    "normalize_scores",
]
