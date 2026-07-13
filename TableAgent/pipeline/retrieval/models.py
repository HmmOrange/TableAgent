from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from TableAgent.pipeline.common import SourceCandidate


@dataclass(frozen=True)
class TableSearchRequest:
    """Stable request passed from QA operators to a table retrieval backend."""

    query: str
    top_k: int = 5
    allowed_table_ids: tuple[str, ...] = ()
    workbook_paths: tuple[Path, ...] = ()
    sheet_names: tuple[str, ...] = ()
    required_headers: tuple[str, ...] = ()
    rerank: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TableCandidate:
    """Retriever-independent description of one ranked table candidate."""

    table_id: str
    workbook_path: Path | None = None
    sheet_name: str = ""
    table_name: str = ""
    description: str = ""
    structure_path: Path | None = None
    score: float = 0.0
    lexical_score: float = 0.0
    embedding_score: float = 0.0
    reranker_score: float | None = None
    reason: str = ""
    retrieval_card: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["SourceCandidate", "TableCandidate", "TableSearchRequest"]
