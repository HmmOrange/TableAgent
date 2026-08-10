from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.sample import EvalSample


@dataclass(frozen=True)
class SourceCandidate:
    directory: Path
    workbook_path: Path
    sheet_name: str
    image_path: Path
    html_path: Path | None
    structure_text: str
    sheet_text: str
    score: float
    lexical_score: float = 0.0
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    embedding_used: bool = False
    retrieval_card: str = ""
    table_id: str = ""
    table_name: str = ""
    table_description: str = ""
    entity_score: float = 0.0
    matched_terms: tuple[str, ...] = ()
    missing_terms: tuple[str, ...] = ()
    retrieval_rank: int = 0
    retrieval_type: str = "data"
    retrieval_level: str = "table"
    retrieval_trace: tuple[dict[str, Any], ...] = ()
    retrieval_audit: tuple[dict[str, Any], ...] = ()
    artifact_id: str = ""
    embedding_vector: tuple[float, ...] = ()
    embedding_model: str = ""
    embedding_source: str = ""
    sheet_names: tuple[str, ...] = ()
    workbook_reference_score: float = 0.0


@dataclass(frozen=True)
class TableSearchRequest:
    """Stable request passed from QA to a table retrieval backend."""

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


@runtime_checkable
class TableRetrieverContract(Protocol):
    """Contract implemented by table-level retrievers."""

    def search(self, request: TableSearchRequest) -> list[TableCandidate]:
        ...


class TableRetriever:
    """Extension point for a table-level retriever implementation."""

    def search(self, request: TableSearchRequest) -> list[TableCandidate]:
        raise NotImplementedError("Table-level retrieval backend has not been implemented yet")


@dataclass
class RetrievalInput:
    sample: EvalSample | None = None
    question: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    workbook_paths: dict[str, Path] = field(default_factory=dict)
    responses: list[LLMResponse] = field(default_factory=list)
    fit_context: Callable[[str], str] | None = None
    perfect: bool = False


@dataclass(frozen=True)
class RetrievalOutput:
    candidate: SourceCandidate | None
