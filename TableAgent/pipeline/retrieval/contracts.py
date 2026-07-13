from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import TableCandidate, TableSearchRequest


@runtime_checkable
class TableRetrieverContract(Protocol):
    """Contract implemented by table-level lexical/embedding/LLM retrievers."""

    def search(self, request: TableSearchRequest) -> list[TableCandidate]:
        ...


class TableRetriever:
    """Extension point for the table-level retriever implementation."""

    def search(self, request: TableSearchRequest) -> list[TableCandidate]:
        raise NotImplementedError("Table-level retrieval backend has not been implemented yet")
