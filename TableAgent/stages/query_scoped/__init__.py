"""Query-scoped structure extraction backed by a reusable static workbook index."""

from .pipeline import QueryScopedStructurePipeline
from .static_ingestion import StaticWorkbookIndex, StaticWorkbookIngestor

__all__ = [
    "QueryScopedStructurePipeline",
    "StaticWorkbookIndex",
    "StaticWorkbookIngestor",
]
