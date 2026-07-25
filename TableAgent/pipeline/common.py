from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from TableAgent.llm import LLMResponse
from TableAgent.schema import EvalSample


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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)) or "item"


def display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


_WORKBOOK_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}


def has_workbook_sources(sample: EvalSample) -> bool:
    """Identify source-retrieval inputs from their workbook paths, not a dataset name."""
    paths = [part.strip() for part in str(sample.table_path or "").split(";") if part.strip()]
    return bool(paths) and all(PureWindowsPath(path).suffix.lower() in _WORKBOOK_SUFFIXES for path in paths)


def token_usage(responses: list[LLMResponse]) -> dict[str, int]:
    return {
        "prompt": sum(response.prompt_tokens for response in responses),
        "completion": sum(response.completion_tokens for response in responses),
    }


def read_image_tiles(directory: Path) -> list[dict[str, Any]]:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tiles = metadata.get("image_tiles", [])
        return tiles if isinstance(tiles, list) else []
    except Exception:
        return []
