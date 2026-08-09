from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any

from TableAgent.llm import LLMResponse
from TableAgent.schema import EvalSample
from TableAgent.stages.retrieval.contracts import SourceCandidate


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
