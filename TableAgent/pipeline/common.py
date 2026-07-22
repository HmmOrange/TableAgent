from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets.base import EvalSample
from utils.llm.base import LLMResponse


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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)) or "item"


def display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def is_siflex(sample: EvalSample) -> bool:
    return "siflex" in str(sample.sample_path or sample.sample_id).lower()


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
