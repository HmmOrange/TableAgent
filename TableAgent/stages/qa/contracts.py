from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from TableAgent.llm import LLMResponse


@dataclass(frozen=True)
class QAInput:
    question: str
    structure_path: Path
    workbook_path: Path
    qa_artifact_dir: Path
    fallback_prompt: str
    fallback_image_path: Path | None = None
    fallback_text_prompt: str | None = None
    related_structure_paths: tuple[Path, ...] = ()
    excluded_sheet_names: tuple[str, ...] = ()
    enable_final_answer_review: bool = False


@dataclass(frozen=True)
class QAOutput:
    response: LLMResponse
    metadata: dict[str, Any] = field(default_factory=dict)
