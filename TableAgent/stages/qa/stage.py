"""The public boundary of the QA stage: its input, its output, and the stage object."""

from __future__ import annotations

from collections.abc import Callable
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
    fallback_prompt: str | Callable[[], str]  # a callable is built only if a fallback runs
    fallback_image_path: Path | None = None
    fallback_text_prompt: str | None = None
    related_structure_paths: tuple[Path, ...] = ()
    excluded_sheet_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class QAOutput:
    response: LLMResponse
    metadata: dict[str, Any] = field(default_factory=dict)


class QAStage:
    """Answers one question over a workbook (see workbook_qa.py)."""

    def __init__(self, answer: Callable[..., tuple]):
        self._answer = answer

    def run(self, stage_input: QAInput) -> QAOutput:
        response, metadata = self._answer(
            question=stage_input.question,
            structure_path=stage_input.structure_path,
            workbook_path=stage_input.workbook_path,
            qa_artifact_dir=stage_input.qa_artifact_dir,
            fallback_prompt=stage_input.fallback_prompt,
            fallback_image_path=stage_input.fallback_image_path,
            fallback_text_prompt=stage_input.fallback_text_prompt,
            related_structure_paths=list(stage_input.related_structure_paths),
            excluded_sheet_names=list(stage_input.excluded_sheet_names),
        )
        return QAOutput(response=response, metadata=metadata)
