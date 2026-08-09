from __future__ import annotations

from collections.abc import Callable

from .contracts import QAInput, QAOutput


class QAStage:
    """Public QA boundary backed by the existing verified QA implementation."""

    def __init__(self, run_verified: Callable[..., tuple]):
        self._run_verified = run_verified

    def run(self, stage_input: QAInput) -> QAOutput:
        response, metadata = self._run_verified(
            question=stage_input.question,
            structure_path=stage_input.structure_path,
            workbook_path=stage_input.workbook_path,
            qa_artifact_dir=stage_input.qa_artifact_dir,
            fallback_prompt=stage_input.fallback_prompt,
            fallback_image_path=stage_input.fallback_image_path,
            fallback_text_prompt=stage_input.fallback_text_prompt,
            related_structure_paths=list(stage_input.related_structure_paths),
            excluded_sheet_names=list(stage_input.excluded_sheet_names),
            enable_final_answer_review=stage_input.enable_final_answer_review,
        )
        return QAOutput(response=response, metadata=metadata)
