from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from TableAgent.configs import TableAgentConfig
from TableAgent.rendering.workbook import WorkbookRenderer
from TableAgent.stages.structure.group_enrichment import GroupEnrichmentStage
from TableAgent.stages.structure.layout.agent import LayoutAgent
from TableAgent.stages.structure.layout.parsing import nullify_structure_ranges
from TableAgent.stages.structure.metadata import SheetMetadata
from TableAgent.stages.structure.verification import DeterministicVerifier


@dataclass(frozen=True)
class LayoutWorkflowResult:
    structure_text: str
    verification: dict[str, Any]
    iterations: int
    image_path: Path | None
    changelog_path: Path
    events_path: Path
    responses: list[Any]


class TableLayoutWorkflow:
    """Render the complete used range once and refine it with verification."""

    def __init__(self, settings: TableAgentConfig, renderer: WorkbookRenderer, layout_agent: LayoutAgent, verifier: DeterministicVerifier, progress_callback: Callable[..., None] | None = None, *, group_enrichment_stage: GroupEnrichmentStage | None = None):
        self.settings = settings
        self.renderer = renderer
        self.layout_agent = layout_agent
        self.verifier = verifier
        self.group_enrichment_stage = group_enrichment_stage
        self.progress_callback = progress_callback

    def set_progress_callback(self, callback: Callable[..., None] | None) -> None:
        self.progress_callback = callback

    def _progress(self, stage: str, **fields: Any) -> None:
        if self.progress_callback:
            self.progress_callback(stage=stage, **fields)

    def run(self, *, workbook_path: Path, sheet_name: str, metadata: SheetMetadata, output_dir: Path) -> LayoutWorkflowResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        iterations_dir = output_dir / "iterations"
        iterations_dir.mkdir(parents=True, exist_ok=True)
        structure_path = output_dir / "structure.yaml"
        changelog_path = output_dir / "changelog.md"
        events_path = output_dir / "events.jsonl"
        metadata_text = metadata.to_yaml()
        (output_dir / "metadata.yaml").write_text(metadata_text, encoding="utf-8")
        structure_text = structure_path.read_text(encoding="utf-8") if structure_path.is_file() else ""
        sheet_range = metadata.used_range or "A1:A1"
        image_path = output_dir / "table.png"
        self._progress("render", workbook=workbook_path.name, sheet=sheet_name, range=sheet_range, iteration=1)
        render_result = self.renderer.source_viewport_to_image(workbook_path, sheet_name, sheet_range, image_path)
        if render_result.html_path and render_result.html_path.is_file():
            (output_dir / "table.html").write_text(render_result.html_path.read_text(encoding="utf-8"), encoding="utf-8")
        render_metadata = image_path.with_suffix(".metadata.json")
        if render_metadata.is_file():
            (output_dir / "table.metadata.json").write_text(render_metadata.read_text(encoding="utf-8"), encoding="utf-8")
        responses: list[Any] = []
        changes: list[str] = []
        verification_info: dict[str, Any] = {"status": "not_good", "feedback": "No layout has been verified.", "sheet_range": sheet_range}
        feedback = ""
        iteration = 0
        while iteration < max(1, self.settings.max_retry):
            iteration += 1
            iteration_dir = iterations_dir / f"{iteration:04d}_{sheet_range.replace(':', '_')}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            (iteration_dir / "structure_before.yaml").write_text(structure_text, encoding="utf-8")
            fields = {"workbook": workbook_path.name, "sheet": sheet_name, "range": sheet_range, "iteration": iteration}
            self._progress("layout", **fields)
            layout = self.layout_agent.run(metadata_text=metadata_text, structure_text=structure_text, image_path=image_path, sheet_range=sheet_range, feedback=feedback, iteration=iteration, iteration_dir=iteration_dir)
            responses.append(layout.response)
            structure_text = layout.structure_text
            (iteration_dir / "structure_after.yaml").write_text(structure_text, encoding="utf-8")
            (iteration_dir / "changelog.md").write_text(layout.changelog + "\n", encoding="utf-8")
            if layout.discarded:
                (iteration_dir / "layout_discarded.txt").write_text(layout.discarded, encoding="utf-8")
            self._progress("verify", **fields)
            verification = self.verifier.run(workbook_path=workbook_path, sheet_name=sheet_name, structure_text=structure_text, iteration_dir=iteration_dir, preflight_errors=layout.preflight_errors)
            structure_text = verification.structure_text
            verification_info = {"status": verification.status, "feedback": verification.feedback, "sheet_range": sheet_range}
            self._append_event(events_path, {"iteration": iteration, "sheet_range": sheet_range, "changed": layout.changed, "layout_token_capped": layout.response.token_capped, "verification": verification_info})
            if verification.is_good:
                if layout.changed:
                    changes.append(f"## Iteration {iteration} - {sheet_range}\n\n{layout.changelog}")
                break
            feedback = verification.feedback
            if iteration >= self.settings.max_retry:
                structure_text = nullify_structure_ranges(structure_text, verification.null_fields)
        if self.group_enrichment_stage is not None and structure_text.strip():
            group_dir = output_dir / "groups"
            self._progress("groups", workbook=workbook_path.name, sheet=sheet_name, range=metadata.used_range)
            enrichment = self.group_enrichment_stage.run(workbook_path=workbook_path, sheet_name=sheet_name, viewport_range=metadata.used_range, structure_text=structure_text, artifact_dir=group_dir)
            responses.extend(enrichment.responses)
            structure_text = enrichment.structure_text
            verification = self.verifier.run(workbook_path=workbook_path, sheet_name=sheet_name, structure_text=structure_text, iteration_dir=group_dir, preflight_errors=enrichment.preflight_errors)
            structure_text = verification.structure_text
            verification_info = {"status": verification.status, "feedback": verification.feedback, "sheet_range": metadata.used_range, "stage": "groups"}
            self._append_event(events_path, {"stage": "groups", "verification": verification_info})
        if structure_text.strip():
            structure_path.write_text(structure_text, encoding="utf-8")
        changelog_path.write_text("\n\n".join(changes).strip() + "\n" if changes else "No change.\n", encoding="utf-8")
        return LayoutWorkflowResult(structure_text, verification_info, iteration, image_path, changelog_path, events_path, responses)

    @staticmethod
    def _append_event(path: Path, event: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
