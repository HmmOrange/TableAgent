from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import openpyxl
from openpyxl.utils.cell import range_boundaries

from TableAgent.agents import LayoutAgent, VerificationAgent
from TableAgent.config import TableAgentConfig
from TableAgent.perception.metadata import SheetMetadata
from TableAgent.perception.structure import nullify_structure_ranges
from TableAgent.pipeline.traversal import (
    Direction,
    DirectionQueue,
    TraversalTask,
    Viewport,
    frontier_directions,
    initial_viewport,
)
from TableAgent.rendering.workbook import WorkbookRenderer


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
    """Priority-queue orchestrator for LayoutAgent and VerificationAgent."""

    def __init__(
        self,
        settings: TableAgentConfig,
        renderer: WorkbookRenderer,
        layout_agent: LayoutAgent,
        verification_agent: VerificationAgent,
        progress_callback: Callable[..., None] | None = None,
    ):
        self.settings = settings
        self.renderer = renderer
        self.layout_agent = layout_agent
        self.verification_agent = verification_agent
        self.progress_callback = progress_callback

    def set_progress_callback(self, callback: Callable[..., None] | None) -> None:
        self.progress_callback = callback

    def _progress(self, stage: str, **fields: Any) -> None:
        if self.progress_callback:
            self.progress_callback(stage=stage, **fields)

    def run(
        self,
        *,
        workbook_path: Path,
        sheet_name: str,
        metadata: SheetMetadata,
        output_dir: Path,
    ) -> LayoutWorkflowResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        iterations_dir = output_dir / "iterations"
        iterations_dir.mkdir(parents=True, exist_ok=True)
        structure_path = output_dir / "structure.yaml"
        changelog_path = output_dir / "changelog.md"
        events_path = output_dir / "events.jsonl"
        metadata_text = metadata.to_yaml()
        (output_dir / "metadata.yaml").write_text(metadata_text, encoding="utf-8")

        structure_text = structure_path.read_text(encoding="utf-8") if structure_path.is_file() else ""
        table_range = metadata.used_range
        start = initial_viewport(
            table_range,
            rows=self.settings.viewport_rows,
            columns=self.settings.viewport_columns,
        )
        queue = DirectionQueue()
        queue.push(TraversalTask(Direction.STAY, start))
        successful_viewports: set[tuple[int, int]] = set()
        retries: dict[tuple[int, int], int] = {}
        zero_change_runs = {direction: 0 for direction in Direction}
        feedback_by_viewport: dict[tuple[int, int], str] = {}
        cumulative_changes: list[str] = []
        last_verification: dict[str, Any] = {
            "status": "not_good",
            "feedback": "No viewport has been verified.",
        }
        responses: list[Any] = []
        first_image: Path | None = None
        iteration = 0

        while queue:
            task = queue.pop()
            if task.direction != Direction.STAY and task.viewport.key in successful_viewports:
                continue
            viewport_range = task.viewport.clipped_a1_range(table_range)
            iteration += 1
            progress_fields = {
                "workbook": workbook_path.name,
                "sheet": sheet_name,
                "range": viewport_range,
                "iteration": iteration,
                "direction": task.direction.name.lower(),
            }
            self._progress("render", **progress_fields)
            iteration_dir = iterations_dir / (
                f"{iteration:04d}_{task.direction.name.lower()}_"
                f"{viewport_range.replace(':', '_')}"
            )
            iteration_dir.mkdir(parents=True, exist_ok=True)
            image_path = iteration_dir / "viewport.png"
            render_result = self.renderer.source_viewport_to_image(
                workbook_path,
                sheet_name,
                viewport_range,
                image_path,
            )
            if first_image is None:
                first_image = image_path
                (output_dir / "table.png").write_bytes(image_path.read_bytes())
                render_metadata_path = image_path.with_suffix(".metadata.json")
                if render_metadata_path.is_file():
                    (output_dir / "table.metadata.json").write_text(
                        render_metadata_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                if render_result.html_path and render_result.html_path.is_file():
                    (output_dir / "table.html").write_text(
                        render_result.html_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )

            (iteration_dir / "structure_before.yaml").write_text(structure_text, encoding="utf-8")
            self._progress("layout", **progress_fields)
            layout = self.layout_agent.run(
                metadata_text=metadata_text,
                structure_text=structure_text,
                image_path=image_path,
                viewport_range=viewport_range,
                direction=task.direction.name.lower(),
                feedback=feedback_by_viewport.get(task.viewport.key, ""),
                iteration=iteration,
                iteration_dir=iteration_dir,
            )
            responses.append(layout.response)
            structure_text = layout.structure_text
            (iteration_dir / "structure_after.yaml").write_text(structure_text, encoding="utf-8")
            (iteration_dir / "changelog.md").write_text(layout.changelog + "\n", encoding="utf-8")
            if layout.discarded:
                (iteration_dir / "layout_discarded.txt").write_text(layout.discarded, encoding="utf-8")

            self._progress("verify", **progress_fields)
            verification = self.verification_agent.run(
                workbook_path=workbook_path,
                sheet_name=sheet_name,
                metadata_text=metadata_text,
                structure_text=structure_text,
                changelog=layout.changelog,
                viewport_range=viewport_range,
                iteration=iteration,
                iteration_dir=iteration_dir,
            )
            structure_text = verification.structure_text
            responses.append(verification.response)
            last_verification = {
                "status": verification.status,
                "feedback": verification.feedback,
                "viewport": viewport_range,
            }
            self._append_event(events_path, {
                "iteration": iteration,
                "direction": task.direction.name.lower(),
                "viewport": viewport_range,
                "changed": layout.changed,
                "layout_token_capped": layout.response.token_capped,
                "layout_directions": layout.directions,
                "verification_token_capped": verification.response.token_capped,
                "verification": last_verification,
                "queue_size": len(queue),
            })

            if not verification.is_good:
                retry_count = retries.get(task.viewport.key, 0) + 1
                retries[task.viewport.key] = retry_count
                feedback_by_viewport[task.viewport.key] = verification.feedback
                if retry_count < self.settings.max_retry:
                    queue.push(TraversalTask(Direction.STAY, task.viewport))
                else:
                    structure_text = nullify_structure_ranges(structure_text, verification.null_fields)
                    (iteration_dir / "structure_after.yaml").write_text(structure_text, encoding="utf-8")
                    cumulative_changes.append(
                        f"## Iteration {iteration} — {viewport_range}\n\n"
                        "Verification retries exhausted; unverifiable ranges were set to null."
                    )
                    # Do not let one imperfect viewport truncate the rest of a larger
                    # sheet. Continue into every in-bounds frontier so later viewports
                    # can extend and repair the accumulated structure.
                    successful_viewports.add(task.viewport.key)
                    frontier = frontier_directions(table_range, task.viewport)
                    suggested = [Direction.parse(value) for value in layout.directions]
                    discovered = [direction for direction in suggested if direction in frontier]
                    for direction in discovered:
                        if direction != Direction.STAY:
                            self._enqueue_shift(
                                queue,
                                task.viewport,
                                direction,
                                successful_viewports,
                                table_range,
                                workbook_path,
                                sheet_name,
                            )
                continue

            structure_path.write_text(structure_text, encoding="utf-8")
            successful_viewports.add(task.viewport.key)
            retries.pop(task.viewport.key, None)
            feedback_by_viewport.pop(task.viewport.key, None)
            if layout.changed:
                cumulative_changes.append(
                    f"## Iteration {iteration} — {task.direction.name.lower()} "
                    f"{viewport_range}\n\n{layout.changelog}"
                )

            frontier = frontier_directions(table_range, task.viewport)
            if task.direction != Direction.STAY and task.direction in frontier:
                if layout.changed:
                    zero_change_runs[task.direction] = 0
                    self._enqueue_shift(
                        queue,
                        task.viewport,
                        task.direction,
                        successful_viewports,
                        table_range,
                        workbook_path,
                        sheet_name,
                    )
                else:
                    zero_change_runs[task.direction] += 1

            suggested = [Direction.parse(value) for value in layout.directions]
            discovered = [direction for direction in suggested if direction in frontier]
            for direction in discovered:
                if direction in {Direction.STAY, task.direction}:
                    continue
                self._enqueue_shift(
                    queue,
                    task.viewport,
                    direction,
                    successful_viewports,
                    table_range,
                    workbook_path,
                    sheet_name,
                )

        if structure_text.strip():
            structure_path.write_text(structure_text, encoding="utf-8")
        changelog_path.write_text(
            "\n\n".join(cumulative_changes).strip() + "\n" if cumulative_changes else "No change.\n",
            encoding="utf-8",
        )
        return LayoutWorkflowResult(
            structure_text=structure_text,
            verification=last_verification,
            iterations=iteration,
            image_path=first_image,
            changelog_path=changelog_path,
            events_path=events_path,
            responses=responses,
        )

    def _enqueue_shift(
        self,
        queue: DirectionQueue,
        viewport: Viewport,
        direction: Direction,
        successful_viewports: set[tuple[int, int]],
        table_range: str | None,
        workbook_path: Path,
        sheet_name: str,
    ) -> None:
        target = viewport.shifted(direction, self.settings.shift_cells)
        if target.key in successful_viewports or not _intersects(target, table_range):
            return
        target_range = target.clipped_a1_range(table_range)
        if not _has_enough_data(workbook_path, sheet_name, target_range):
            return
        queue.push(TraversalTask(direction, target))

    @staticmethod
    def _append_event(path: Path, event: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _intersects(viewport: Viewport, table_range: str | None) -> bool:
    if not table_range:
        return viewport.row == 1 and viewport.column == 1
    min_col, min_row, max_col, max_row = range_boundaries(table_range)
    viewport_max_row = viewport.row + viewport.rows - 1
    viewport_max_col = viewport.column + viewport.columns - 1
    return not (
        viewport_max_row < min_row
        or viewport.row > max_row
        or viewport_max_col < min_col
        or viewport.column > max_col
    )


def _has_enough_data(
    workbook_path: Path,
    sheet_name: str,
    cell_range: str,
    *,
    min_coverage: float = 0.4,
) -> bool:
    """Return whether a candidate viewport has enough real values to process."""
    min_col, min_row, max_col, max_row = range_boundaries(cell_range)
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=False)
    try:
        worksheet = workbook[sheet_name]
        rows_with_values: set[int] = set()
        cols_with_values: set[int] = set()
        for row in worksheet.iter_rows(
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
        ):
            for cell in row:
                value = cell.value
                if value is not None and str(value).strip():
                    rows_with_values.add(cell.row)
                    cols_with_values.add(cell.column)
        row_count = max(1, max_row - min_row + 1)
        col_count = max(1, max_col - min_col + 1)
        return (
            len(rows_with_values) / row_count >= min_coverage
            or len(cols_with_values) / col_count >= min_coverage
        )
    finally:
        workbook.close()
