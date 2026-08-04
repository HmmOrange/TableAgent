from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

from TableAgent.schema.qa import QAResult
from TableAgent.schema.subtask import SubTask


class QAArtifactMixin:
    """Persist QA plans, generated code, notebook cells, and result metadata."""

    @staticmethod
    def _make_run_id(question: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", question.strip()).strip("_").lower()
        return f"{timestamp}_{(slug or 'qa_run')[:60]}"

    def _persist_run_artifacts(
        self, result: QAResult, run_dir: Path, event_start_index: int
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        cells_dir = run_dir / "cells"
        cells_dir.mkdir(parents=True, exist_ok=True)

        events = self.env.logger.events[event_start_index:]
        artifacts: dict[str, str] = {
            "run_dir": str(run_dir),
            "events_jsonl": str(run_dir / "events.jsonl"),
            "plan_json": str(run_dir / "plan.json"),
            "result_json": str(run_dir / "result.json"),
            "cells_dir": str(cells_dir),
        }

        self._write_events_jsonl(run_dir / "events.jsonl", events)
        self._write_json(
            run_dir / "plan.json",
            [self._subtask_to_dict(subtask) for subtask in result.plan],
        )

        for index, cell in enumerate(self.env.notebook.cells, start=1):
            cell_path = cells_dir / f"{index:02d}_{cell.cell_id}.py"
            cell_path.write_text(cell.code.rstrip() + "\n", encoding="utf-8")

        if self.env.notebook.cells:
            artifacts["cells_index"] = str(cells_dir)

        answer_output = self._find_answer_output(result)
        if answer_output is not None:
            answer_path = run_dir / "answer.py"
            answer_path.write_text(answer_output.code.rstrip() + "\n", encoding="utf-8")
            artifacts["answer_py"] = str(answer_path)

        generated_dir = run_dir / "generated_code"
        generated_dir.mkdir(parents=True, exist_ok=True)
        for index, output in enumerate(result.subtask_outputs, start=1):
            safe_id = (
                re.sub(r"[^a-zA-Z0-9_.-]+", "_", output.subtask_id).strip("_")
                or f"subtask_{index}"
            )
            code_path = generated_dir / f"{index:02d}_{safe_id}.py"
            code_path.write_text(output.code.rstrip() + "\n", encoding="utf-8")
        artifacts["generated_code_dir"] = str(generated_dir)

        try:
            notebook_path = self.env.export_notebook(run_dir / "notebook.ipynb")
            artifacts["notebook_ipynb"] = str(notebook_path)
        except Exception as exc:
            artifacts["notebook_export_error"] = str(exc)

        result.artifacts = artifacts
        self._write_json(
            run_dir / "result.json",
            {
                "question": result.question,
                "success": result.success,
                "final_answer": result.final_answer,
                "error": result.error,
                "execution_time": result.execution_time,
                "replan_count": result.replan_count,
                "token_usage": result.token_usage,
                "artifacts": artifacts,
                "plan": [
                    self._subtask_to_dict(subtask) for subtask in result.plan
                ],
                "subtask_outputs": [
                    {
                        "subtask_id": output.subtask_id,
                        "description": output.description,
                        "success": output.success,
                        "observation": output.observation,
                        "reasoning": output.reasoning,
                        "code": output.code,
                        "layer": output.layer,
                        "category": output.category,
                        "namespace_updates": list(output.namespace_updates.keys()),
                    }
                    for output in result.subtask_outputs
                ],
            },
        )
        self.env.logger.log_event("run_artifact_complete", artifacts)

    @staticmethod
    def _find_answer_output(result: QAResult):
        synthesis_ids = {
            subtask.id for subtask in result.plan if subtask.layer == "synthesis"
        }
        for output in reversed(result.subtask_outputs):
            if output.subtask_id in synthesis_ids and output.code.strip():
                return output
        for output in reversed(result.subtask_outputs):
            if "final_answer" in output.code and output.code.strip():
                return output
        return None

    @staticmethod
    def _subtask_to_dict(subtask: SubTask) -> dict[str, Any]:
        return {
            "id": subtask.id,
            "description": subtask.description,
            "layer": subtask.layer,
            "category": subtask.category,
            "depends_on": list(subtask.depends_on),
            "status": subtask.status,
            "metadata": subtask.metadata,
            "assigned_agent": subtask.assigned_agent,
            "observation": subtask.observation,
        }

    @staticmethod
    def _write_events_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
