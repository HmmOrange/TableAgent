from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.base import PipelineOutput
from TableAgent.pipeline.common import (
    display_path,
    has_workbook_sources,
    read_image_tiles,
    token_usage,
)
from TableAgent.rendering.converter import sample_to_xlsx
from TableAgent.schema import EvalSample
from TableAgent.structure.layout.parsing import _is_valid_structure


class PipelineRunMixin:
    """Execute prepared or cached TableAgent QA runs."""

    def run(self, sample: EvalSample) -> PipelineOutput:
        if self.settings.phase == "structure":
            raise RuntimeError("structure phase does not run question answering")
        if has_workbook_sources(sample) and self.settings.should_retrieve(sample):
            if self.settings.phase == "all" and not self.settings.perfect_retrieval:
                if sample.sample_id in self._prepared_source_samples:
                    self._prepared_source_samples.discard(sample.sample_id)
                else:
                    self.source_preparer.prepare(
                        [sample], regenerate_invalid=True, force=True
                    )
            responses: list[LLMResponse] = []
            candidate = (
                self.source_retriever.select_perfect(sample)
                if self.settings.perfect_retrieval
                else self.source_retriever.select(
                    sample, responses, self._fit_context
                )
            )
            if candidate is None:
                raise RuntimeError(
                    f"Missing or stale structure cache for sample {sample.sample_id!r}; "
                    "run structure or all first"
                )
            return self._run_prepared_source(
                sample, candidate, responses, self.start_timer()
            )
        if self.settings.phase == "all":
            record = self._verified_samples.get(sample.sample_id)
            if record is None:
                record = self.structure_cache.prepare(sample, force=True)
                self._verified_samples[sample.sample_id] = record
        else:
            record = self.structure_cache.load(sample)
        if record is None or not record.valid:
            raise RuntimeError(
                f"Missing or stale structure cache for sample {sample.sample_id!r}; "
                "run structure or all first"
            )
        return self._run_cached_qa(sample, record)

    def _run_legacy(self, sample: EvalSample) -> PipelineOutput:
        start_time = self.start_timer()
        responses: list[LLMResponse] = []

        self._progress("prepare", sample=sample.sample_id)
        self.source_preparer.prepare([sample], regenerate_invalid=False)
        self._progress("retrieval", sample=sample.sample_id)
        candidate = self.source_retriever.select(
            sample, responses, self._fit_context
        )
        if candidate is not None:
            return self._run_prepared_source(
                sample, candidate, responses, start_time
            )

        sample_dir = self._sample_dir(sample)
        sample_dir.mkdir(parents=True, exist_ok=True)
        workbook = sample_to_xlsx(sample, sample_dir / "table.xlsx")
        sheet_name = workbook.sheet_names[0]
        metadata = self._metadata_for_workbook_sheet(workbook.path, sheet_name)
        workflow_result = self.layout_workflow.run(
            workbook_path=workbook.path,
            sheet_name=sheet_name,
            metadata=metadata,
            output_dir=sample_dir,
        )
        responses.extend(workflow_result.responses)
        structure_text = workflow_result.structure_text

        structure_path = sample_dir / "structure.yaml"
        if _is_valid_structure(structure_text):
            structure_path.write_text(structure_text, encoding="utf-8")
        else:
            structure_path.unlink(missing_ok=True)
        image_path = sample_dir / "table.png"
        html_path = sample_dir / "table.html"
        table_context = self._fit_context(sample.table_content)
        self._progress(
            "qa",
            sample=sample.sample_id,
            workbook=workbook.path.name,
            sheet=sheet_name,
        )
        answer_response, qa_info = self._run_verified_qa(
            question=sample.question,
            structure_path=structure_path,
            workbook_path=workbook.path,
            qa_artifact_dir=self._qa_sample_dir(sample),
            fallback_prompt=self.prompts.answer_prompt(
                sample, table_context, structure_text
            ),
        )
        responses.append(answer_response)
        self._progress(
            "done",
            sample=sample.sample_id,
            workbook=workbook.path.name,
            sheet=sheet_name,
        )

        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=structure_text,
            predicted_answer=answer_response.content,
            latency=self.stop_timer(start_time),
            token_usage=token_usage(responses),
            metadata={
                "structure_path": display_path(structure_path),
                "workbook_path": str(workbook.path),
                "image_path": display_path(
                    image_path
                    if image_path.is_file()
                    else workflow_result.image_path
                )
                if workflow_result.image_path
                else None,
                "html_path": display_path(html_path) if html_path.is_file() else None,
                "workbook_source_format": workbook.source_format,
                "workbook_sheets": workbook.sheet_names,
                "verification": workflow_result.verification,
                "artifact_dir": display_path(sample_dir),
                "image_tiles": read_image_tiles(sample_dir),
                "metadata_yaml_path": display_path(sample_dir / "metadata.yaml"),
                "render_metadata_path": display_path(
                    sample_dir / "table.metadata.json"
                )
                if (sample_dir / "table.metadata.json").is_file()
                else None,
                "changelog_path": display_path(workflow_result.changelog_path),
                "events_path": display_path(workflow_result.events_path),
                "iteration_artifact_dir": display_path(sample_dir / "iterations"),
                "qa": qa_info,
            },
        )

    def _run_cached_qa(self, sample, record) -> PipelineOutput:
        start_time = self.start_timer()
        structure_text = record.structure_path.read_text(encoding="utf-8")
        answer_response, qa_info = self._run_verified_qa(
            question=sample.question,
            structure_path=record.structure_path,
            workbook_path=record.workbook_path,
            qa_artifact_dir=self._qa_sample_dir(sample),
            fallback_prompt=self.prompts.answer_prompt(
                sample, self._fit_context(sample.table_content), structure_text
            ),
        )
        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=structure_text,
            predicted_answer=answer_response.content,
            latency=self.stop_timer(start_time),
            token_usage=token_usage([answer_response]),
            metadata={
                "structure_path": display_path(record.structure_path),
                "workbook_path": str(record.workbook_path),
                "workbook_source_format": "verification-cache",
                "workbook_sheets": [record.sheet_name],
                "artifact_dir": display_path(record.directory),
                "image_path": display_path(record.directory / "table.png"),
                "html_path": display_path(record.directory / "table.html")
                if (record.directory / "table.html").is_file()
                else None,
                "metadata_yaml_path": display_path(
                    record.directory / "metadata.yaml"
                ),
                "changelog_path": display_path(record.directory / "changelog.md"),
                "events_path": display_path(record.directory / "events.jsonl"),
                "iteration_artifact_dir": display_path(
                    record.directory / "iterations"
                ),
                "cache_key": record.key,
                "cache_dir": display_path(record.directory),
                "cache_hit": record.cache_hit,
                "verification": {"status": record.status},
                "qa": qa_info,
            },
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "pipeline_type": self.name,
            "llm": self._client_config(self.llm),
            "layout_vlm": self._client_config(self.layout_vlm),
            "agent": {
                **self._serialize_config_value(self.settings),
                "active_artifact_dir": str(self._artifact_dir),
            },
            "prompt": {
                "answer_system_prompt": self.answer_system_prompt,
                "answer_user_prompt_template": self.answer_user_prompt_template,
            },
        }

    @classmethod
    def _serialize_config_value(cls, value: Any) -> Any:
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            return {
                key: cls._serialize_config_value(item) for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._serialize_config_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value
