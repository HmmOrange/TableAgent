from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.base import PipelineOutput
from TableAgent.shared.pipeline import (
    display_path,
    has_workbook_sources,
    token_usage,
)
from TableAgent.schema import EvalSample
from TableAgent.stages.qa import QAInput
from TableAgent.stages.retrieval import RetrievalInput


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
            candidate = self.retrieval_stage.run(
                RetrievalInput(
                    sample=sample,
                    responses=responses,
                    fit_context=self._fit_context,
                    perfect=self.settings.perfect_retrieval,
                )
            ).candidate
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

    def _run_cached_qa(self, sample, record) -> PipelineOutput:
        start_time = self.start_timer()
        structure_text = record.structure_path.read_text(encoding="utf-8")
        qa_output = self.qa_stage.run(QAInput(
            question=sample.question,
            structure_path=record.structure_path,
            workbook_path=record.workbook_path,
            qa_artifact_dir=self._qa_sample_dir(sample),
            fallback_prompt=self.prompts.answer_prompt(
                sample, self._fit_context(sample.table_content), structure_text
            ),
        ))
        answer_response, qa_info = qa_output.response, qa_output.metadata
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
