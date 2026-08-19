from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.base import PipelineOutput
from TableAgent.pipeline.component import RuntimeComponent
from TableAgent.pipeline.contracts import PipelineRuntimeContract
from TableAgent.pipeline.sample import has_workbook_sources
from TableAgent.utils.llm_metrics import token_usage
from TableAgent.utils.paths import display_path
from TableAgent.pipeline.sample import EvalSample
from TableAgent.stages.qa import QAInput
from TableAgent.stages.retrieval import RetrievalInput


def serialize_config_value(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: serialize_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class PipelineRunner(RuntimeComponent):
    """Execute prepared or cached QA through explicit stage contracts."""

    def __init__(self, runtime: PipelineRuntimeContract):
        super().__init__(runtime)

    def _force_structure(self) -> bool:
        if bool(getattr(self.settings, "reuse_structure", False)):
            return False
        # Structure already prepared once for this pipeline instance (e.g. earlier repeat).
        if bool(getattr(self.runtime, "_structure_prepared", False)):
            return False
        if bool(getattr(self.settings, "force_structure", False)):
            return True
        # Default: regenerate structure once for structure/all runs.
        return self.settings.phase in {"all", "structure"}

    def run(self, sample: EvalSample) -> PipelineOutput:
        if self.settings.phase == "structure":
            raise RuntimeError("structure phase does not run question answering")
        force_structure = self._force_structure()
        if has_workbook_sources(sample) and self.settings.should_retrieve(sample):
            structure_runtime = 0.0
            if self.settings.phase == "all" and not self.settings.perfect_retrieval:
                if sample.sample_id in self._prepared_source_samples:
                    self._prepared_source_samples.discard(sample.sample_id)
                else:
                    structure_started = self.start_timer()
                    self.source_preparer.prepare(
                        [sample],
                        regenerate_invalid=True,
                        force=force_structure,
                    )
                    structure_runtime = self.stop_timer(structure_started)
            responses: list[LLMResponse] = []
            retrieval_started = self.start_timer()
            candidate = self.stages.retrieval.run(
                RetrievalInput(
                    sample=sample,
                    responses=responses,
                    fit_context=self._fit_context,
                    perfect=self.settings.perfect_retrieval,
                )
            ).candidate
            retrieval_runtime = self.stop_timer(retrieval_started)
            if candidate is None:
                raise RuntimeError(
                    f"Missing or stale structure cache for sample {sample.sample_id!r}; "
                    "run structure or all first"
                )
            qa_started = self.start_timer()
            run_prepared = self._run_prepared_source
            try:
                output = run_prepared(
                    sample,
                    candidate,
                    responses,
                    qa_started,
                    structure_runtime=structure_runtime,
                    retrieval_runtime=retrieval_runtime,
                )
            except TypeError:
                # Older/mocked callables may not accept stage-runtime kwargs.
                output = run_prepared(sample, candidate, responses, qa_started)
            # Backfill stage timings if the source-QA path ignored kwargs (tests/mocks).
            if getattr(output, "metadata", None) is not None:
                metadata = dict(output.metadata)
                structure_seconds = float(
                    metadata.get("structure_runtime") or structure_runtime or 0.0
                )
                retrieval_seconds = float(
                    metadata.get("retrieval_runtime")
                    or (metadata.get("stage_runtimes") or {}).get("retrieval")
                    or retrieval_runtime
                    or 0.0
                )
                qa_seconds = float(
                    metadata.get("qa_runtime")
                    or (metadata.get("stage_runtimes") or {}).get("qa")
                    or metadata.get("qa", {}).get("execution_time")
                    or 0.0
                )
                if qa_seconds <= 0:
                    # Prefer pure QA time by subtracting known non-QA stages from total latency.
                    total_latency = float(output.latency or 0.0)
                    qa_seconds = max(total_latency - structure_seconds - retrieval_seconds, 0.0)
                stage_runtimes = {
                    "structure": structure_seconds,
                    "retrieval": retrieval_seconds,
                    "qa": qa_seconds,
                    "total": structure_seconds + retrieval_seconds + qa_seconds,
                }
                metadata.update(
                    {
                        # structure_runtime covers understanding + structure extraction.
                        "structure_runtime": structure_seconds,
                        "understanding_runtime": structure_seconds,
                        "retrieval_runtime": retrieval_seconds,
                        "qa_runtime": qa_seconds,
                        "stage_runtimes": stage_runtimes,
                    }
                )
                output.metadata = metadata
                output.latency = stage_runtimes["total"]
            return output
        structure_runtime = 0.0
        if self.settings.phase == "all":
            record = self._verified_samples.get(sample.sample_id)
            if record is None:
                structure_started = self.start_timer()
                record = self.structure_cache.prepare(sample, force=force_structure)
                structure_runtime = self.stop_timer(structure_started)
                self._verified_samples[sample.sample_id] = record
            elif getattr(record, "cache_hit", False) is False:
                # Record was freshly built earlier in this process.
                structure_runtime = float(
                    getattr(record, "structure_runtime", 0.0) or 0.0
                )
        else:
            record = self.structure_cache.load(sample)
        if record is None or not record.valid:
            raise RuntimeError(
                f"Missing or stale structure cache for sample {sample.sample_id!r}; "
                "run structure or all first"
            )
        return self._run_cached_qa(
            sample,
            record,
            structure_runtime=structure_runtime,
        )

    def _run_cached_qa(
        self,
        sample,
        record,
        *,
        structure_runtime: float = 0.0,
    ) -> PipelineOutput:
        qa_started = self.start_timer()
        materialize = getattr(
            self.runtime,
            "_materialize_structure_artifact",
            None,
        )
        if callable(materialize):
            record = materialize(record)
        structure_text = record.structure_path.read_text(encoding="utf-8")
        qa_output = self.stages.qa.run(QAInput(
            question=sample.question,
            structure_path=record.structure_path,
            workbook_path=record.workbook_path,
            qa_artifact_dir=self._qa_sample_dir(sample),
            fallback_prompt=self.prompts.answer_prompt(
                sample, self._fit_context(sample.table_content), structure_text
            ),
        ))
        answer_response, qa_info = qa_output.response, qa_output.metadata
        qa_runtime = self.stop_timer(qa_started)
        if isinstance(qa_info, dict):
            qa_runtime = float(qa_info.get("execution_time") or qa_runtime or 0.0)
        stage_runtimes = {
            "structure": float(structure_runtime or 0.0),
            "retrieval": 0.0,
            "qa": float(qa_runtime or 0.0),
            "total": float(structure_runtime or 0.0) + float(qa_runtime or 0.0),
        }
        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=structure_text,
            predicted_answer=answer_response.content,
            latency=stage_runtimes["total"],
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
                "structure_runtime": stage_runtimes["structure"],
                "understanding_runtime": stage_runtimes["structure"],
                "retrieval_runtime": stage_runtimes["retrieval"],
                "qa_runtime": stage_runtimes["qa"],
                "stage_runtimes": stage_runtimes,
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

    @staticmethod
    def _client_config(client: Any) -> dict[str, Any]:
        return {
            "model_name": getattr(client, "model_name", None),
            "temperature": getattr(client, "temperature", None),
            "max_tokens": getattr(client, "max_tokens", None),
        }

    @classmethod
    def _serialize_config_value(cls, value: Any) -> Any:
        return serialize_config_value(value)
