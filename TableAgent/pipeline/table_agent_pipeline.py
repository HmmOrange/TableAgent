from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import openpyxl
import yaml
from datasets.base import EvalSample
from pipelines.base import BasePipeline, PipelineOutput
from TableAgent.prompts.answer import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT_TEMPLATE
from TableAgent.prompts.reranker import RERANKER_SYSTEM_PROMPT, RERANKER_USER_PROMPT_TEMPLATE
from table2img.core import RenderResult, render_document
from utils.workbook_converter import sample_to_xlsx
from utils.llm.base import BaseLLM, LLMResponse
from utils.log.logger import Logger

from TableAgent.configs import TableAgentConfig
from TableAgent.QA.agents.answer_agent import QAAgent
from TableAgent.QA.runner import TableQARunner
from TableAgent.perception.metadata import SheetMetadata
from TableAgent.structure.layout.agent import LayoutAgent
from TableAgent.structure.layout.parsing import _is_valid_structure
from TableAgent.pipeline.common import (
    SourceCandidate,
    display_path,
    is_siflex,
    read_image_tiles,
    safe_name,
    token_usage,
)
from TableAgent.pipeline.prompting import PromptBuilder
from TableAgent.pipeline.retrieval import SourceRetriever
from TableAgent.pipeline.siflex_formatter import SiflexAnswerFormatterAgent
from TableAgent.pipeline.source_preparer import SourcePreparer
from TableAgent.structure.layout.workflow import TableLayoutWorkflow
from TableAgent.pipeline.structure_cache import StructureCache, StructureCacheRecord
from TableAgent.rendering.workbook import WorkbookRenderer
from TableAgent.structure.verification import DeterministicVerifier

if TYPE_CHECKING:
    from TableAgent.pipeline.retrieval import TableRetrieverContract

logger = Logger(__name__)


class TableAgentPipeline(BasePipeline):
    name = "table_agent"
    prepare_samples_before_run = True
    answer_system_prompt = ANSWER_SYSTEM_PROMPT
    answer_user_prompt_template = ANSWER_USER_PROMPT_TEMPLATE
    reranker_system_prompt = RERANKER_SYSTEM_PROMPT
    reranker_user_prompt_template = RERANKER_USER_PROMPT_TEMPLATE

    def __init__(
        self,
        llm_client: BaseLLM | None,
        layout_vlm_client: BaseLLM | None,
        config: dict[str, Any] | None = None,
        renderer: Callable[..., RenderResult] = render_document,
        table_retriever: TableRetrieverContract | None = None,
    ):
        self.llm = llm_client
        self.layout_vlm = layout_vlm_client
        self.settings = TableAgentConfig.from_config(config)
        self._artifact_dir = self.settings.artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.phase in {"qa", "all"} and self.llm is None:
            raise ValueError(f"TableAgent phase '{self.settings.phase}' requires an answer LLM client")
        if self.settings.phase in {"structure", "all"} and self.layout_vlm is None:
            raise ValueError(f"TableAgent phase '{self.settings.phase}' requires a layout VLM client")
        self.prompts = PromptBuilder(self.settings, self)
        self.workbook_renderer = WorkbookRenderer(self.settings, renderer, logger)
        self.layout_agent = LayoutAgent(self.layout_vlm) if self.layout_vlm is not None else None
        self.verifier = DeterministicVerifier()
        self.qa_agent = QAAgent(self.llm, self.answer_system_prompt) if self.llm is not None else None
        self.table_retriever = table_retriever
        self.layout_workflow = (
            TableLayoutWorkflow(
                self.settings,
                self.workbook_renderer,
                self.layout_agent,
                self.verifier,
                progress_callback=self._progress,
            )
            if self.layout_agent is not None
            else None
        )
        self.source_preparer = SourcePreparer(
            self.settings,
            self._analyze_source_sheet,
            progress_callback=self._progress,
        )
        self.source_retriever = SourceRetriever(self.settings, self.llm, self, self.prompts)
        self.siflex_formatter = SiflexAnswerFormatterAgent(self.llm) if self.llm is not None else None
        self.structure_cache = StructureCache(
            self.settings,
            self.layout_workflow,
            self._metadata_for_workbook_sheet,
        )
        self._verified_samples: dict[str, StructureCacheRecord] = {}
        self._progress_callback: Callable[[str], None] | None = None
        self._apply_generation_cap()

    def prepare_samples(self, samples: list[EvalSample], logger: Any | None = None) -> None:
        if self.settings.phase == "qa":
            missing = []
            for sample in samples:
                if self.settings.run_retrieval and is_siflex(sample) and self.source_retriever.load_candidates(sample):
                    continue
                record = self.structure_cache.load(sample)
                if record is None or not record.valid:
                    missing.append(sample.sample_id)
            if missing:
                raise RuntimeError(
                    "Missing or stale TableAgent structure caches for: "
                    + ", ".join(missing[:20])
                    + ". Run with --table-agent-phase structure or all first."
                )
            return
        records = self.verify_samples(samples, force=self.settings.phase == "all")
        failed = [record for record in records if not record.valid]
        if failed:
            raise RuntimeError(f"TableAgent verification failed for {len(failed)} cache entries")

    def verify_samples(self, samples: list[EvalSample], *, force: bool = True) -> list[StructureCacheRecord]:
        siflex_samples = [sample for sample in samples if is_siflex(sample) and self.settings.run_retrieval]
        standard_samples = [
            sample for sample in samples if not is_siflex(sample) or not self.settings.run_retrieval
        ]
        records = []
        for sample in standard_samples:
            record = self.structure_cache.prepare(sample, force=force)
            records.append(record)
            self._progress(
                "structure_done",
                sample=sample.sample_id,
                workbook=record.workbook_path.name,
                sheet=record.sheet_name,
            )
        self._verified_samples.update({sample.sample_id: record for sample, record in zip(standard_samples, records)})
        if siflex_samples:
            self.source_preparer.prepare(siflex_samples, regenerate_invalid=force)
            seen: set[Path] = set()
            for sample in siflex_samples:
                candidates = self.source_retriever.load_candidates(sample)
                if not candidates:
                    failure_dir = self.settings.source_artifact_dir or self.settings.structure_cache_dir
                    key = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()[:24]
                    records.append(StructureCacheRecord(
                        key=key,
                        directory=failure_dir,
                        workbook_path=Path(str(sample.table_path).split(";")[0]),
                        sheet_name="",
                        structure_path=failure_dir / "structure.yaml",
                        manifest_path=failure_dir / "metadata.json",
                        status="not_good",
                        cache_hit=False,
                    ))
                    continue
                for candidate in candidates:
                    if candidate.directory in seen:
                        continue
                    seen.add(candidate.directory)
                    key = hashlib.sha256(str(candidate.directory.resolve()).encode("utf-8")).hexdigest()[:24]
                    records.append(StructureCacheRecord(
                        key=key,
                        directory=candidate.directory,
                        workbook_path=candidate.workbook_path,
                        sheet_name=candidate.sheet_name,
                        structure_path=candidate.directory / "structure.yaml",
                        manifest_path=candidate.directory / "metadata.json",
                        status="good",
                        cache_hit=not force,
                    ))
        return records

    @staticmethod
    def structure_progress_totals(samples: list[EvalSample]) -> dict[str, Any]:
        """Count the structure work units shown by the CLI progress bar."""
        standard_samples = [sample for sample in samples if not is_siflex(sample)]
        siflex_samples = [sample for sample in samples if is_siflex(sample)]
        sheets_per_file = {f"sample:{sample.sample_id}": 1 for sample in standard_samples}
        files_per_key = {key: 1 for key in sheets_per_file}

        for source_path in SourcePreparer._source_paths(siflex_samples):
            try:
                workbook = openpyxl.load_workbook(source_path, read_only=True, data_only=True)
                try:
                    sheet_count = max(len(workbook.sheetnames), 1)
                finally:
                    workbook.close()
            except Exception:
                sheet_count = 1
            key = f"book:{source_path.name}"
            sheets_per_file[key] = sheets_per_file.get(key, 0) + sheet_count
            files_per_key[key] = files_per_key.get(key, 0) + 1

        return {
            "files": sum(files_per_key.values()),
            "sheets": sum(sheets_per_file.values()),
            "sheets_per_file": sheets_per_file,
            "files_per_key": files_per_key,
        }

    def set_progress_callback(self, callback: Callable[[str], None] | None) -> None:
        self._progress_callback = callback

    def _progress(self, stage: str, **fields: Any) -> None:
        if self._progress_callback is None:
            return
        labels = {
            "prepare": "prepare",
            "prepare_extract": "prepare:extract",
            "prepare_metadata": "prepare:metadata",
            "prepare_cached": "prepare:cached",
            "prepare_error": "prepare:error",
            "prepare_layout": "prepare:layout",
            "prepare_done": "prepare:done",
            "retrieval": "retrieve",
            "rerank": "rerank",
            "render": "render",
            "layout": "layout",
            "verify": "verify",
            "structure_done": "structure:done",
            "qa": "qa",
            "answer": "answer",
            "done": "done",
        }
        parts = [labels.get(stage, stage)]
        if stage in {"prepare_layout", "prepare_done", "render", "layout", "verify"}:
            ordered_fields = [
                ("range", "range"),
                ("iteration", "iter"),
                ("direction", "dir"),
                ("workbook", "book"),
                ("sheet", "sheet"),
                ("sample", "sample"),
            ]
        else:
            ordered_fields = [
                ("sample", "sample"),
                ("workbook", "book"),
                ("sheet", "sheet"),
                ("table", "table"),
                ("range", "range"),
                ("iteration", "iter"),
                ("direction", "dir"),
            ]
        for key, label in ordered_fields:
            value = fields.get(key)
            if value is None or value == "":
                continue
            text = str(value)
            parts.append(f"{label}={text}")
        self._progress_callback(" | ".join(parts))

    def set_run_id(self, run_id: int) -> Path:
        if run_id < 1:
            raise ValueError("run_id must be at least 1")
        if self.settings.run_artifact_dir is not None:
            repeat_dir = self.settings.repeat_dir_template.format(run_id=run_id)
            self._artifact_dir = self.settings.run_artifact_dir / repeat_dir
        else:
            self._artifact_dir = self.settings.artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        return self._artifact_dir

    def run(self, sample: EvalSample) -> PipelineOutput:
        if self.settings.phase == "structure":
            raise RuntimeError("structure phase does not run question answering")
        if is_siflex(sample) and self.settings.run_retrieval:
            if self.settings.phase == "all" and not self.source_retriever.load_candidates(sample):
                self.source_preparer.prepare([sample], regenerate_invalid=True)
            responses: list[LLMResponse] = []
            candidate = self.source_retriever.select(sample, responses, self._fit_context)
            if candidate is None:
                raise RuntimeError(
                    f"Missing or stale structure cache for sample {sample.sample_id!r}; "
                    "run structure or all first"
                )
            return self._run_prepared_source(sample, candidate, responses, self.start_timer())
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
        candidate = self.source_retriever.select(sample, responses, self._fit_context)
        if candidate is not None:
            return self._run_prepared_source(sample, candidate, responses, start_time)

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
        verification = workflow_result.verification

        structure_path = sample_dir / "structure.yaml"
        if _is_valid_structure(structure_text):
            structure_path.write_text(structure_text, encoding="utf-8")
        else:
            structure_path.unlink(missing_ok=True)
        image_path = sample_dir / "table.png"
        html_path = sample_dir / "table.html"
        table_context = self._fit_context(sample.table_content)
        self._progress("qa", sample=sample.sample_id, workbook=workbook.path.name, sheet=sheet_name)
        answer_response, qa_info = self._run_verified_qa(
            question=sample.question,
            structure_path=structure_path,
            workbook_path=workbook.path,
            qa_artifact_dir=sample_dir / "qa",
            fallback_prompt=self.prompts.answer_prompt(sample, table_context, structure_text),
        )
        responses.append(answer_response)
        predicted_answer = self._format_siflex_answer(sample, answer_response.content, responses)
        self._progress("done", sample=sample.sample_id, workbook=workbook.path.name, sheet=sheet_name)

        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=structure_text,
            predicted_answer=predicted_answer,
            latency=self.stop_timer(start_time),
            token_usage=token_usage(responses),
            metadata={
                "structure_path": display_path(structure_path),
                "workbook_path": str(workbook.path),
                "image_path": display_path(image_path if image_path.is_file() else workflow_result.image_path)
                if workflow_result.image_path
                else None,
                "html_path": display_path(html_path) if html_path.is_file() else None,
                "workbook_source_format": workbook.source_format,
                "workbook_sheets": workbook.sheet_names,
                "verification": verification,
                "artifact_dir": display_path(sample_dir),
                "image_tiles": read_image_tiles(sample_dir),
                "metadata_yaml_path": display_path(sample_dir / "metadata.yaml"),
                "render_metadata_path": display_path(sample_dir / "table.metadata.json")
                if (sample_dir / "table.metadata.json").is_file()
                else None,
                "changelog_path": display_path(workflow_result.changelog_path),
                "events_path": display_path(workflow_result.events_path),
                "iteration_artifact_dir": display_path(sample_dir / "iterations"),
                "qa": qa_info,
            },
        )

    def _run_cached_qa(self, sample: EvalSample, record: StructureCacheRecord) -> PipelineOutput:
        start_time = self.start_timer()
        structure_text = record.structure_path.read_text(encoding="utf-8")
        answer_response, qa_info = self._run_verified_qa(
            question=sample.question,
            structure_path=record.structure_path,
            workbook_path=record.workbook_path,
            qa_artifact_dir=self._sample_dir(sample) / "qa",
            fallback_prompt=self.prompts.answer_prompt(sample, self._fit_context(sample.table_content), structure_text),
        )
        responses = [answer_response]
        predicted_answer = self._format_siflex_answer(sample, answer_response.content, responses)
        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=structure_text,
            predicted_answer=predicted_answer,
            latency=self.stop_timer(start_time),
            token_usage=token_usage(responses),
            metadata={
                "structure_path": display_path(record.structure_path),
                "workbook_path": str(record.workbook_path),
                "workbook_source_format": "verification-cache",
                "workbook_sheets": [record.sheet_name],
                "artifact_dir": display_path(record.directory),
                "image_path": display_path(record.directory / "table.png"),
                "html_path": display_path(record.directory / "table.html") if (record.directory / "table.html").is_file() else None,
                "metadata_yaml_path": display_path(record.directory / "metadata.yaml"),
                "changelog_path": display_path(record.directory / "changelog.md"),
                "events_path": display_path(record.directory / "events.jsonl"),
                "iteration_artifact_dir": display_path(record.directory / "iterations"),
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
                **{key: str(value) if isinstance(value, Path) else value for key, value in vars(self.settings).items()},
                "active_artifact_dir": str(self._artifact_dir),
            },
            "prompt": {
                "answer_system_prompt": self.answer_system_prompt,
                "answer_user_prompt_template": self.answer_user_prompt_template,
            },
        }

    def _analyze_source_sheet(
        self,
        source_path: Path,
        sheet_name: str,
        metadata: SheetMetadata,
        sheet_dir: Path,
    ) -> str:
        if self.layout_workflow is None:
            raise RuntimeError("Source verification requires a layout VLM client")
        self._progress("prepare", workbook=source_path.name, sheet=sheet_name, range=metadata.used_range)
        result = self.layout_workflow.run(
            workbook_path=source_path,
            sheet_name=sheet_name,
            metadata=metadata,
            output_dir=sheet_dir,
        )
        return result.structure_text

    def _run_prepared_source(
        self,
        sample: EvalSample,
        candidate: SourceCandidate,
        responses: list[LLMResponse],
        start_time: float,
    ) -> PipelineOutput:
        image_prompt = self.prompts.answer_prompt(sample, "[Table image provided]", candidate.structure_text)
        fallback_prompt = self.prompts.answer_prompt(sample, self._fit_context(candidate.sheet_text), candidate.structure_text)
        structure_path = candidate.directory / "structure.yaml"
        if candidate.table_id:
            structure_path = self._sample_dir(sample) / "retrieved_structure.yaml"
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_text(candidate.structure_text, encoding="utf-8")
        self._progress(
            "qa",
            sample=sample.sample_id,
            workbook=candidate.workbook_path.name,
            sheet=candidate.sheet_name,
            table=candidate.table_id,
        )
        answer_response, qa_info = self._run_verified_qa(
            question=sample.question,
            structure_path=structure_path,
            workbook_path=candidate.workbook_path,
            qa_artifact_dir=self._sample_dir(sample) / "qa",
            fallback_prompt=image_prompt,
            fallback_image_path=candidate.image_path,
            fallback_text_prompt=fallback_prompt,
        )
        responses.append(answer_response)
        predicted_answer = self._format_siflex_answer(sample, answer_response.content, responses)
        self._progress(
            "done",
            sample=sample.sample_id,
            workbook=candidate.workbook_path.name,
            sheet=candidate.sheet_name,
        )
        retrieval_info = {
            "score": candidate.score,
            "lexical_score": getattr(candidate, "lexical_score", candidate.score),
            "embedding_score": getattr(candidate, "embedding_score", 0.0),
            "embedding_used": getattr(candidate, "embedding_used", False),
            "fallback_used": getattr(candidate, "fallback_used", False),
            "table_id": getattr(candidate, "table_id", ""),
            "table_name": getattr(candidate, "table_name", ""),
            "table_description": getattr(candidate, "table_description", ""),
        }
        if hasattr(candidate, "reranker_selected_index"):
            retrieval_info["reranker_selected_index"] = getattr(candidate, "reranker_selected_index")
            retrieval_info["reranker_rationale"] = getattr(candidate, "reranker_rationale", "")

        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=candidate.structure_text,
            predicted_answer=predicted_answer,
            latency=self.stop_timer(start_time),
            token_usage=token_usage(responses),
            metadata={
                "structure_path": display_path(structure_path),
                "thinking_trace_path": display_path(candidate.directory / "thinking_trace.txt")
                if (candidate.directory / "thinking_trace.txt").is_file()
                else None,
                "workbook_path": str(candidate.workbook_path.resolve()),
                "image_path": display_path(candidate.image_path),
                "html_path": display_path(candidate.html_path) if candidate.html_path else None,
                "workbook_source_format": "xlsx",
                "workbook_sheets": [candidate.sheet_name],
                "verification": {"status": "good", "feedback": "Retrieved from encoded source"},
                "artifact_dir": display_path(candidate.directory),
                "image_tiles": read_image_tiles(candidate.directory),
                "retrieval_info": retrieval_info,
                "metadata_yaml_path": display_path(candidate.directory / "metadata.yaml"),
                "render_metadata_path": display_path(candidate.directory / "table.metadata.json")
                if (candidate.directory / "table.metadata.json").is_file()
                else None,
                "changelog_path": display_path(candidate.directory / "changelog.md"),
                "events_path": display_path(candidate.directory / "events.jsonl"),
                "iteration_artifact_dir": display_path(candidate.directory / "iterations"),
                "qa": qa_info,
            },
        )

    def _run_verified_qa(
        self,
        *,
        question: str,
        structure_path: Path,
        workbook_path: Path,
        qa_artifact_dir: Path,
        fallback_prompt: str,
        fallback_image_path: Path | None = None,
        fallback_text_prompt: str | None = None,
    ) -> tuple[LLMResponse, dict[str, Any]]:
        """Run the notebook QA phase against the persisted verified structure."""
        qa_token_usage = {"prompt": 0, "completion": 0}
        with TableQARunner(
                structure_path=str(structure_path),
                workbook_path=str(workbook_path),
                llm_client=self.llm,
                config={
                    "table_agent": {
                        **vars(self.settings),
                        "artifact_dir": str(qa_artifact_dir),
                    },
                    "qa_artifact_dir": str(qa_artifact_dir),
                },
                table_retriever=self.table_retriever,
            ) as runner:
            result = runner.run(question)
            qa_token_usage = result.token_usage
            qa_info = {
                "success": result.success,
                "error": result.error,
                "execution_time": result.execution_time,
                "token_usage": result.token_usage,
                "artifacts": result.artifacts,
                "fallback_used": not result.success,
            }
            if result.success and result.final_answer is not None:
                return LLMResponse(
                    content=result.final_answer,
                    prompt_tokens=int(result.token_usage.get("prompt", 0) or 0),
                    completion_tokens=int(result.token_usage.get("completion", 0) or 0),
                ), qa_info

        if result.success:
            raise RuntimeError("TableQARunner returned success without a final answer")
        response = self.qa_agent.run(
            prompt=fallback_prompt,
            image_path=fallback_image_path,
            fallback_prompt=fallback_text_prompt,
        )
        response.prompt_tokens += int(qa_token_usage.get("prompt", 0) or 0)
        response.completion_tokens += int(qa_token_usage.get("completion", 0) or 0)
        return response, qa_info

    def _format_siflex_answer(
        self,
        sample: EvalSample,
        draft_answer: str,
        responses: list[LLMResponse],
    ) -> str:
        if not is_siflex(sample) or not isinstance(sample.raw, dict):
            return draft_answer
        answer_type = str(sample.raw.get("answer_type", "")).strip().lower()
        if not self.siflex_formatter.supports(answer_type):
            return draft_answer
        result = self.siflex_formatter.run(
            question=sample.question,
            answer_type=answer_type,
            draft_answer=draft_answer,
        )
        responses.append(result.response)
        return result.answer

    @staticmethod
    def _select_table_id(structure_path: Path, question: str) -> str | None:
        """Choose the most question-relevant table in a multi-table structure."""
        try:
            payload = yaml.safe_load(structure_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return None
        if not isinstance(payload, dict):
            return None
        query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        best: tuple[int, str] | None = None
        for key, table in payload.items():
            if not isinstance(table, dict):
                continue
            table_id = str(table.get("id") or key)
            searchable = " ".join(
                [table_id, str(table.get("name") or ""), str(table.get("description") or "")]
                + [
                    " ".join(str(header.get(field) or "") for field in ("id", "label", "description"))
                    for header in table.get("headers") or []
                    if isinstance(header, dict)
                ]
            )
            score = len(query_terms & set(re.findall(r"[a-z0-9]+", searchable.lower())))
            candidate = (score, table_id)
            if best is None or candidate > best:
                best = candidate
        return best[1] if best else None

    def _generate_answer_with_image(self, *, prompt: str, image_path: Path, fallback_prompt: str | None = None) -> LLMResponse:
        return self.qa_agent.run(prompt=prompt, image_path=image_path, fallback_prompt=fallback_prompt)

    def _fit_context(self, table_content: str) -> str:
        if len(table_content) <= self.settings.max_context_chars:
            return table_content
        return table_content[: self.settings.max_context_chars] + "\n...TRUNCATED..."

    def _answer_prompt(self, sample: EvalSample, table_context: str, structure_text: str) -> str:
        return self.prompts.answer_prompt(sample, table_context, structure_text)

    def _sample_dir(self, sample: EvalSample) -> Path:
        raw = f"{sample.sample_id}:{sample.table_id}:{sample.question}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return self._artifact_dir / safe_name(sample.sample_id)[:80] / digest

    @staticmethod
    def _metadata_for_workbook_sheet(workbook_path: Path, sheet_name: str) -> SheetMetadata:
        workbook = openpyxl.load_workbook(workbook_path, read_only=False, data_only=False)
        try:
            worksheet = workbook[sheet_name]
            used_range = worksheet.calculate_dimension()
            merged_ranges = [str(cell_range) for cell_range in worksheet.merged_cells.ranges]
            if used_range == "A1:A1" and worksheet["A1"].value is None:
                used_range = None
            return SheetMetadata(sheet_name, used_range, merged_ranges)
        finally:
            workbook.close()

    def _apply_generation_cap(self) -> None:
        if self.settings.generation_max_tokens is None:
            return
        if hasattr(self.llm, "max_tokens"):
            self.llm.max_tokens = self.settings.generation_max_tokens
        if hasattr(self.layout_vlm, "max_tokens"):
            self.layout_vlm.max_tokens = self.settings.generation_max_tokens

    @staticmethod
    def _client_config(client: Any) -> dict[str, Any]:
        return {
            "model_name": getattr(client, "model_name", None),
            "temperature": getattr(client, "temperature", None),
            "max_tokens": getattr(client, "max_tokens", None),
        }
