from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable

import openpyxl
import yaml
from datasets.base import EvalSample
from pipelines.base import BasePipeline, PipelineOutput
from TableAgent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
    RERANKER_SYSTEM_PROMPT,
    RERANKER_USER_PROMPT_TEMPLATE,
)
from table2img.core import RenderResult, render_document
from utils.workbook_converter import sample_to_xlsx
from utils.llm.base import BaseLLM, LLMResponse
from utils.log.logger import Logger

from TableAgent.config import TableAgentConfig
from TableAgent.agents import LayoutAgent, QAAgent, VerificationAgent
from TableAgent.QA.runner import TableQARunner
from TableAgent.perception.metadata import SheetMetadata
from TableAgent.perception.structure import _is_valid_structure
from TableAgent.pipeline.common import (
    SourceCandidate,
    display_path,
    read_image_tiles,
    safe_name,
    token_usage,
)
from TableAgent.pipeline.prompting import PromptBuilder
from TableAgent.pipeline.retrieval import SourceRetriever
from TableAgent.pipeline.source_preparer import SourcePreparer
from TableAgent.pipeline.layout_workflow import TableLayoutWorkflow
from TableAgent.rendering.workbook import WorkbookRenderer

logger = Logger(__name__)


class TableAgentPipeline(BasePipeline):
    name = "table_agent"
    prepare_samples_before_run = False
    answer_system_prompt = ANSWER_SYSTEM_PROMPT
    answer_user_prompt_template = ANSWER_USER_PROMPT_TEMPLATE
    reranker_system_prompt = RERANKER_SYSTEM_PROMPT
    reranker_user_prompt_template = RERANKER_USER_PROMPT_TEMPLATE

    def __init__(
        self,
        llm_client: BaseLLM,
        layout_vlm_client: BaseLLM,
        config: dict[str, Any] | None = None,
        renderer: Callable[..., RenderResult] = render_document,
    ):
        self.llm = llm_client
        self.layout_vlm = layout_vlm_client
        self.settings = TableAgentConfig.from_config(config)
        self._artifact_dir = self.settings.artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = PromptBuilder(self.settings, self)
        self.workbook_renderer = WorkbookRenderer(self.settings, renderer, logger)
        self.layout_agent = LayoutAgent(self.layout_vlm)
        self.verification_agent = VerificationAgent(self.llm)
        self.qa_agent = QAAgent(self.llm, self.answer_system_prompt)
        self.layout_workflow = TableLayoutWorkflow(
            self.settings,
            self.workbook_renderer,
            self.layout_agent,
            self.verification_agent,
        )
        self.source_preparer = SourcePreparer(self.settings, self._analyze_source_sheet)
        self.source_retriever = SourceRetriever(self.settings, self.llm, self, self.prompts)
        self._apply_generation_cap()

    def prepare_samples(self, samples: list[EvalSample], logger: Any | None = None) -> None:
        self.source_preparer.prepare(samples, logger=logger)

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
        start_time = self.start_timer()
        responses: list[LLMResponse] = []

        self.source_preparer.prepare([sample], regenerate_invalid=False)
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
        answer_response, qa_info = self._run_verified_qa(
            question=sample.question,
            structure_path=structure_path,
            workbook_path=workbook.path,
            qa_artifact_dir=sample_dir / "qa",
            fallback_prompt=self.prompts.answer_prompt(sample, table_context, structure_text),
        )
        responses.append(answer_response)

        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=structure_text,
            predicted_answer=answer_response.content,
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
                "changelog_path": display_path(workflow_result.changelog_path),
                "events_path": display_path(workflow_result.events_path),
                "iteration_artifact_dir": display_path(sample_dir / "iterations"),
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
        answer_response, qa_info = self._run_verified_qa(
            question=sample.question,
            structure_path=candidate.directory / "structure.yaml",
            workbook_path=candidate.workbook_path,
            qa_artifact_dir=candidate.directory / "qa",
            fallback_prompt=image_prompt,
            fallback_image_path=candidate.image_path,
            fallback_text_prompt=fallback_prompt,
        )
        responses.append(answer_response)
        retrieval_info = {
            "score": candidate.score,
            "lexical_score": getattr(candidate, "lexical_score", candidate.score),
            "embedding_score": getattr(candidate, "embedding_score", 0.0),
            "embedding_used": getattr(candidate, "embedding_used", False),
            "fallback_used": getattr(candidate, "fallback_used", False),
        }
        if hasattr(candidate, "reranker_selected_index"):
            retrieval_info["reranker_selected_index"] = getattr(candidate, "reranker_selected_index")
            retrieval_info["reranker_rationale"] = getattr(candidate, "reranker_rationale", "")

        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=candidate.structure_text,
            predicted_answer=answer_response.content,
            latency=self.stop_timer(start_time),
            token_usage=token_usage(responses),
            metadata={
                "structure_path": display_path(candidate.directory / "structure.yaml"),
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
        runner = None
        qa_token_usage = {"prompt": 0, "completion": 0}
        try:
            runner = TableQARunner(
                structure_path=str(structure_path),
                workbook_path=str(workbook_path),
                llm_client=self.llm,
                config={
                    "qa_artifact_dir": str(qa_artifact_dir),
                },
            )
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
        except Exception as exc:
            if runner is not None:
                qa_token_usage = runner.token_usage()
            qa_info = {
                "success": False,
                "error": str(exc),
                "token_usage": qa_token_usage,
                "artifacts": {},
                "fallback_used": True,
            }
        finally:
            if runner is not None:
                runner.env.workbook.close()

        response = self.qa_agent.run(
            prompt=fallback_prompt,
            image_path=fallback_image_path,
            fallback_prompt=fallback_text_prompt,
        )
        response.prompt_tokens += int(qa_token_usage.get("prompt", 0) or 0)
        response.completion_tokens += int(qa_token_usage.get("completion", 0) or 0)
        return response, qa_info

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
