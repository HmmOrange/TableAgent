from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Any

import openpyxl
import yaml

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.base import PipelineOutput
from TableAgent.pipeline.common import (
    SourceCandidate,
    display_path,
    read_image_tiles,
    safe_name,
    token_usage,
)
from TableAgent.schema import EvalSample


class PipelineSourceQAMixin:
    """Answer against a retrieved prepared source and report retrieval evidence."""

    def _run_prepared_source(
        self,
        sample: EvalSample,
        candidate: SourceCandidate,
        responses: list[LLMResponse],
        start_time: float,
    ) -> PipelineOutput:
        is_metadata_retrieval = candidate.retrieval_type == "metadata"
        image_prompt = self.prompts.answer_prompt(
            sample, "[Table image provided]", candidate.structure_text
        )
        fallback_prompt = self.prompts.answer_prompt(
            sample,
            self._fit_context(candidate.sheet_text),
            candidate.structure_text,
        )
        structure_path = candidate.directory / "structure.yaml"
        if candidate.table_id or is_metadata_retrieval:
            filename = (
                "retrieved_metadata.yaml"
                if is_metadata_retrieval
                else "retrieved_structure.yaml"
            )
            structure_path = self._sample_dir(sample) / filename
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_text(candidate.structure_text, encoding="utf-8")
        related_structure_paths = self._related_structure_paths(candidate)
        self._progress(
            "qa",
            sample=sample.sample_id,
            workbook=candidate.workbook_path.name,
            sheet=candidate.sheet_name,
            table=candidate.table_id,
        )
        if is_metadata_retrieval:
            answer_response = self.qa_agent.run(
                prompt=fallback_prompt,
                image_path=None,
                fallback_prompt=fallback_prompt,
            )
            qa_info = {
                "success": True,
                "error": None,
                "execution_time": 0,
                "token_usage": {
                    "prompt": int(getattr(answer_response, "prompt_tokens", 0) or 0),
                    "completion": int(
                        getattr(answer_response, "completion_tokens", 0) or 0
                    ),
                },
                "artifacts": {},
                "fallback_used": True,
                "mode": "metadata_context",
                "answer_route": "metadata_context",
            }
        else:
            answer_response, qa_info = self._run_verified_qa(
                question=sample.question,
                structure_path=structure_path,
                workbook_path=candidate.workbook_path,
                qa_artifact_dir=self._sample_dir(sample) / "qa",
                fallback_prompt=image_prompt,
                fallback_image_path=candidate.image_path,
                fallback_text_prompt=fallback_prompt,
                related_structure_paths=related_structure_paths,
                excluded_sheet_names=self._perfect_retrieval_excluded_sheets(
                    candidate.workbook_path
                ),
                enable_final_answer_review=True,
            )
        responses.append(answer_response)
        self._progress(
            "done",
            sample=sample.sample_id,
            workbook=candidate.workbook_path.name,
            sheet=candidate.sheet_name,
        )
        retrieval_info = {
            "score": candidate.score,
            "lexical_score": candidate.lexical_score,
            "bm25_score": candidate.bm25_score,
            "embedding_score": candidate.embedding_score,
            "embedding_used": candidate.embedding_used,
            "fallback_used": getattr(candidate, "fallback_used", False),
            "retrieval_type": candidate.retrieval_type,
            "retrieval_level": candidate.retrieval_level,
            "retrieval_trace": list(candidate.retrieval_trace),
            "entity_score": candidate.entity_score,
            "matched_terms": list(candidate.matched_terms),
            "missing_terms": list(candidate.missing_terms),
            "retrieval_rank": candidate.retrieval_rank,
            "table_id": candidate.table_id,
            "table_name": candidate.table_name,
            "table_description": candidate.table_description,
            "perfect_retrieval": self.settings.perfect_retrieval,
            "retrieval_audit": list(candidate.retrieval_audit),
        }
        if hasattr(candidate, "reranker_selected_index"):
            retrieval_info["reranker_selected_index"] = getattr(
                candidate, "reranker_selected_index"
            )
            retrieval_info["reranker_rationale"] = getattr(
                candidate, "reranker_rationale", ""
            )

        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=candidate.structure_text,
            predicted_answer=answer_response.content,
            latency=self.stop_timer(start_time),
            token_usage=token_usage(responses),
            metadata={
                "structure_path": display_path(structure_path),
                "thinking_trace_path": display_path(
                    candidate.directory / "thinking_trace.txt"
                )
                if (candidate.directory / "thinking_trace.txt").is_file()
                else None,
                "workbook_path": str(candidate.workbook_path.resolve()),
                "image_path": display_path(candidate.image_path),
                "html_path": display_path(candidate.html_path)
                if candidate.html_path
                else None,
                "workbook_source_format": "xlsx",
                "workbook_sheets": list(candidate.sheet_names)
                or [candidate.sheet_name],
                "verification": self._prepared_verification(candidate.directory),
                "artifact_dir": display_path(candidate.directory),
                "image_tiles": read_image_tiles(candidate.directory),
                "retrieval_info": retrieval_info,
                "metadata_yaml_path": display_path(
                    candidate.directory / "metadata.yaml"
                ),
                "render_metadata_path": display_path(
                    candidate.directory / "table.metadata.json"
                )
                if (candidate.directory / "table.metadata.json").is_file()
                else None,
                "changelog_path": display_path(candidate.directory / "changelog.md"),
                "events_path": display_path(candidate.directory / "events.jsonl"),
                "iteration_artifact_dir": display_path(
                    candidate.directory / "iterations"
                ),
                "qa": qa_info,
            },
        )

    @staticmethod
    def _prepared_verification(directory: Path) -> dict[str, Any]:
        metadata_path = directory / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        verification = metadata.get("verification") if isinstance(metadata, dict) else None
        if isinstance(verification, dict):
            return dict(verification)
        return {"status": "good", "feedback": "Retrieved from encoded source"}

    def _related_structure_paths(self, candidate: SourceCandidate) -> list[Path]:
        source_root = candidate.directory.parent
        workbook_path = candidate.workbook_path.resolve()
        paths = []
        if not source_root.is_dir():
            return paths
        for source_dir in sorted(source_root.iterdir()):
            metadata_path = source_dir / "metadata.json"
            structure_path = source_dir / "structure.yaml"
            if not metadata_path.is_file() or not structure_path.is_file():
                continue
            try:
                metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            candidate_workbook_text = str(metadata.get("workbook_path", ""))
            candidate_workbook = Path(candidate_workbook_text)
            same_path = candidate_workbook.resolve() == workbook_path
            same_name = safe_name(
                PureWindowsPath(candidate_workbook_text).name
            ) == safe_name(workbook_path.name)
            if candidate_workbook_text and (same_path or same_name):
                sheet_name = str(metadata.get("sheet_name", ""))
                if (
                    self.settings.perfect_retrieval
                    and self.source_retriever.is_perfect_retrieval_excluded(
                        workbook_path, sheet_name
                    )
                ):
                    continue
                paths.append(structure_path)
        return paths

    def _perfect_retrieval_excluded_sheets(
        self, workbook_path: Path
    ) -> list[str]:
        if not self.settings.perfect_retrieval:
            return []
        workbook = openpyxl.load_workbook(
            workbook_path, read_only=True, data_only=True
        )
        try:
            return [
                sheet_name
                for sheet_name in workbook.sheetnames
                if self.source_retriever.is_perfect_retrieval_excluded(
                    workbook_path, sheet_name
                )
            ]
        finally:
            workbook.close()

    def _fit_context(self, table_content: str) -> str:
        if len(table_content) <= self.settings.max_context_chars:
            return table_content
        return table_content[: self.settings.max_context_chars] + "\n...TRUNCATED..."

    def _answer_prompt(
        self, sample: EvalSample, table_context: str, structure_text: str
    ) -> str:
        return self.prompts.answer_prompt(sample, table_context, structure_text)

    def _sample_dir(self, sample: EvalSample) -> Path:
        raw = f"{sample.sample_id}:{sample.table_id}:{sample.question}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return self._artifact_dir / safe_name(sample.sample_id)[:80] / digest
