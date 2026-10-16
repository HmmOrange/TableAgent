from __future__ import annotations

import json
import re
import shutil
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.component import RuntimeComponent
from TableAgent.pipeline.contracts import PipelineOutput, PipelineRuntimeContract
from TableAgent.pipeline.sample import EvalSample, has_workbook_sources
from TableAgent.stages.qa import QAInput
from TableAgent.stages.structure.layout.parsing import extract_layout_structure_result
from TableAgent.utils.llm_metrics import token_usage
from TableAgent.utils.paths import display_path, safe_name

from .prompts import (
    LOCATE_PROMPT_TEMPLATE,
    LOCATE_SYSTEM_PROMPT,
    REPAIR_PROMPT_TEMPLATE,
    STRUCTURE_PROMPT_TEMPLATE,
    STRUCTURE_SYSTEM_PROMPT,
)
from .static_ingestion import StaticRegion, StaticWorkbookIndex, StaticWorkbookIngestor


_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)
_WORD = re.compile(r"\w+", flags=re.UNICODE)


class QueryScopedStructurePipeline(RuntimeComponent):
    """Locate text regions, extract a minimal structure, verify it, then run QA."""

    def __init__(self, runtime: PipelineRuntimeContract):
        super().__init__(runtime)
        self.ingestor = StaticWorkbookIngestor(runtime.settings, runtime.llm)
        self._indexes: dict[tuple[str, tuple[str, ...]], StaticWorkbookIndex] = {}
        self._index_lock = threading.RLock()

    def prepare_samples(self, samples: list[EvalSample]) -> None:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        tasks: list[tuple[Path, tuple[str, ...], str]] = []
        for sample in samples:
            if not has_workbook_sources(sample):
                raise ValueError(
                    "query-scoped structure requires .xlsx/.xlsm workbook sources; "
                    f"sample {sample.sample_id!r} is unsupported"
                )
            selected_sheets = self._selected_sheets(sample)
            for path in self._workbook_paths(sample):
                key = (str(path.resolve()), selected_sheets)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append((path, selected_sheets, sample.sample_id))

        def prepare_one(
            task: tuple[Path, tuple[str, ...], str],
        ) -> tuple[tuple[Path, tuple[str, ...], str], StaticWorkbookIndex]:
            path, selected_sheets, _ = task
            return task, self._prepare_index(path, selected_sheets)

        def report_prepared(
            task: tuple[Path, tuple[str, ...], str],
            index: StaticWorkbookIndex,
        ) -> None:
            path, _, sample_id = task
            for sheet in index.sheets:
                self.runtime._progress(
                    "structure_done",
                    sample=sample_id,
                    workbook=path.name,
                    sheet=sheet.sheet_name,
                    status="cached" if index.cache_hit else "indexed",
                )

        worker_count = (
            min(
                max(1, int(getattr(self.settings, "max_workers", 1))),
                len(tasks),
            )
            if tasks
            else 0
        )
        if worker_count > 1:
            first_error: Exception | None = None
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="table-agent-query-static",
            ) as executor:
                futures = [executor.submit(prepare_one, task) for task in tasks]
                for future in as_completed(futures):
                    try:
                        report_prepared(*future.result())
                    except Exception as exc:
                        # ThreadPoolExecutor waits for submitted workers when its
                        # context exits. Keep draining their results so successful
                        # work still emits structure_done instead of silently
                        # finishing after the first failed future.
                        if first_error is None:
                            first_error = exc
            if first_error is not None:
                raise first_error
        else:
            for task in tasks:
                report_prepared(*prepare_one(task))

    def run(self, sample: EvalSample) -> PipelineOutput:
        if not has_workbook_sources(sample):
            raise ValueError("query-scoped structure requires workbook sources")
        started = self.start_timer()
        selected_sheets = self._selected_sheets(sample)
        indexes = [
            self._prepare_index(path, selected_sheets)
            for path in self._workbook_paths(sample)
        ]
        query_responses: list[LLMResponse] = []
        query_dir = self._qa_sample_dir(sample) / "query_structure"
        query_dir.mkdir(parents=True, exist_ok=True)

        locate_prompt = self._locate_prompt(sample.question, indexes)
        (query_dir / "locate_prompt.txt").write_text(
            locate_prompt, encoding="utf-8"
        )
        locate_response = self.llm.generate(
            locate_prompt,
            system_prompt=LOCATE_SYSTEM_PROMPT,
        )
        query_responses.append(locate_response)
        (query_dir / "locate_response.txt").write_text(
            locate_response.content, encoding="utf-8"
        )
        selected, reasons, locate_fallback = self._parse_selection(
            locate_response.content,
            sample.question,
            indexes,
        )
        primary_workbook = selected[0].workbook_path.resolve()
        dropped = [
            region.region_id
            for region in selected
            if region.workbook_path.resolve() != primary_workbook
        ]
        selected = [
            region
            for region in selected
            if region.workbook_path.resolve() == primary_workbook
        ]
        selection_payload = {
            "question": sample.question,
            "fallback_used": locate_fallback,
            "selected_regions": [
                {
                    "region_id": region.region_id,
                    "workbook_path": str(region.workbook_path),
                    "sheet_name": region.sheet_name,
                    "cell_range": region.cell_range,
                    "reason": reasons.get(region.region_id, ""),
                }
                for region in selected
            ],
            "dropped_cross_workbook_regions": dropped,
        }
        (query_dir / "selection.json").write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        grouped: dict[str, list[StaticRegion]] = defaultdict(list)
        for region in selected:
            grouped[region.sheet_name].append(region)
        primary_sheet = selected[0].sheet_name
        ordered_sheets = [primary_sheet] + sorted(
            name for name in grouped if name != primary_sheet
        )
        structure_paths: list[Path] = []
        structure_texts: dict[str, str] = {}
        verification_payload: dict[str, Any] = {}
        selected_evidence: list[str] = []

        for sheet_name in ordered_sheets:
            regions = grouped[sheet_name]
            sheet_dir = query_dir / "sheets" / safe_name(sheet_name)
            sheet_dir.mkdir(parents=True, exist_ok=True)
            index = next(
                item
                for item in indexes
                if item.workbook_path.resolve() == primary_workbook
            )
            sheet_index = next(
                item for item in index.sheets if item.sheet_name == sheet_name
            )
            header_context = sheet_index.header_context_path.read_text(
                encoding="utf-8"
            )
            evidence = "\n\n".join(
                region.source_path.read_text(encoding="utf-8")
                for region in regions
            )
            evidence = self._limit(
                evidence, self.settings.query_location_max_prompt_chars
            )
            selected_evidence.append(evidence)
            structure_prompt = STRUCTURE_PROMPT_TEMPLATE.format(
                question=sample.question,
                workbook_name=Path(primary_workbook).name,
                sheet_name=sheet_name,
                region_ids=", ".join(region.region_id for region in regions),
                header_context=header_context,
                region_cells=evidence,
            )
            (sheet_dir / "structure_prompt.txt").write_text(
                structure_prompt, encoding="utf-8"
            )
            response = self.llm.generate(
                structure_prompt,
                system_prompt=STRUCTURE_SYSTEM_PROMPT,
            )
            query_responses.append(response)
            (sheet_dir / "structure_response.txt").write_text(
                response.content, encoding="utf-8"
            )
            structure_text, verification = self._verify_with_repair(
                sample=sample,
                workbook_path=Path(primary_workbook),
                sheet_name=sheet_name,
                evidence=evidence,
                initial_response=response,
                sheet_dir=sheet_dir,
                responses=query_responses,
            )
            structure_path = sheet_dir / "structure.yaml"
            structure_path.write_text(structure_text + "\n", encoding="utf-8")
            structure_paths.append(structure_path)
            structure_texts[sheet_name] = structure_text
            verification_payload[sheet_name] = verification

        primary_structure_path = query_dir / "structure.yaml"
        shutil.copyfile(structure_paths[0], primary_structure_path)
        structure_runtime = self.stop_timer(started)

        qa_started = self.start_timer()
        fallback_context = self._limit(
            "\n\n".join(selected_evidence), self.settings.max_context_chars
        )
        qa_output = self.stages.qa.run(
            QAInput(
                question=sample.question,
                structure_path=primary_structure_path,
                workbook_path=Path(primary_workbook),
                qa_artifact_dir=self._qa_sample_dir(sample),
                fallback_prompt=self.runtime._answer_prompt(
                    sample,
                    fallback_context,
                    structure_texts[primary_sheet],
                ),
                related_structure_paths=tuple(structure_paths[1:]),
                enable_final_answer_review=self.settings.qa_final_answer_review,
            )
        )
        query_responses.append(qa_output.response)
        qa_runtime = self.stop_timer(qa_started)
        if isinstance(qa_output.metadata, dict):
            qa_runtime = float(
                qa_output.metadata.get("execution_time") or qa_runtime
            )
        total_runtime = structure_runtime + qa_runtime
        static_usage = {
            "prompt": sum(
                response.prompt_tokens
                for index in indexes
                for response in index.responses
            ),
            "completion": sum(
                response.completion_tokens
                for index in indexes
                for response in index.responses
            ),
        }
        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table=structure_texts[primary_sheet],
            predicted_answer=qa_output.response.content,
            latency=total_runtime,
            token_usage=token_usage(query_responses),
            metadata={
                "query_scoped_structure": True,
                "structure_path": display_path(primary_structure_path),
                "related_structure_paths": [
                    display_path(path) for path in structure_paths[1:]
                ],
                "workbook_path": str(primary_workbook),
                "workbook_sheets": ordered_sheets,
                "artifact_dir": display_path(query_dir),
                "static_cache_dirs": [
                    display_path(index.cache_dir) for index in indexes
                ],
                "static_cache_hits": [index.cache_hit for index in indexes],
                "static_ingestion_token_usage": static_usage,
                "location": selection_payload,
                "verification": verification_payload,
                "qa": qa_output.metadata,
                "structure_runtime": structure_runtime,
                "understanding_runtime": structure_runtime,
                "retrieval_runtime": 0.0,
                "qa_runtime": qa_runtime,
                "stage_runtimes": {
                    "structure": structure_runtime,
                    "retrieval": 0.0,
                    "qa": qa_runtime,
                    "total": total_runtime,
                },
            },
        )

    def _verify_with_repair(
        self,
        *,
        sample: EvalSample,
        workbook_path: Path,
        sheet_name: str,
        evidence: str,
        initial_response: LLMResponse,
        sheet_dir: Path,
        responses: list[LLMResponse],
    ) -> tuple[str, dict[str, Any]]:
        content = initial_response.content
        last_feedback = "Structure could not be parsed."
        last_structure_text = ""
        attempts: list[dict[str, Any]] = []
        max_attempts = self.settings.query_structure_max_retries + 1
        for attempt in range(1, max_attempts + 1):
            parsed = extract_layout_structure_result(content)
            iteration_dir = sheet_dir / "iterations" / f"iteration_{attempt:02d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            (iteration_dir / "model_response.txt").write_text(
                content, encoding="utf-8"
            )
            (iteration_dir / "structure_after.yaml").write_text(
                parsed.structure_text, encoding="utf-8"
            )
            result = self.runtime.verifier.run(
                workbook_path=workbook_path,
                sheet_name=sheet_name,
                structure_text=parsed.structure_text,
                iteration_dir=iteration_dir,
                preflight_errors=parsed.preflight_errors,
            )
            if parsed.structure_text.strip():
                last_structure_text = parsed.structure_text.strip()
            attempts.append(
                {
                    "attempt": attempt,
                    "status": result.status,
                    "feedback": result.feedback,
                    "null_fields": result.null_fields,
                    "iteration_dir": display_path(iteration_dir),
                }
            )
            if result.is_good:
                return result.structure_text, {
                    "status": "good",
                    "attempts": attempts,
                }
            last_feedback = result.feedback or last_feedback
            if attempt >= max_attempts:
                break
            repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
                question=sample.question,
                workbook_name=workbook_path.name,
                sheet_name=sheet_name,
                evidence=evidence,
                candidate=parsed.structure_text or content,
                feedback=last_feedback,
            )
            (iteration_dir / "repair_prompt.txt").write_text(
                repair_prompt, encoding="utf-8"
            )
            repair = self.llm.generate(
                repair_prompt,
                system_prompt=STRUCTURE_SYSTEM_PROMPT,
            )
            responses.append(repair)
            content = repair.content
        return last_structure_text or content.strip(), {
            "status": "not_good",
            "feedback": last_feedback,
            "attempts": attempts,
            "retry_exhausted": True,
            "used_unverified_structure": True,
        }

    def _locate_prompt(
        self,
        question: str,
        indexes: list[StaticWorkbookIndex],
    ) -> str:
        workbook_summary = "\n\n".join(
            f"## Workbook: {index.workbook_path.name}\n{index.summary}"
            for index in indexes
        )
        entries = []
        for index in indexes:
            for region in index.regions:
                entries.append(
                    f"- region_id: {region.region_id}\n"
                    f"  workbook: {index.workbook_path.name}\n"
                    f"  sheet: {region.sheet_name}\n"
                    f"  range: {region.cell_range}\n"
                    f"  summary: {region.summary}"
                )
        prompt = LOCATE_PROMPT_TEMPLATE.format(
            question=question,
            workbook_summary=workbook_summary,
            region_catalog="\n".join(entries),
            max_regions=self.settings.query_location_max_regions,
        )
        return self._limit(prompt, self.settings.query_location_max_prompt_chars)

    def _parse_selection(
        self,
        content: str,
        question: str,
        indexes: list[StaticWorkbookIndex],
    ) -> tuple[list[StaticRegion], dict[str, str], bool]:
        by_id = {
            region.region_id: region
            for index in indexes
            for region in index.regions
        }
        selected_ids: list[str] = []
        reasons: dict[str, str] = {}
        match = _JSON_OBJECT.search(content)
        try:
            payload = json.loads(match.group(0) if match else content)
            raw_regions = payload.get("selected_regions", [])
            if isinstance(raw_regions, list):
                for item in raw_regions:
                    if isinstance(item, str):
                        region_id, reason = item, ""
                    elif isinstance(item, dict):
                        region_id = str(item.get("region_id") or "")
                        reason = str(item.get("reason") or "")
                    else:
                        continue
                    if region_id in by_id and region_id not in selected_ids:
                        selected_ids.append(region_id)
                        reasons[region_id] = reason
        except (json.JSONDecodeError, AttributeError, TypeError):
            selected_ids = []
        max_regions = self.settings.query_location_max_regions
        selected_ids = selected_ids[:max_regions]
        if selected_ids:
            return [by_id[value] for value in selected_ids], reasons, False

        question_terms = set(_WORD.findall(question.casefold()))
        ranked: list[tuple[int, str, StaticRegion]] = []
        for region_id, region in by_id.items():
            haystack = f"{region.sheet_name} {region.summary}".casefold()
            overlap = len(question_terms.intersection(_WORD.findall(haystack)))
            ranked.append((overlap, region_id, region))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        fallback = [item[2] for item in ranked[:max_regions]]
        if not fallback:
            raise RuntimeError(
                "Static ingestion produced no selectable spreadsheet regions"
            )
        return (
            fallback,
            {item.region_id: "lexical fallback" for item in fallback},
            True,
        )

    def _prepare_index(
        self,
        path: Path,
        selected_sheets: tuple[str, ...],
    ) -> StaticWorkbookIndex:
        key = (str(path.resolve()), selected_sheets)
        with self._index_lock:
            cached = self._indexes.get(key)
        if cached is not None:
            return cached
        index = self.ingestor.prepare(path, selected_sheets=selected_sheets)
        with self._index_lock:
            self._indexes[key] = index
        return index

    @staticmethod
    def _workbook_paths(sample: EvalSample) -> list[Path]:
        return [
            Path(value.strip())
            for value in str(sample.table_path or "").split(";")
            if value.strip()
        ]

    @staticmethod
    def _selected_sheets(sample: EvalSample) -> tuple[str, ...]:
        raw = sample.raw if isinstance(sample.raw, dict) else {}
        values = raw.get("selected_sheets") or []
        if isinstance(values, str):
            values = [values]
        return tuple(str(value) for value in values if str(value))

    @staticmethod
    def _limit(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[:limit] + "\n...TRUNCATED..."
