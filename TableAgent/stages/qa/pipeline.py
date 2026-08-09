from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

from TableAgent.llm import LLMResponse
from TableAgent.QA.runner import TableQARunner


class PipelineQAMixin:
    """Run verified QA and provide conservative fallback helpers."""

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
        related_structure_paths: list[Path] | None = None,
        excluded_sheet_names: list[str] | None = None,
        enable_final_answer_review: bool = False,
    ) -> tuple[LLMResponse, dict[str, Any]]:
        structure_error: str | None = None
        structure_failure_source = "missing_structure"
        try:
            structure_text = structure_path.read_text(encoding="utf-8")
        except OSError:
            structure_error = "Missing structure.yaml: no usable structure was supplied"
        else:
            parsed_structure = self._parse_structure(structure_text)
            if not structure_text.strip():
                structure_error = "Missing structure.yaml: the supplied structure is empty"
            elif (
                not parsed_structure
                or parsed_structure.get("status") == "not_good"
                or "error" in parsed_structure
            ):
                structure_error = "Invalid structure.yaml: the supplied structure is not usable YAML"
                structure_failure_source = "invalid_structure"

        if structure_error is not None:
            response = self.qa_agent.run(
                prompt=fallback_prompt,
                image_path=fallback_image_path,
                fallback_prompt=fallback_text_prompt,
            )
            return response, self._fallback_qa_info(
                response, structure_error, structure_failure_source
            )

        runner_kwargs = {
            "structure_path": str(structure_path),
            "workbook_path": str(workbook_path),
            "llm_client": self.llm,
            "config": {
                "table_agent": {
                    **self._serialize_config_value(self.settings),
                    "artifact_dir": str(qa_artifact_dir),
                    "qa_excluded_sheet_names": list(excluded_sheet_names or []),
                    "qa_final_answer_review": enable_final_answer_review,
                },
                "qa_artifact_dir": str(qa_artifact_dir),
            },
            "table_retriever": self.table_retriever,
            "related_structure_paths": related_structure_paths,
        }
        if self._progress_callback is not None:
            runner_kwargs["progress_callback"] = self._progress_callback
        pipeline_module = sys.modules.get(self.__class__.__module__)
        runner_type = getattr(pipeline_module, "TableQARunner", TableQARunner)
        with runner_type(**runner_kwargs) as runner:
            result = runner.run(question)
            llm_calls = list(getattr(result, "llm_calls", []) or [])
            capped_call_count = sum(
                1 for call in llm_calls if bool(call.get("token_capped"))
            )
            qa_info = {
                "success": result.success,
                "error": result.error,
                "execution_time": result.execution_time,
                "token_usage": result.token_usage,
                "artifacts": result.artifacts,
                "fallback_used": not result.success,
                "replan_count": int(getattr(result, "replan_count", 0) or 0),
                "subtask_retry_count": int(
                    getattr(result, "subtask_retry_count", 0) or 0
                ),
                "qa_max_retries": int(getattr(result, "qa_max_retries", 0) or 0),
                "llm_call_count": len(llm_calls),
                "token_capped_call_count": capped_call_count,
                "llm_total_ms": sum(
                    int(call.get("duration_ms", 0) or 0) for call in llm_calls
                ),
                "llm_calls": llm_calls,
                "generation_max_tokens": self.settings.generation_max_tokens,
                "thinking_enabled": self._qa_thinking_enabled(),
                "answer_route": (
                    "common_info"
                    if result.success
                    and getattr(result, "plan", None)
                    and all(
                        subtask.category == "common_info"
                        for subtask in getattr(result, "plan", [])
                    )
                    else "normal"
                ),
            }
            if result.success and result.final_answer is not None:
                return (
                    LLMResponse(
                        content=result.final_answer,
                        prompt_tokens=int(result.token_usage.get("prompt", 0) or 0),
                        completion_tokens=int(
                            result.token_usage.get("completion", 0) or 0
                        ),
                    ),
                    qa_info,
                )

        if result.success:
            raise RuntimeError("TableQARunner returned success without a final answer")
        verified_fallback_prompt = self._verified_observation_fallback_prompt(
            question, result
        )
        if verified_fallback_prompt:
            response = self.qa_agent.run(prompt=verified_fallback_prompt)
            qa_info["fallback_source"] = "verified_inspection_observations"
        else:
            response = self.qa_agent.run(
                prompt=fallback_prompt,
                image_path=fallback_image_path,
                fallback_prompt=fallback_text_prompt,
            )
            qa_info["fallback_source"] = "source_context"
        response.prompt_tokens += int(result.token_usage.get("prompt", 0) or 0)
        response.completion_tokens += int(
            result.token_usage.get("completion", 0) or 0
        )
        return response, qa_info

    @staticmethod
    def _parse_structure(structure_text: str) -> dict[str, Any]:
        from TableAgent.structure.layout.parsing import _parse_yaml_mapping

        return _parse_yaml_mapping(structure_text)

    def _fallback_qa_info(
        self, response: LLMResponse, error: str, source: str
    ) -> dict[str, Any]:
        return {
            "success": False,
            "error": error,
            "execution_time": 0,
            "token_usage": {
                "prompt": int(getattr(response, "prompt_tokens", 0) or 0),
                "completion": int(getattr(response, "completion_tokens", 0) or 0),
            },
            "artifacts": {},
            "fallback_used": True,
            "fallback_source": source,
            "replan_count": 0,
            "subtask_retry_count": 0,
            "qa_max_retries": self.settings.qa_max_retries,
            "llm_call_count": 1,
            "token_capped_call_count": int(
                bool(getattr(response, "token_capped", False))
            ),
            "llm_total_ms": 0,
            "llm_calls": [],
            "generation_max_tokens": self.settings.generation_max_tokens,
            "thinking_enabled": self._qa_thinking_enabled(),
            "answer_route": "fallback",
        }

    def _qa_thinking_enabled(self) -> bool:
        extra_body = getattr(self.llm, "extra_body", {})
        if not isinstance(extra_body, dict):
            return True
        template_kwargs = extra_body.get("chat_template_kwargs")
        if not isinstance(template_kwargs, dict):
            return True
        return bool(template_kwargs.get("enable_thinking", True))

    def _verified_observation_fallback_prompt(
        self, question: str, result: Any
    ) -> str | None:
        failure_text = "\n".join(
            [str(getattr(result, "error", "") or "")]
            + [
                str(getattr(output, "observation", "") or "")
                for output in getattr(result, "subtask_outputs", [])
                if not getattr(output, "success", False)
            ]
        ).casefold()
        unsafe_markers = (
            "wrong table", "wrong sheet", "wrong header", "wrong column",
            "incorrect header", "incorrect column", "filter drift",
            "incomplete coverage", "wrong date", "wrong target",
            "unfiltered aggregate", "incorrect count", "incorrect value",
        )
        if any(marker in failure_text for marker in unsafe_markers):
            return None

        subtasks = {subtask.id: subtask for subtask in getattr(result, "plan", [])}
        observations = []
        for output in getattr(result, "subtask_outputs", []):
            subtask = subtasks.get(output.subtask_id)
            layer = getattr(output, "layer", "") or getattr(subtask, "layer", "")
            if not output.success or layer != "inspect":
                continue
            observation = str(output.observation or "").strip()
            if observation:
                observations.append(
                    f"## Verified inspection: {output.description}\n{observation}"
                )
        if not observations:
            return None
        evidence = "\n\n".join(observations)
        if len(evidence) > self.settings.max_context_chars:
            evidence = evidence[: self.settings.max_context_chars] + "\n...[truncated]"
        return (
            f"Question:\n{question}\n\n"
            "The QA synthesis failed, but the following observations were successfully computed from the "
            "verified spreadsheet structure and workbook. Answer using only these observations. Preserve exact "
            "header-to-value ownership and every label explicitly enumerated in the question, apply every constraint "
            "stated in the question, and do not add fields that were not requested. If the observations contain "
            "multiple records, select only records matching all question constraints.\n\n"
            f"{evidence}\n\nAnswer:"
        )

    @staticmethod
    def _select_table_id(structure_path: Path, question: str) -> str | None:
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
                [
                    table_id,
                    str(table.get("name") or ""),
                    str(table.get("description") or ""),
                ]
                + [
                    " ".join(
                        str(header.get(field) or "")
                        for field in ("id", "label", "description")
                    )
                    for header in table.get("headers") or []
                    if isinstance(header, dict)
                ]
            )
            score = len(
                query_terms
                & set(re.findall(r"[a-z0-9]+", searchable.lower()))
            )
            candidate = (score, table_id)
            if best is None or candidate > best:
                best = candidate
        return best[1] if best else None

    def _generate_answer_with_image(
        self, *, prompt: str, image_path: Path, fallback_prompt: str | None = None
    ) -> LLMResponse:
        return self.qa_agent.run(
            prompt=prompt, image_path=image_path, fallback_prompt=fallback_prompt
        )
