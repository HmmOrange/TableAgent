"""The QA stage behind TableAgentPipeline: answer by exploring the workbook, or fall back to a direct answer.

Fallbacks: without a usable structure.yaml the model answers from the source context directly; when the
exploration fails, it answers from the needs the exploration did find, or else from the source context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.component import RuntimeComponent
from TableAgent.stages.qa.orchestrator import AnswerResult, answer_question
from TableAgent.stages.qa.ledger import Ledger
from TableAgent.stages.qa.prompts import EVIDENCE_ANSWER_PROMPT
from TableAgent.stages.structure.layout.parsing import _parse_yaml_mapping

FAILURE_NOTE_CHARS = 1500


class WorkbookQA(RuntimeComponent):
    def run(
        self,
        *,
        question: str,
        structure_path: Path,
        workbook_path: Path,
        qa_artifact_dir: Path,
        fallback_prompt: str | Callable[[], str],
        fallback_image_path: Path | None = None,
        fallback_text_prompt: str | None = None,
        related_structure_paths: list[Path] | None = None,
        excluded_sheet_names: list[str] | None = None,
    ) -> tuple[LLMResponse, dict[str, Any]]:
        problem, source = _structure_problem(structure_path)
        if problem:
            response = self.qa_agent.run(
                prompt=_resolve(fallback_prompt), image_path=fallback_image_path, fallback_prompt=fallback_text_prompt,
            )
            return response, {**self._info(None, response), "error": problem, "fallback_source": source}

        result = answer_question(
            self.llm,
            question,
            Path(workbook_path),
            [Path(structure_path), *(Path(path) for path in related_structure_paths or [])],
            Path(qa_artifact_dir),
            {name.strip().casefold() for name in excluded_sheet_names or [] if name.strip()},
            progress=self._progress_callback,
            checks=self.settings.qa_checker,
            sees_value=self.settings.qa_checker_sees_value,
            view=self.settings.qa_checker_view,
            synthesis_check=self.settings.qa_synthesis_check,
            vision_client=self.layout_vlm,
            renderer=self.workbook_renderer,
        )
        if result.success:
            response = LLMResponse(
                content=result.answer,
                prompt_tokens=result.token_usage["prompt"],
                completion_tokens=result.token_usage["completion"],
            )
            return response, self._info(result, response)

        evidence = _checked_evidence(result.ledger)
        if evidence:
            response = self.qa_agent.run(prompt=EVIDENCE_ANSWER_PROMPT.format(question=question, evidence=evidence))
            source = "found_evidence"
        else:
            note = (
                "An exploration of the workbook failed; do not repeat its mistake:\n"
                f"{str(result.error)[:FAILURE_NOTE_CHARS]}\n\n"
            )
            response = self.qa_agent.run(
                prompt=note + _resolve(fallback_prompt),
                image_path=fallback_image_path,
                fallback_prompt=note + fallback_text_prompt if fallback_text_prompt else None,
            )
            source = "source_context"
        response.prompt_tokens += result.token_usage["prompt"]
        response.completion_tokens += result.token_usage["completion"]
        return response, {**self._info(result, response), "fallback_source": source}

    def _info(self, result: AnswerResult | None, response: LLMResponse) -> dict[str, Any]:
        """The QA metadata reported per sample; `result` is None when no exploration ran."""
        calls = result.llm_calls if result is not None else []
        ledger = result.ledger if result is not None else None
        return {
            "success": bool(result and result.success),
            "error": result.error if result is not None else None,
            "execution_time": result.seconds if result is not None else 0,
            "token_usage": {"prompt": int(response.prompt_tokens or 0), "completion": int(response.completion_tokens or 0)},
            "artifacts": result.artifacts if result is not None else {},
            "fallback_used": not (result and result.success),
            "answer_route": "workbook" if result and result.success else "fallback",
            "replan_count": ledger.hypothesis_revisions if ledger else 0,
            "step_count": len(ledger.decisions) if ledger else 0,
            "inspection_count": len(ledger.evidence) if ledger else 0,
            "revision_count": ledger.revision_count() if ledger else 0,
            "llm_call_count": len(calls) if result is not None else 1,
            "token_capped_call_count": sum(1 for call in calls if call.get("token_capped")),
            "llm_total_ms": sum(int(call.get("duration_ms", 0)) for call in calls),
            "llm_calls": calls,
            "generation_max_tokens": self.settings.generation_max_tokens,
            "thinking_enabled": self._thinking_enabled(),
        }

    def _thinking_enabled(self) -> bool:
        extra_body = getattr(self.llm, "extra_body", {})
        template_kwargs = extra_body.get("chat_template_kwargs") if isinstance(extra_body, dict) else None
        return bool(template_kwargs.get("enable_thinking", True)) if isinstance(template_kwargs, dict) else True


def _structure_problem(structure_path: Path) -> tuple[str, str]:
    """Why structure.yaml cannot guide the exploration, and the fallback source to report; empty when usable."""
    try:
        text = Path(structure_path).read_text(encoding="utf-8")
    except OSError:
        return "Missing structure.yaml: no usable structure was supplied", "missing_structure"
    if not text.strip():
        return "Missing structure.yaml: the supplied structure is empty", "missing_structure"
    parsed = _parse_yaml_mapping(text)
    if not parsed or parsed.get("status") == "not_good" or "error" in parsed:
        return "Invalid structure.yaml: the supplied structure is not usable YAML", "invalid_structure"
    return "", ""


def _checked_evidence(ledger: Ledger) -> str:
    """Found needs that no check refuted (or whose refuting check was ruled wrong), one line each."""
    lines = []
    for need in ledger.found_needs():
        item = ledger.latest(need.id)
        if item.disputed:
            continue
        lines.append(
            f"- {need.description}: {json.dumps(item.value, ensure_ascii=False, default=str)} "
            f"(cells {', '.join(item.cells) or 'not given'}; {item.where or 'where not given'})"
        )
    return "\n".join(lines)


def _resolve(prompt: str | Callable[[], str]) -> str:
    """Fallback prompts may be built lazily, only when a fallback runs."""
    return prompt() if callable(prompt) else prompt
