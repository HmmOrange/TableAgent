"""The QA stage behind TableAgentPipeline: answer by exploring the workbook, or fall back to a direct answer.

Fallback: when the exploration fails, the model answers from the needs the exploration did find, or else from the source context.
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

FAILURE_NOTE_CHARS = 1500


class WorkbookQA(RuntimeComponent):
    def run(
        self,
        *,
        question: str,
        workbook_path: Path,
        qa_artifact_dir: Path,
        fallback_prompt: str | Callable[[], str],
        fallback_image_path: Path | None = None,
        fallback_text_prompt: str | None = None,
        excluded_sheet_names: list[str] | None = None,
    ) -> tuple[LLMResponse, dict[str, Any]]:
        result = answer_question(
            self.llm,
            question,
            Path(workbook_path),
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

    def _info(self, result: AnswerResult, response: LLMResponse) -> dict[str, Any]:
        """The QA metadata reported per sample."""
        calls = result.llm_calls
        ledger = result.ledger
        return {
            "success": result.success,
            "error": result.error,
            "execution_time": result.seconds,
            "token_usage": {"prompt": int(response.prompt_tokens or 0), "completion": int(response.completion_tokens or 0)},
            "artifacts": result.artifacts,
            "fallback_used": not result.success,
            "answer_route": "workbook" if result.success else "fallback",
            "replan_count": ledger.hypothesis_revisions,
            "step_count": len(ledger.decisions),
            "inspection_count": len(ledger.evidence),
            "revision_count": ledger.revision_count(),
            "llm_call_count": len(calls),
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
