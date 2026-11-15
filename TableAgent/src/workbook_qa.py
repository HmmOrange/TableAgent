"""The QA agent: answer by exploring the workbook (orchestrator.py), or fall back to a direct answer.

Fallback: when the exploration fails, the model answers from the needs the exploration did find, or else from the
table context the caller gives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from TableAgent.llm import BaseLLM, LLMResponse
from TableAgent.rendering.workbook import WorkbookRenderer
from TableAgent.settings import QASettings
from TableAgent.src.ledger import Ledger
from TableAgent.src.orchestrator import AnswerResult, answer_question
from TableAgent.src.prompts import DIRECT_ANSWER_SYSTEM_PROMPT, EVIDENCE_ANSWER_PROMPT
from TableAgent.src.utils.direct_answer import DirectAnswerer

FAILURE_NOTE_CHARS = 1500


class WorkbookQA:
    """Answers one question over one workbook.

    `vision_llm`, when given, lets the exploration look at rendered cell ranges. `settings.generation_max_tokens` is set as the
    generation cap of both models.
    """

    def __init__(self, llm: BaseLLM, settings: QASettings, vision_llm: BaseLLM | None = None):
        self.llm = llm
        self.vision_llm = vision_llm
        self.settings = settings
        self.renderer = WorkbookRenderer(settings)
        self.answerer = DirectAnswerer(llm, DIRECT_ANSWER_SYSTEM_PROMPT)
        self.progress: Callable[[str], None] | None = None
        if settings.generation_max_tokens is not None:
            for client in (llm, vision_llm):
                if hasattr(client, "max_tokens"):
                    client.max_tokens = settings.generation_max_tokens

    def run(
        self,
        *,
        question: str,
        workbook_path: Path,
        qa_artifact_dir: Path,
        fallback_prompt: str | Callable[[], str],
    ) -> tuple[LLMResponse, dict[str, Any]]:
        """The answer and the QA metadata; `fallback_prompt` (or a callable building it) is used only on failure."""
        result = answer_question(
            self.llm,
            question,
            Path(workbook_path),
            Path(qa_artifact_dir),
            set(),
            progress=self.progress,
            checks=self.settings.checker,
            sees_value=self.settings.checker_sees_value,
            view=self.settings.checker_view,
            synthesis_check=self.settings.synthesis_check,
            vision_client=self.vision_llm,
            renderer=self.renderer,
            plan_count=self.settings.plan_count,
            inspector_max_turns=self.settings.inspector_max_turns,
            max_returns=self.settings.max_returns,
            disabled_tools=self.settings.disabled_tools,
            save_artifacts=self.settings.save_artifacts,
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
            response = self.answerer.run(prompt=EVIDENCE_ANSWER_PROMPT.format(question=question, evidence=evidence))
            source = "found_evidence"
        else:
            note = (
                "An exploration of the workbook failed; do not repeat its mistake:\n"
                f"{str(result.error)[:FAILURE_NOTE_CHARS]}\n\n"
            )
            response = self.answerer.run(prompt=note + _resolve(fallback_prompt))
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
            "plan_count": self.settings.plan_count,
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
