from __future__ import annotations

import json
import re
from typing import Any, Optional

from TableAgent.QA.actions.base_action import BaseReviewAction, ReviewRequest, ReviewResult
from TableAgent.QA.language import answer_uses_required_language, required_answer_language
from TableAgent.prompts.review import REVIEW_SYSTEM_PROMPT, REVIEW_USER_PROMPT_TEMPLATE

_HIDDEN_WORKSPACE_NAMES = {
    "pd",
    "np",
    "openpyxl",
    "env",
    "operators",
    "Cell",
    "CellRange",
    "AxisSelection",
    "Header",
    "namespace",
}


def _format_workspace(env: Any, max_vars: int = 40) -> str:
    namespace = getattr(env, "execution_namespace", {})
    if not isinstance(namespace, dict):
        return "No workspace namespace is available."

    names = [
        name
        for name in namespace.keys()
        if isinstance(name, str)
        and not name.startswith("__")
        and name not in _HIDDEN_WORKSPACE_NAMES
    ]
    if not names:
        return "No user variables are currently defined."

    lines = []
    for name in names[:max_vars]:
        try:
            if hasattr(env, "preview_variable"):
                summary = env.preview_variable(name, rows=5, max_chars=400)
            elif hasattr(env, "notebook"):
                summary = env.notebook.summarize_value(namespace[name], max_chars=400)
            else:
                summary = repr(namespace[name])
        except Exception as exc:
            summary = f"<unavailable: {exc}>"
        lines.append(f"- {name}: {summary}")

    if len(names) > max_vars:
        lines.append(f"- ... {len(names) - max_vars} more variables omitted")
    return "\n".join(lines)


def _format_prior_history(env: Any) -> str:
    if hasattr(env, "get_history"):
        return env.get_history(last_n=5, include_output=True, max_code_len=1200, max_output_len=1000)
    if hasattr(env, "notebook"):
        return env.notebook.get_history(last_n=5, include_output=True, max_code_len=1200, max_output_len=1000)
    return "No notebook history is available."


class ReviewSubtaskAction(BaseReviewAction):
    """Action that decides whether a ReAct attempt completed its subtask."""
    name = "review_subtask"
    desc = "Review generated code, observation, and namespace state before accepting a subtask attempt."

    def __init__(self, env: Any, llm_client: Optional[Any] = None):
        self.env = env
        self.llm_client = llm_client

    def run(self, request: ReviewRequest) -> ReviewResult:
        local_result = self._local_review(request)
        if not local_result.accepted:
            result = local_result
        elif self.llm_client:
            result = self._llm_review(request)
        else:
            result = local_result

        self.env.logger.log_event("review_subtask", {
            "action": self.name,
            "subtask_id": request.subtask.id,
            "round_num": request.round_num,
            "accepted": result.accepted,
            "score": result.score,
            "feedback": result.feedback,
        })
        return result

    def _local_review(self, request: ReviewRequest) -> ReviewResult:
        if not request.execution.success:
            return ReviewResult(
                accepted=False,
                feedback=request.execution.error or "Execution failed; revise the code.",
                score=0.0,
            )

        if request.require_final_answer and "final_answer" not in self.env.execution_namespace:
            return ReviewResult(
                accepted=False,
                feedback="Execution succeeded, but `final_answer` was not set.",
                score=0.0,
            )

        if request.require_final_answer:
            answer = str(self.env.execution_namespace.get("final_answer", ""))
            language = required_answer_language(request.question)
            if not answer_uses_required_language(answer, language, question=request.question):
                return ReviewResult(
                    accepted=False,
                    feedback=f"`final_answer` must be written in {language}.",
                    score=0.0,
                )

        if request.subtask.layer == "table_inspect":
            selected = self.env.execution_namespace.get("selected_table_ids")
            has_selection = bool(selected) and (
                isinstance(selected, str)
                or isinstance(selected, (list, tuple, set))
            )
            if not has_selection:
                return ReviewResult(
                    accepted=False,
                    feedback="Execution succeeded, but `selected_table_ids` was not set to a non-empty table_id list.",
                    score=0.2,
                )

        if request.subtask.layer == "inspect":
            has_signal = bool(request.execution.namespace_updates) or bool(request.execution.output.strip())
            if not has_signal:
                return ReviewResult(
                    accepted=False,
                    feedback="Execution succeeded but produced no observable data or namespace updates.",
                    score=0.2,
                )

        return ReviewResult(accepted=True, feedback="Attempt accepted.", score=1.0)

    def _llm_review(self, request: ReviewRequest) -> ReviewResult:
        prompt = REVIEW_USER_PROMPT_TEMPLATE.format(
            question=request.question,
            answer_language=required_answer_language(request.question),
            subtask_id=request.subtask.id,
            layer=request.subtask.layer,
            subtask_description=request.subtask.description,
            description=request.description,
            code=request.code,
            success=request.execution.success,
            output=request.execution.output or "(no output)",
            error=request.execution.error or "(no error)",
            namespace_updates=", ".join(request.execution.namespace_updates.keys()) or "None",
            current_workspace=_format_workspace(self.env),
            prior_history=_format_prior_history(self.env),
            final_answer_requirement="`final_answer` must be set." if request.require_final_answer else "No final_answer required.",
        )
        self.env.logger.log_event("review_prompt", {"prompt": prompt, "system_prompt": REVIEW_SYSTEM_PROMPT})

        try:
            response = self.llm_client.generate(prompt, system_prompt=REVIEW_SYSTEM_PROMPT)
        except Exception as exc:
            return ReviewResult(
                accepted=False,
                feedback=f"Reviewer LLM failed: {exc}",
                score=0.0,
            )

        content = response.content
        self.env.logger.log_event("review_response", {"content": content})

        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        payload = json_match.group(1) if json_match else content.strip()
        try:
            data = json.loads(payload)
        except Exception as exc:
            return ReviewResult(
                accepted=False,
                feedback=f"Reviewer output must be valid JSON: {exc}",
                score=0.0,
            )
        if not isinstance(data, dict):
            return ReviewResult(
                accepted=False,
                feedback="Reviewer JSON must be an object.",
                score=0.0,
            )

        accepted = bool(data.get("accepted", False))
        try:
            score = float(data.get("score", 1.0 if accepted else 0.0))
        except (TypeError, ValueError):
            score = 1.0 if accepted else 0.0
        score = max(0.0, min(1.0, score))
        feedback = str(data.get("feedback", "")).strip() or ("Accepted." if accepted else "Rejected.")
        return ReviewResult(accepted=accepted, feedback=feedback, score=score)
