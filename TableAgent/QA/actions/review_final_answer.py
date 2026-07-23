from __future__ import annotations

import json
import re
from typing import Any

from TableAgent.QA.actions.base_action import ReviewResult
from TableAgent.prompts.review import (
    FINAL_ANSWER_REVIEW_SYSTEM_PROMPT,
    FINAL_ANSWER_REVIEW_USER_PROMPT_TEMPLATE,
)


class ReviewFinalAnswerAction:
    """Independently verify the final answer against successful runtime evidence."""

    def __init__(self, env: Any, llm_client: Any | None = None):
        self.env = env
        self.llm_client = llm_client

    def run(self, *, question: str, plan: list[Any], outputs: list[Any], final_answer: str) -> ReviewResult:
        if self.llm_client is None:
            return ReviewResult(accepted=True, feedback="Final review skipped without an LLM client.", score=1.0)

        plan_text = json.dumps([
            {
                "id": subtask.id,
                "layer": subtask.layer,
                "category": subtask.category,
                "description": subtask.description,
                "depends_on": list(subtask.depends_on),
            }
            for subtask in plan
        ], ensure_ascii=False, indent=2)
        evidence_sections = []
        for output in outputs:
            if not output.success:
                continue
            observation = str(output.observation or "")
            code = str(output.code or "")
            evidence_sections.append(
                f"## {output.subtask_id} [{output.layer}/{output.category}]\n"
                f"Code:\n```python\n{code[-3000:]}\n```\n"
                f"Observation:\n{observation[:4000]}"
            )
        evidence = "\n\n".join(evidence_sections) or "No successful runtime evidence was produced."
        grouped_headers = self._grouped_header_context()
        prompt = FINAL_ANSWER_REVIEW_USER_PROMPT_TEMPLATE.format(
            question=question,
            plan=plan_text,
            evidence=evidence,
            grouped_headers=grouped_headers,
            final_answer=final_answer,
        )
        self.env.logger.log_event("final_answer_review_prompt", {
            "prompt": prompt,
            "system_prompt": FINAL_ANSWER_REVIEW_SYSTEM_PROMPT,
        })
        try:
            response = self.llm_client.generate(prompt, system_prompt=FINAL_ANSWER_REVIEW_SYSTEM_PROMPT)
        except Exception as exc:
            self.env.logger.log_event("final_answer_review_error", {"error": str(exc)})
            return ReviewResult(accepted=True, feedback=f"Final review unavailable: {exc}", score=0.0)

        self.env.logger.log_event("final_answer_review_response", {"content": response.content})
        json_match = re.search(r"```json\s*(.*?)\s*```", str(response.content), re.DOTALL)
        payload = json_match.group(1) if json_match else str(response.content).strip()
        try:
            data = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ReviewResult(accepted=True, feedback="Final reviewer returned invalid JSON; review skipped.", score=0.0)
        if not isinstance(data, dict):
            return ReviewResult(accepted=True, feedback="Final reviewer returned a non-object; review skipped.", score=0.0)
        if "accepted" not in data:
            return ReviewResult(accepted=True, feedback="Final reviewer omitted `accepted`; review skipped.", score=0.0)
        accepted_value = data.get("accepted")
        if not isinstance(accepted_value, bool):
            return ReviewResult(
                accepted=True,
                feedback="Final reviewer returned a non-boolean `accepted`; review skipped.",
                score=0.0,
            )
        accepted = accepted_value
        try:
            score = max(0.0, min(1.0, float(data.get("score", 1.0 if accepted else 0.0))))
        except (TypeError, ValueError):
            score = 1.0 if accepted else 0.0
        feedback = str(data.get("feedback") or ("Accepted." if accepted else "Rejected.")).strip()
        return ReviewResult(accepted=accepted, feedback=feedback, score=score)

    def _grouped_header_context(self) -> str:
        """Expose verified sibling headers so final review can detect partial group coverage."""
        operators = getattr(self.env, "operators", None)
        if operators is None or not hasattr(operators, "list_headers"):
            return "No grouped headers were available to the final reviewer."
        namespace = getattr(self.env, "execution_namespace", {})
        table_ids = namespace.get("selected_table_ids") if isinstance(namespace, dict) else None
        if isinstance(table_ids, str):
            table_ids = [table_ids]
        if not table_ids:
            try:
                table_ids = operators.list_tables() if hasattr(operators, "list_tables") else []
            except Exception:
                table_ids = []

        lines = []
        for table_id in table_ids or []:
            try:
                headers = operators.list_headers(str(table_id))
            except Exception:
                continue
            for header in headers:
                children = getattr(header, "sub_headers", []) or []
                if not children:
                    continue
                child_text = ", ".join(
                    f"{getattr(child, 'id', '')} ({getattr(child, 'label', '')})"
                    for child in children
                )
                lines.append(
                    f"table={table_id}; parent={getattr(header, 'id', '')} "
                    f"({getattr(header, 'label', '')}); children=[{child_text}]"
                )
        return "\n".join(lines) if lines else "No grouped headers were available to the final reviewer."


__all__ = ["ReviewFinalAnswerAction"]
