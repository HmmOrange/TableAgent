from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from TableAgent.QA.actions.base_action import BasePlanAction, PlanGenerationRequest, PlanGenerationResult
from TableAgent.QA.actions.llm_code_generation import get_structure_summary
from TableAgent.QA.prompts.planner_prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT_TEMPLATE
from TableAgent.schema.subtask import SubTask

PLAN_REPAIR_SYSTEM_PROMPT = """You are a strict JSON formatter for a table-QA plan.
Return only one JSON object with a non-empty `subtasks` list. Each subtask must have
`id`, `description`, `layer` (`inspect` or `synthesis`), and `depends_on` (a list).
Include at least one inspect subtask and a final synthesis subtask. No prose."""


def parse_planner_output(content: str) -> list[SubTask]:
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    payload = json_match.group(1) if json_match else content.strip()
    try:
        data = json.loads(payload)
    except Exception as exc:
        raise ValueError("Planner output must be valid JSON or a ```json code block.") from exc

    items = data.get("subtasks", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Planner JSON must be a list or an object with a 'subtasks' list.")

    subtasks = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Every planner subtask must be a JSON object.")
        subtask_id = str(item.get("id", "")).strip()
        if not subtask_id:
            raise ValueError("Every planner subtask must include a non-empty 'id'.")
        description = str(item.get("description", "")).strip()
        if not description:
            raise ValueError(f"Subtask '{subtask_id}' must include a non-empty 'description'.")
        layer = item.get("layer", "inspect")
        if layer not in ("inspect", "synthesis"):
            raise ValueError(f"Subtask '{subtask_id}' has invalid layer {layer!r}; expected 'inspect' or 'synthesis'.")
        subtasks.append(SubTask(
            id=subtask_id,
            description=description,
            layer=layer,  # type: ignore
            depends_on=_split_depends_on(item.get("depends_on", [])),
            status="pending",
        ))
    if not subtasks:
        raise ValueError("Planner JSON did not contain any subtasks.")
    return subtasks


def _split_depends_on(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


class WriteQAPlanAction(BasePlanAction):
    """Action that writes a two-layer QA plan for one workbook question."""
    name = "write_qa_plan"
    desc = "Generate field-inspection and synthesis subtasks for a table QA question."

    def __init__(self, env: Any, llm_client: Optional[Any] = None):
        self.env = env
        self.llm_client = llm_client

    def run(self, request: PlanGenerationRequest) -> PlanGenerationResult:
        self.env.logger.log_event("planning_start", {
            "action": self.name,
            "question": request.question,
            "table_id": request.table_id,
        })

        if not self.llm_client:
            raise ValueError("WriteQAPlanAction requires an llm_client.")

        struct_summary = get_structure_summary(self.env, request.table_id)
        prompt = PLANNER_USER_PROMPT_TEMPLATE.format(
            question=request.question,
            table_structure=struct_summary,
        )
        self.env.logger.log_event("planner_prompt", {"prompt": prompt, "system_prompt": PLANNER_SYSTEM_PROMPT})
        response = self.llm_client.generate(prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
        raw_response = response.content
        self.env.logger.log_event("planner_response", {"content": raw_response})
        try:
            subtasks = parse_planner_output(raw_response)
        except ValueError as exc:
            repair_prompt = (
                f"Question: {request.question}\nTable id: {request.table_id}\n"
                "Convert the attempted plan below into the required concise JSON.\n\n"
                f"Attempted plan:\n{raw_response[-6000:]}"
            )
            repair_response = self.llm_client.generate(repair_prompt, system_prompt=PLAN_REPAIR_SYSTEM_PROMPT)
            raw_response = repair_response.content
            self.env.logger.log_event("planner_repair_response", {"content": raw_response})
            try:
                subtasks = parse_planner_output(raw_response)
            except ValueError:
                self.env.logger.log_event("planner_error", {
                    "error": str(exc),
                    "raw_response": raw_response,
                })
                raise exc

        for subtask in subtasks:
            if not subtask.metadata:
                subtask.metadata = {}
            subtask.metadata.setdefault("table_id", request.table_id)

        self.env.logger.log_event("planning_complete", {
            "subtasks": [str(s) for s in subtasks],
        })
        return PlanGenerationResult(subtasks=subtasks, raw_response=raw_response)
