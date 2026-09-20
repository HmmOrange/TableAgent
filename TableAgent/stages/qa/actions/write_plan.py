from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from TableAgent.stages.qa.actions.base_action import BasePlanAction, PlanGenerationRequest, PlanGenerationResult
from TableAgent.stages.qa.actions.llm_code_generation import get_structure_summary, get_table_catalog_summary
from TableAgent.stages.qa.header_hints import question_header_hints
from TableAgent.stages.qa.group_hints import question_group_hints
from TableAgent.stages.qa.prompts.planner import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT_TEMPLATE
from TableAgent.stages.qa.models.subtask import SubTask
from TableAgent.stages.qa.schemas import (
    PLAN_SCHEMA,
    QAPlan,
    ValidationError,
    generate_json,
    schema_if_enabled,
    validation_message,
)

PLAN_REPAIR_SYSTEM_PROMPT = """You are a strict JSON formatter for a table-QA plan.
Return only one JSON object with a non-empty `subtasks` list. Each subtask must have
`id`, `description`, `layer` (`table_inspect`, `inspect`, or `synthesis`), `category`
(`normal` or `common_info`), and `depends_on` (a list). Non-synthesis common-info
tasks require metadata.common_info_scope as workbook, sheet, or table. No prose."""


def parse_planner_output(content: str) -> list[SubTask]:
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    payload = json_match.group(1) if json_match else content.strip()
    data = None
    try:
        data = json.loads(payload)
    except Exception:
        decoder = json.JSONDecoder()
        candidates = []
        for index, character in enumerate(payload):
            if character not in "[{":
                continue
            try:
                candidate, _ = decoder.raw_decode(payload[index:])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(candidate, list)
                and bool(candidate)
                and all(isinstance(item, dict) and item.get("id") for item in candidate)
            ) or (
                isinstance(candidate, dict) and isinstance(candidate.get("subtasks"), list)
            ):
                candidates.append(candidate)
        if candidates:
            data = candidates[-1]
    if data is None:
        raise ValueError("Planner output must contain a valid JSON plan or a ```json code block.")

    items = data.get("subtasks", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Planner JSON must be a list or an object with a 'subtasks' list.")

    try:
        plan = QAPlan.model_validate({"subtasks": items})
    except ValidationError as exc:
        raise ValueError(f"Planner JSON is not a usable plan -- {validation_message(exc)}") from exc

    return [
        SubTask(
            id=subtask.id,
            description=subtask.description,
            layer=subtask.layer,
            category=subtask.category,
            depends_on=list(subtask.depends_on),
            status="pending",
            metadata=dict(subtask.metadata),
        )
        for subtask in plan.subtasks
    ]


def _related_structure_summary(env: Any) -> str:
    lines = []
    seen = set()
    for source in getattr(env, "related_structures", []):
        structure = source.get("structure") or {}
        table_id = str(source.get("table_id") or structure.get("id") or "")
        sheet_name = str(structure.get("sheet") or "")
        table_name = str(structure.get("name") or table_id)
        key = (sheet_name, table_id, table_name)
        if key in seen:
            continue
        seen.add(key)
        headers = []
        for header in structure.get("headers") or []:
            label = str(getattr(header, "label", "") or getattr(header, "id", ""))
            description = str(getattr(header, "description", ""))
            headers.append(f"{label}: {description}" if description else label)
        lines.append(
            "\n".join([
                f"- table_id: {table_id}",
                f"  name: {table_name}",
                f"  description: {structure.get('description', '')}",
                f"  sheet: {sheet_name}",
                f"  headers: {'; '.join(headers) if headers else '(none)'}",
            ])
        )
    return "\n".join(lines)


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

        table_catalog = get_table_catalog_summary(self.env)
        if request.table_id:
            struct_summary = get_structure_summary(self.env, request.table_id)
        else:
            struct_summary = "\n\n".join(
                get_structure_summary(self.env, table_id)
                for table_id in self.env.operators.list_tables()
            )
        related_summary = _related_structure_summary(self.env)
        if related_summary:
            struct_summary = f"{struct_summary}\n\nRelated prepared-sheet structures:\n{related_summary}"
        prompt = PLANNER_USER_PROMPT_TEMPLATE.format(
            question=request.question,
            workbook_sheets=", ".join(self.env.workbook.sheetnames),
            table_catalog=table_catalog,
            table_structure=struct_summary,
        )
        table_ids = [request.table_id] if request.table_id else self.env.operators.list_tables()
        prompt += (
            "\n\nExact question-to-header matches:\n"
            f"{question_header_hints(self.env, request.question, table_ids)}\n"
            "\nQuestion-to-group matches:\n"
            f"{question_group_hints(self.env, request.question, table_ids)}\n"
            "Treat exact header and group matches as authoritative during inspection and synthesis. "
            "A group is a worksheet section that scopes a block of records: when the question targets one, "
            "plan to read evidence from inside its data_range rather than from the whole table."
        )
        if request.failure_context:
            previous_plan = json.dumps(request.previous_plan or [], ensure_ascii=False, indent=2)
            prompt += (
                "\n\nReplanning context:\n"
                "The previous plan failed during execution. Create a corrected complete plan using only the "
                "runtime evidence below. Do not invent facts or reuse a failed operation unchanged. Preserve any "
                "valid table and field selections, but change the decomposition, dependencies, or inspected fields "
                "when the evidence requires it.\n\n"
                f"Previous plan:\n{previous_plan}\n\n"
                f"Execution failure and reviewer feedback:\n{request.failure_context}"
            )
        self.env.logger.log_event("planner_prompt", {"prompt": prompt, "system_prompt": PLANNER_SYSTEM_PROMPT})
        response = generate_json(
            self.llm_client, prompt, system_prompt=PLANNER_SYSTEM_PROMPT, schema=PLAN_SCHEMA
        )
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
            repair_response = generate_json(
                self.llm_client,
                repair_prompt,
                system_prompt=PLAN_REPAIR_SYSTEM_PROMPT,
                schema=schema_if_enabled(self.env, PLAN_SCHEMA),
            )
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

        subtasks = self._apply_routing_policy(subtasks)
        has_normal_inspection = any(
            subtask.layer == "inspect" and subtask.category == "normal"
            for subtask in subtasks
        )
        needs_table_inspect = len(self.env.operators.list_tables()) > 1 and has_normal_inspection
        if needs_table_inspect and not any(subtask.layer == "table_inspect" for subtask in subtasks):
            subtasks.insert(0, SubTask(
                id="select_relevant_tables",
                description="Select the relevant table_id or table_ids for the question.",
                layer="table_inspect",
                category="normal",
                depends_on=[],
                status="pending",
            ))
            for subtask in subtasks[1:]:
                if subtask.layer == "inspect" and "select_relevant_tables" not in subtask.depends_on:
                    subtask.depends_on.insert(0, "select_relevant_tables")
        elif needs_table_inspect:
            table_inspect_id = next(
                (subtask.id for subtask in subtasks if subtask.layer == "table_inspect"),
                None,
            )
            if table_inspect_id:
                for subtask in subtasks:
                    if subtask.layer == "inspect" and table_inspect_id not in subtask.depends_on:
                        subtask.depends_on.insert(0, table_inspect_id)
        else:
            subtasks = self._drop_redundant_table_inspection(subtasks)

        for subtask in subtasks:
            if not subtask.metadata:
                subtask.metadata = {}
            if request.table_id and subtask.layer != "table_inspect":
                subtask.metadata.setdefault("table_id", request.table_id)

        self.env.logger.log_event("planning_complete", {
            "subtasks": [str(s) for s in subtasks],
        })
        return PlanGenerationResult(subtasks=subtasks, raw_response=raw_response)

    def _drop_redundant_table_inspection(self, subtasks: list[SubTask]) -> list[SubTask]:
        """Remove a table-selection step when there is only one table to select.

        The prompt's worked example opens with `select_relevant_tables`, so the planner
        writes one almost every time -- on this benchmark, in 98% of runs against
        single-table workbooks, where the runner has already put the only table in
        `selected_table_ids` before planning starts. The step carries no information and
        still costs a generation, an execution and a review, and it is the *first* step,
        so anything that wobbles there is inherited by everything after it. Dropping it
        removes a fork rather than trying to make the fork behave.
        """
        table_ids = self.env.operators.list_tables()
        if len(table_ids) != 1:
            return subtasks
        redundant = {
            subtask.id for subtask in subtasks if subtask.layer == "table_inspect"
        }
        if not redundant:
            return subtasks
        kept = [subtask for subtask in subtasks if subtask.id not in redundant]
        if not kept:
            # A plan that is nothing but table selection still has to run: answering with
            # no inspection at all would be worse than one redundant step.
            return subtasks
        for subtask in kept:
            subtask.depends_on = [
                dependency
                for dependency in subtask.depends_on
                if dependency not in redundant
            ]
        self.env.logger.log_event(
            "table_inspection_pruned",
            {"removed": sorted(redundant), "table_id": table_ids[0]},
        )
        return kept

    def _apply_routing_policy(self, subtasks: list[SubTask]) -> list[SubTask]:
        mode = str(getattr(self.env, "qa_routing_mode", "auto"))
        enabled = bool(getattr(self.env, "qa_common_info_enabled", True))
        if mode == "normal" or not enabled:
            for subtask in subtasks:
                subtask.category = "normal"
        elif mode == "common_info":
            for subtask in subtasks:
                subtask.category = "common_info"
                if subtask.layer != "synthesis":
                    metadata = subtask.metadata or {}
                    metadata.setdefault("common_info_scope", "workbook")
                    subtask.metadata = metadata

        by_id = {subtask.id: subtask for subtask in subtasks}
        for subtask in subtasks:
            if subtask.layer != "synthesis":
                continue
            categories = {
                by_id[dependency].category
                for dependency in subtask.depends_on
                if dependency in by_id
            }
            if categories:
                subtask.category = "normal" if "normal" in categories else "common_info"
        return subtasks
