from __future__ import annotations

import json
import re
from typing import Any, List, Optional

from TableAgent.QA.actions.base_action import BasePlanAction, PlanGenerationRequest, PlanGenerationResult
from TableAgent.QA.actions.llm_code_generation import get_structure_summary, get_table_catalog_summary
from TableAgent.prompts.planner import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT_TEMPLATE
from TableAgent.schema.subtask import SubTask

PLAN_REPAIR_SYSTEM_PROMPT = """You are a strict JSON formatter for a table-QA plan.
Return only one JSON object with a non-empty `subtasks` list. Each subtask must have
    `id`, `description`, `layer` (`table_inspect`, `inspect`, or `synthesis`), `category` (`normal` or `common_info`),
    and `depends_on` (a list). Every non-synthesis `common_info` subtask must include
    metadata.common_info_scope as `workbook`, `sheet`, or `table`.
    Include one table_inspect subtask for normal multi-table plans, at least one inspect subtask, and a final synthesis
    subtask. Common-info workbook plans may omit table_inspect. For mixed common-info and normal plans, the final
    synthesis must have category `normal` and depend on both branches. Do not create overlapping sheet/table
    common-info subtasks for the same target unless explicitly requested. No prose."""

PLAN_CATEGORY_REVIEW_SYSTEM_PROMPT = """You are an independent spreadsheet-QA plan routing reviewer.
Return a corrected complete plan as JSON using the same schema as the input plan.

Routing rules:
- `common_info` is only for describing the workbook, a sheet, or a table itself: identity, purpose, organization, and
  verified top-level headers.
- Questions about specific records, named items, defects, incidents, products, values, filters, lists of data values, or
  calculations require `normal` inspection even if they use wording such as general information.
- A pure common-information plan must end in `common_info` synthesis.
- A final `normal` synthesis must depend on at least one normal data-inspection branch. Mixed questions may combine both.

Do not answer the question, use a golden answer, or add question-specific facts. Return JSON only."""


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
        if layer not in ("table_inspect", "inspect", "synthesis"):
            raise ValueError(
                f"Subtask '{subtask_id}' has invalid layer {layer!r}; expected 'table_inspect', 'inspect', or 'synthesis'."
            )
        category = str(item.get("category", item.get("kind", "normal"))).strip().lower()
        if category not in {"normal", "common_info"}:
            raise ValueError(
                f"Subtask '{subtask_id}' has invalid category {category!r}; expected 'normal' or 'common_info'."
            )
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        metadata = dict(metadata)
        if item.get("common_info_scope") and "common_info_scope" not in metadata:
            metadata["common_info_scope"] = item["common_info_scope"]
        if item.get("target_names") and "target_names" not in metadata:
            metadata["target_names"] = item["target_names"]
        if category == "common_info" and layer != "synthesis":
            scope = str(metadata.get("common_info_scope") or "").strip().lower()
            if scope not in {"workbook", "sheet", "table"}:
                raise ValueError(
                    f"Common-info subtask '{subtask_id}' must include metadata.common_info_scope "
                    "as 'workbook', 'sheet', or 'table'."
                )
            metadata["common_info_scope"] = scope
        subtasks.append(SubTask(
            id=subtask_id,
            description=description,
            layer=layer,  # type: ignore
            category=category,  # type: ignore
            depends_on=_split_depends_on(item.get("depends_on", [])),
            status="pending",
            metadata=metadata,
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


def _synthesis_dependency_categories(subtasks: list[SubTask], synthesis: SubTask) -> set[str]:
    by_id = {subtask.id: subtask for subtask in subtasks}
    categories = set()
    pending = list(synthesis.depends_on)
    seen = set()
    while pending:
        dependency_id = pending.pop()
        if dependency_id in seen or dependency_id not in by_id:
            continue
        seen.add(dependency_id)
        dependency = by_id[dependency_id]
        if dependency.layer != "synthesis":
            categories.add(dependency.category)
        pending.extend(dependency.depends_on)
    return categories


def _needs_category_review(subtasks: list[SubTask]) -> bool:

    for subtask in subtasks:
        if subtask.layer != "synthesis":
            continue
        categories = _synthesis_dependency_categories(subtasks, subtask)
        if subtask.category == "normal" and categories == {"common_info"}:
            return True
        if subtask.category == "common_info" and "normal" in categories:
            return True
    return False


def _repair_synthesis_categories(subtasks: list[SubTask]) -> None:
    for subtask in subtasks:
        if subtask.layer != "synthesis":
            continue
        categories = _synthesis_dependency_categories(subtasks, subtask)
        if "normal" in categories:
            subtask.category = "normal"
        elif categories == {"common_info"}:
            subtask.category = "common_info"


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
        if request.failure_context:
            previous_plan = json.dumps(request.previous_plan or [], ensure_ascii=False, indent=2)
            prompt += (
                "\n\nReplanning context:\n"
                "The previous plan failed during execution. Create a corrected complete plan using only the "
                "runtime evidence below. Do not invent facts or reuse a failed operation unchanged. Preserve any "
                "valid structural-vs-data classification decisions, but change the decomposition, dependencies, "
                "or inspected fields when the evidence requires it.\n\n"
                f"Previous plan:\n{previous_plan}\n\n"
                f"Execution failure and reviewer feedback:\n{request.failure_context}"
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

        if getattr(self.env, "enable_plan_category_review", False) or _needs_category_review(subtasks):
            category_review_prompt = (
                f"Question:\n{request.question}\n\n"
                "Review and correct the routing categories and dependencies in this plan:\n"
                f"{json.dumps({'subtasks': [self._subtask_payload(item) for item in subtasks]}, ensure_ascii=False, indent=2)}"
            )
            self.env.logger.log_event("planner_category_review_prompt", {
                "prompt": category_review_prompt,
                "system_prompt": PLAN_CATEGORY_REVIEW_SYSTEM_PROMPT,
            })
            category_response = self.llm_client.generate(
                category_review_prompt,
                system_prompt=PLAN_CATEGORY_REVIEW_SYSTEM_PROMPT,
            )
            self.env.logger.log_event("planner_category_review_response", {"content": category_response.content})
            try:
                subtasks = parse_planner_output(category_response.content)
            except ValueError as exc:
                self.env.logger.log_event("planner_category_review_error", {"error": str(exc)})
            _repair_synthesis_categories(subtasks)

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

        for subtask in subtasks:
            if not subtask.metadata:
                subtask.metadata = {}
            if request.table_id and subtask.layer != "table_inspect":
                subtask.metadata.setdefault("table_id", request.table_id)

        self.env.logger.log_event("planning_complete", {
            "subtasks": [str(s) for s in subtasks],
        })
        return PlanGenerationResult(subtasks=subtasks, raw_response=raw_response)

    @staticmethod
    def _subtask_payload(subtask: SubTask) -> dict[str, Any]:
        return {
            "id": subtask.id,
            "description": subtask.description,
            "layer": subtask.layer,
            "category": subtask.category,
            "depends_on": list(subtask.depends_on),
            "metadata": dict(subtask.metadata or {}),
        }
