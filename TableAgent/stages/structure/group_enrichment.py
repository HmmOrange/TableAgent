from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from TableAgent.llm import LLMResponse
from TableAgent.rendering.workbook import WorkbookRenderer
from TableAgent.stages.structure.layout.parsing import _normalize_layout_group


GROUP_SYSTEM_PROMPT = (
    "You inspect a complete spreadsheet image after its tables and headers have "
    "already been extracted. Return YAML only. Identify semantic groups without "
    "changing or repeating the completed table structure."
)

GROUP_USER_PROMPT_TEMPLATE = """Inspect the complete worksheet image and the completed table below.

Workbook: {workbook_name}
Worksheet: {sheet_name}
Complete used range: {viewport_range}
Table: {table_key}

Completed table from structure.yaml (immutable context):
{table_text}

A group is a visibly labelled, contiguous worksheet region whose label supplies
shared context to multiple records or data cells. A header says what a field
measures, an individual record label says what one row represents, and a group
label says which shared section owns a block of related records. Repeated record
sequences beneath peer section labels are strong group evidence.

Titles, table headers, parent header bands, individual records, sources, notes,
and footnotes are not groups. A total is a group only when it introduces and owns
a section; an isolated total record is not automatically a group.

For each group, use exactly these six fields:
- id: unique stable snake_case identifier within this table
- label: exact visible worksheet text
- group_range: exact cell or merged range containing the visible group label only
- data_range: calculable cells owned by the group, excluding the label and other contextual cells
- axis: row, column, or region
- description: short semantic explanation

The label must apply beyond its own cell. `group_range` is only the label's visible
cell or merged span; do not expand it to include the owned data block. Groups may
overlap or have different shapes.
Do not invent members or header references.

Return only:

groups:
  - id: <group_id>
    label: "<exact visible label>"
    group_range: <exact A1 range>
    data_range: <exact A1 range>
    axis: <row|column|region>
    description: "<semantic role>"

Return `groups: []` when this table has no qualifying groups.
Do not return the table key, table metadata, headers, structure.yaml, explanations, or markdown.
"""

GROUP_REPAIR_PROMPT_TEMPLATE = """{prompt}

Your previous response was invalid.
Validation errors:
{errors}

Previous response:
{response}

Return corrected YAML only: one top-level `groups` key containing a list. Do not
return the table key, table metadata, headers, structure.yaml, explanations, or markdown.
"""

_FENCE = re.compile(r"^```(?:yaml)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class GroupEnrichmentOutput:
    structure_text: str
    preflight_errors: list[str]
    image_path: Path
    responses: list[LLMResponse]


class GroupEnrichmentStage:
    def __init__(self, renderer: WorkbookRenderer, vlm: Any):
        self.renderer = renderer
        self.vlm = vlm

    def run(
        self,
        *,
        workbook_path: Path,
        sheet_name: str,
        viewport_range: str,
        structure_text: str,
        artifact_dir: Path,
    ) -> GroupEnrichmentOutput:
        try:
            structure = yaml.safe_load(structure_text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Completed structure must be valid YAML: {exc}") from exc
        if not isinstance(structure, dict):
            raise ValueError("Completed structure must be a YAML mapping")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        image_path = artifact_dir / "worksheet.png"
        self.renderer.source_viewport_to_image(
            workbook_path,
            sheet_name,
            viewport_range,
            image_path,
        )

        all_errors: list[str] = []
        responses: list[LLMResponse] = []
        used_artifact_names: set[str] = set()
        for table_key, table in structure.items():
            if not isinstance(table, dict):
                continue
            safe_key = _safe_artifact_name(str(table_key), used_artifact_names)
            table_dir = artifact_dir / safe_key
            table_dir.mkdir(parents=True, exist_ok=True)
            prompt = GROUP_USER_PROMPT_TEMPLATE.format(
                workbook_name=workbook_path.name,
                sheet_name=sheet_name,
                viewport_range=viewport_range,
                table_key=table_key,
                table_text=yaml.safe_dump(table, sort_keys=False, allow_unicode=True).strip(),
            )
            (table_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            response = self.vlm.generate_with_image(
                prompt=prompt,
                image_path=image_path,
                system_prompt=GROUP_SYSTEM_PROMPT,
            )
            responses.append(response)
            (table_dir / "response.txt").write_text(response.content, encoding="utf-8")

            try:
                groups, errors = parse_group_enrichment(response.content, str(table_key))
            except ValueError as exc:
                groups, errors = [], [str(exc)]
            if errors:
                repair_prompt = GROUP_REPAIR_PROMPT_TEMPLATE.format(
                    prompt=prompt,
                    errors="\n".join(f"- {error}" for error in errors),
                    response=response.content,
                )
                (table_dir / "repair_prompt.txt").write_text(repair_prompt, encoding="utf-8")
                response = self.vlm.generate_with_image(
                    prompt=repair_prompt,
                    image_path=image_path,
                    system_prompt=GROUP_SYSTEM_PROMPT,
                )
                responses.append(response)
                (table_dir / "repair_response.txt").write_text(response.content, encoding="utf-8")
                try:
                    groups, errors = parse_group_enrichment(response.content, str(table_key))
                except ValueError as exc:
                    groups, errors = [], [str(exc)]

            table.pop("groups", None)
            table["groups"] = groups
            all_errors.extend(errors)

        enriched = yaml.safe_dump(structure, sort_keys=False, allow_unicode=True).strip()
        (artifact_dir / "structure_after.yaml").write_text(enriched, encoding="utf-8")
        return GroupEnrichmentOutput(enriched, all_errors, image_path, responses)


def parse_group_enrichment(content: str, table_key: str) -> tuple[list[dict[str, Any]], list[str]]:
    response_text = str(content).strip()
    fenced = _FENCE.match(response_text)
    if fenced:
        response_text = fenced.group(1).strip()
    try:
        payload = yaml.safe_load(response_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Group response must be valid YAML: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"groups"}:
        raise ValueError("Group response must contain exactly one top-level `groups` key")
    raw_groups = payload["groups"]
    if not isinstance(raw_groups, list):
        raise ValueError("Group response `groups` value must be a list")

    errors: list[str] = []
    normalized_groups: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, group in enumerate(raw_groups):
        normalized = _normalize_layout_group(
            group,
            used_ids=used_ids,
            path=f"{table_key}.groups[{index}]",
            preflight_errors=errors,
        )
        if normalized is not None:
            normalized_groups.append(normalized)
    return normalized_groups, errors


def _safe_artifact_name(table_key: str, used_names: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", table_key).strip("._") or "table"
    candidate = base
    suffix = 2
    while candidate.casefold() in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate.casefold())
    return candidate
