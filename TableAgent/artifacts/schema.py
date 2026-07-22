from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from TableAgent.llm import BaseLLM


SUMMARY_SYSTEM_PROMPT = """You summarize spreadsheet structure for a workbook artifact.
Return exactly one JSON object with one non-empty string field: description.
Use the dominant language present in the supplied workbook labels and descriptions.
Be concise, factual, and do not invent information that is not present in the structure.
"""


class SummaryGenerator:
    def __init__(self, llm: BaseLLM, *, repair_retries: int = 1):
        self.llm = llm
        self.repair_retries = max(0, int(repair_retries))

    def sheet_description(self, sheet_name: str, structure_text: str) -> str:
        prompt = (
            f"Sheet name: {sheet_name}\n"
            "Summarize the purpose and contents of this worksheet from its structure YAML.\n"
            f"Structure YAML:\n{structure_text}"
        )
        return self._description(prompt)

    def workbook_description(self, schema_text: str) -> str:
        prompt = (
            "Summarize the workbook as a whole from this schema YAML. Mention the main "
            "worksheets and subject areas without inventing facts.\n"
            f"Schema YAML:\n{schema_text}"
        )
        return self._description(prompt)

    def _description(self, prompt: str) -> str:
        last_content = ""
        for attempt in range(self.repair_retries + 1):
            retry_note = "" if attempt == 0 else (
                "\nPrevious output was invalid. Return only the required JSON object."
                f"\nPrevious output:\n{last_content}"
            )
            response = self.llm.generate(
                prompt + retry_note,
                system_prompt=SUMMARY_SYSTEM_PROMPT,
            )
            last_content = response.content
            try:
                return _parse_description(response.content)
            except ValueError:
                if attempt >= self.repair_retries:
                    raise
        raise ValueError("Summary generation failed")


def build_workbook_schema(
    sheet_structures: Iterable[tuple[str, Path]],
    output_path: Path,
    summarizer: SummaryGenerator,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    used_ids: set[str] = set()
    for sheet_name, structure_path in sheet_structures:
        structure_text = structure_path.read_text(encoding="utf-8")
        structure = yaml.safe_load(structure_text)
        if not isinstance(structure, dict):
            raise ValueError(f"Invalid structure YAML for sheet '{sheet_name}'")
        sheet_id = _sheet_id(sheet_name, used_ids)
        result[sheet_name] = {
            "id": sheet_id,
            "description": summarizer.sheet_description(sheet_name, structure_text),
            "structure": structure,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return result


def _sheet_id(sheet_name: str, used_ids: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z]+", "_", sheet_name).strip("_").lower() or "sheet"
    if base[0].isdigit():
        base = f"sheet_{base}"
    candidate = base
    index = 2
    while candidate in used_ids:
        candidate = f"{base}_{index}"
        index += 1
    used_ids.add(candidate)
    return candidate


def _parse_description(content: str) -> str:
    text = str(content or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    payload = fenced.group(1).strip() if fenced else text
    try:
        value = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Summary response must be a JSON object") from exc
    description = value.get("description") if isinstance(value, dict) else None
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Summary response must contain a non-empty description")
    return description.strip()


__all__ = ["SUMMARY_SYSTEM_PROMPT", "SummaryGenerator", "build_workbook_schema"]
