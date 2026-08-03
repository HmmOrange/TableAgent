from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from TableAgent.prompts.common_info import (
    COMMON_INFO_LOCALIZATION_PROMPT,
    COMMON_INFO_LOCALIZATION_SYSTEM_PROMPT,
)
from TableAgent.schema.header import Header
from TableAgent.schema.qa import AgentOutput
from TableAgent.schema.subtask import SubTask


@dataclass(frozen=True)
class CommonInfoRecord:
    scope: str
    name: str
    description: str
    sheet_name: str
    table_names: tuple[str, ...]
    headers: tuple[tuple[str, str], ...]


class CommonInfoSubtaskAction:
    """Answer workbook, sheet, and table structure questions from verified metadata."""

    def __init__(self, env: Any, llm_client: Any | None = None):
        self.env = env
        self.llm_client = llm_client

    def run(self, question: str, subtask: SubTask) -> AgentOutput:
        if subtask.layer == "synthesis":
            verified_answer = self._synthesize(subtask)
            answer = self._localize(question, verified_answer)
            self.env.execution_namespace["final_answer"] = answer
            updates = {"final_answer": answer}
        else:
            records = self._records(question, subtask)
            answer = self._render(records)
            outputs = self.env.execution_namespace.setdefault("common_info_outputs", {})
            records_by_task = self.env.execution_namespace.setdefault("common_info_records", {})
            outputs[subtask.id] = answer
            records_by_task[subtask.id] = records
            updates = {
                "common_info_outputs": outputs,
                "common_info_records": records_by_task,
            }

        code = f"final_answer = {answer!r}" if subtask.layer == "synthesis" else ""
        subtask.status = "success"
        subtask.assigned_agent = self.__class__.__name__
        subtask.code_attempt = code
        subtask.observation = answer
        return AgentOutput(
            subtask_id=subtask.id,
            description="Extract verified workbook structure information.",
            code=code,
            success=True,
            observation=answer,
            reasoning="Used verified structure metadata without inferring business data.",
            namespace_updates=updates,
            category="common_info",
        )

    def _localize(self, question: str, answer: str) -> str:
        if self.llm_client is None:
            return answer
        response = self.llm_client.generate(
            COMMON_INFO_LOCALIZATION_PROMPT.format(question=question, answer=answer),
            system_prompt=COMMON_INFO_LOCALIZATION_SYSTEM_PROMPT,
        )
        localized = str(getattr(response, "content", "") or "").strip()
        if not localized or localized.upper().startswith("ERROR:"):
            raise RuntimeError("Common-information localization failed")
        return localized

    def _synthesize(self, subtask: SubTask) -> str:
        records_by_task = self.env.execution_namespace.get("common_info_records", {})
        records = []
        if isinstance(records_by_task, dict):
            for dependency in subtask.depends_on:
                value = records_by_task.get(dependency)
                if isinstance(value, list):
                    records.extend(value)
        if records:
            return self._render(list(dict.fromkeys(records)))

        outputs = self.env.execution_namespace.get("common_info_outputs", {})
        if isinstance(outputs, dict):
            answers = [
                str(outputs[dependency]).strip()
                for dependency in subtask.depends_on
                if dependency in outputs and str(outputs[dependency]).strip()
            ]
            if answers:
                return "\n\n".join(dict.fromkeys(answers))
        return "No verified structural information matched the requested scope."

    def _records(self, question: str, subtask: SubTask) -> list[CommonInfoRecord]:
        metadata = subtask.metadata if isinstance(subtask.metadata, dict) else {}
        scope = str(metadata.get("common_info_scope") or "").strip().lower()
        if scope not in {"workbook", "sheet", "table"}:
            raise ValueError(
                f"Common-info subtask '{subtask.id}' requires "
                "metadata.common_info_scope as workbook, sheet, or table"
            )
        targets = self._target_names(metadata.get("target_names"))
        sources = self._structure_sources()
        if scope == "table":
            return self._table_records(sources, targets, question, subtask.description)
        return self._sheet_records(
            sources,
            targets if scope == "sheet" else [],
            question,
            subtask.description,
            select_all=scope == "workbook",
        )

    def _structure_sources(self) -> list[dict[str, Any]]:
        sources = [
            {"table_id": table_id, "structure": structure}
            for table_id, structure in self.env.structures.items()
        ]
        sources.extend(getattr(self.env, "related_structures", []))
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for source in sources:
            structure = source.get("structure") or {}
            key = (
                str(structure.get("sheet") or ""),
                str(source.get("table_id") or structure.get("id") or ""),
            )
            unique.setdefault(key, source)
        return list(unique.values())

    def _table_records(
        self,
        sources: list[dict[str, Any]],
        targets: list[str],
        question: str,
        description: str,
    ) -> list[CommonInfoRecord]:
        selected = self._match_sources(sources, targets, f"{question} {description}")
        if not selected:
            selected_ids = self.env.execution_namespace.get("selected_table_ids", [])
            selected = [source for source in sources if source.get("table_id") in selected_ids]
        if not selected:
            selected = sources
        return [self._table_record(source) for source in selected]

    def _sheet_records(
        self,
        sources: list[dict[str, Any]],
        targets: list[str],
        question: str,
        description: str,
        *,
        select_all: bool,
    ) -> list[CommonInfoRecord]:
        grouped: dict[str, list[dict[str, Any]]] = {
            str(name): [] for name in self.env.workbook.sheetnames
        }
        for source in sources:
            sheet = str((source.get("structure") or {}).get("sheet") or "")
            if sheet:
                grouped.setdefault(sheet, []).append(source)
        selected_names = list(grouped) if select_all else self._match_names(
            list(grouped), targets, f"{question} {description}"
        )
        if not selected_names:
            selected_names = list(grouped)
        return [self._sheet_record(name, grouped[name]) for name in selected_names]

    @classmethod
    def _table_record(cls, source: dict[str, Any]) -> CommonInfoRecord:
        structure = source.get("structure") or {}
        name = str(structure.get("name") or source.get("table_id") or "Unknown table")
        return CommonInfoRecord(
            scope="table",
            name=name,
            description=str(structure.get("description") or ""),
            sheet_name=str(structure.get("sheet") or ""),
            table_names=(name,),
            headers=tuple(cls._header_pairs(structure.get("headers") or [])),
        )

    @classmethod
    def _sheet_record(cls, sheet_name: str, sources: list[dict[str, Any]]) -> CommonInfoRecord:
        descriptions: list[str] = []
        table_names: list[str] = []
        headers: list[tuple[str, str]] = []
        for source in sources:
            structure = source.get("structure") or {}
            name = str(structure.get("name") or source.get("table_id") or "")
            description = str(structure.get("description") or "")
            if name and name not in table_names:
                table_names.append(name)
            if description and description not in descriptions:
                descriptions.append(description)
            for pair in cls._header_pairs(structure.get("headers") or []):
                if pair not in headers:
                    headers.append(pair)
        return CommonInfoRecord(
            scope="sheet",
            name=sheet_name,
            description=" ".join(descriptions),
            sheet_name=sheet_name,
            table_names=tuple(table_names),
            headers=tuple(headers),
        )

    @staticmethod
    def _header_pairs(headers: list[Header]) -> list[tuple[str, str]]:
        return list(dict.fromkeys(
            (str(header.label or header.id), str(header.description or ""))
            for header in headers
        ))

    @staticmethod
    def _render(records: list[CommonInfoRecord]) -> str:
        if not records:
            return "No verified structural information matched the requested scope."
        blocks = []
        for record in records:
            title = "Sheet" if record.scope == "sheet" else "Table"
            lines = [f"## {title}: {_escape(record.name)}"]
            if record.scope == "table" and record.sheet_name:
                lines.append(f"- Sheet: {_escape(record.sheet_name)}")
            lines.append(f"- Description: {_escape(record.description or 'No verified description.')}" )
            if record.scope == "sheet" and record.table_names:
                lines.append("- Tables: " + ", ".join(map(_escape, record.table_names)))
            lines.extend(["", "| Header | Description |", "| --- | --- |"])
            lines.extend(
                f"| {_escape(label)} | {_escape(description)} |"
                for label, description in record.headers
            )
            if not record.headers:
                lines.append("| (No verified headers) | |")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @classmethod
    def _match_sources(
        cls,
        sources: list[dict[str, Any]],
        targets: list[str],
        text: str,
    ) -> list[dict[str, Any]]:
        haystack = _normalize(text)
        normalized_targets = [_normalize(target) for target in targets]
        matched = []
        for source in sources:
            structure = source.get("structure") or {}
            values = [_normalize(source.get("table_id")), _normalize(structure.get("name"))]
            if normalized_targets and any(target == value or target in value for target in normalized_targets for value in values):
                matched.append(source)
            elif not normalized_targets and any(value and value in haystack for value in values):
                matched.append(source)
        return matched

    @staticmethod
    def _match_names(names: list[str], targets: list[str], text: str) -> list[str]:
        haystack = _normalize(text)
        normalized_targets = [_normalize(target) for target in targets]
        return [
            name
            for name in names
            if (
                any(target == _normalize(name) or target in _normalize(name) for target in normalized_targets)
                if normalized_targets
                else _normalize(name) in haystack
            )
        ]

    @staticmethod
    def _target_names(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def _escape(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().replace("|", "\\|")
