from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any

from TableAgent.schema.header import Header
from TableAgent.schema.qa import AgentOutput
from TableAgent.schema.subtask import SubTask
from TableAgent.prompts.common_info import (
    COMMON_INFO_LANGUAGE_SYSTEM_PROMPT,
    COMMON_INFO_LANGUAGE_USER_PROMPT_TEMPLATE,
)
from TableAgent.QA.language import answer_uses_required_language, required_answer_language


@dataclass(frozen=True)
class CommonInfoRecord:
    scope: str
    name: str
    description: str
    sheet_name: str
    table_names: tuple[str, ...]
    headers: tuple[tuple[str, str], ...]


class CommonInfoSubtaskAction:
    """Answer structural/common-information subtasks from verified metadata only."""

    def __init__(self, env: Any, llm_client: Any | None = None):
        self.env = env
        self.llm_client = llm_client

    def run(self, question: str, subtask: SubTask) -> AgentOutput:
        if subtask.layer == "synthesis":
            verified_answer = self._synthesize(subtask)
            answer = self._localize_answer(question, verified_answer)
            self.env.execution_namespace["final_answer"] = answer
            updates = {"final_answer": answer}
            code = f"final_answer = {answer!r}"
            description = "Return the verified common-information summary."
        else:
            records = self._records(question, subtask)
            answer = self._render(records)
            outputs = self.env.execution_namespace.setdefault("common_info_outputs", {})
            outputs[subtask.id] = answer
            records_by_subtask = self.env.execution_namespace.setdefault("common_info_records", {})
            records_by_subtask[subtask.id] = records
            updates = {
                "common_info_outputs": outputs,
                "common_info_records": records_by_subtask,
            }
            code = f"common_info_outputs[{subtask.id!r}] = {answer!r}"
            description = "Extract verified sheet/table descriptions and headers."

        subtask.status = "success"
        subtask.assigned_agent = self.__class__.__name__
        subtask.code_attempt = code
        subtask.observation = answer
        return AgentOutput(
            subtask_id=subtask.id,
            description=description,
            code=code,
            success=True,
            observation=answer,
            reasoning="Used verified structure metadata; no business-data inference was performed.",
            namespace_updates=updates,
        )

    def _localize_answer(self, question: str, answer: str) -> str:
        if self.llm_client is None:
            raise RuntimeError("Common-info answer localization requires an LLM client.")
        answer_language = required_answer_language(question)
        base_prompt = COMMON_INFO_LANGUAGE_USER_PROMPT_TEMPLATE.format(
            question=question,
            answer_language=answer_language,
            answer=answer,
        )
        prompt = base_prompt
        for attempt in range(2):
            response = self.llm_client.generate(
                prompt,
                system_prompt=COMMON_INFO_LANGUAGE_SYSTEM_PROMPT,
            )
            localized = str(getattr(response, "content", "") or "").strip()
            if (
                localized
                and not localized.startswith("ERROR:")
                and answer_uses_required_language(localized, answer_language)
            ):
                return localized
            prompt = (
                f"{base_prompt}\n\n"
                f"The previous response failed language validation. Every translatable prose segment must be in "
                f"{answer_language}. Return a fully translated answer, not the original text.\n\n"
                f"Invalid previous response:\n{localized}"
            )
        raise RuntimeError(
            f"The LLM did not return a common-info answer in the required language: {answer_language}."
        )

    def _synthesize(self, subtask: SubTask) -> str:
        outputs = self.env.execution_namespace.get("common_info_outputs", {})
        if not isinstance(outputs, dict):
            outputs = {}
        records_by_subtask = self.env.execution_namespace.get("common_info_records", {})
        if isinstance(records_by_subtask, dict):
            dependency_records = []
            for dependency in subtask.depends_on:
                records = records_by_subtask.get(dependency, [])
                if isinstance(records, list):
                    dependency_records.extend(records)
            if dependency_records:
                return self._render(self._deduplicate_records(dependency_records))
        dependency_answers = [
            str(outputs[dependency]).strip()
            for dependency in subtask.depends_on
            if dependency in outputs and str(outputs[dependency]).strip()
        ]
        if not dependency_answers:
            dependency_answers = [str(value).strip() for value in outputs.values() if str(value).strip()]
        return "\n\n".join(dict.fromkeys(dependency_answers)) or (
            "No verified common information was produced by the inspection subtasks."
        )

    @classmethod
    def _deduplicate_records(cls, records: list[CommonInfoRecord]) -> list[CommonInfoRecord]:
        unique = []
        for record in records:
            if record not in unique:
                unique.append(record)

        sheet_records = [record for record in unique if record.scope == "sheet"]
        deduplicated = []
        for record in unique:
            if record.scope == "table" and any(
                cls._sheet_covers_table(sheet_record, record)
                for sheet_record in sheet_records
            ):
                continue
            deduplicated.append(record)
        return deduplicated

    @classmethod
    def _sheet_covers_table(cls, sheet_record: CommonInfoRecord, table_record: CommonInfoRecord) -> bool:
        if cls._normalize(sheet_record.name) != cls._normalize(table_record.sheet_name):
            return False
        table_name = cls._normalize(table_record.name)
        table_names = {cls._normalize(name) for name in sheet_record.table_names}
        if not table_name or table_name not in table_names:
            return False
        return set(table_record.headers).issubset(set(sheet_record.headers))

    def _records(self, question: str, subtask: SubTask) -> list[CommonInfoRecord]:
        metadata = subtask.metadata if isinstance(subtask.metadata, dict) else {}
        scope = str(metadata.get("common_info_scope") or "").strip().lower()
        if scope not in {"workbook", "sheet", "table"}:
            raise ValueError(
                f"Common-info subtask '{subtask.id}' requires the LLM to provide "
                "metadata.common_info_scope as 'workbook', 'sheet', or 'table'."
            )
        target_names = self._target_names(metadata.get("target_names"))
        sources = self._structure_sources()
        if scope == "table":
            return self._table_records(sources, target_names, question, subtask.description)
        return self._sheet_records(
            sources,
            target_names if scope == "sheet" else [],
            question,
            subtask.description,
            select_all=scope == "workbook",
        )

    def _structure_sources(self) -> list[dict[str, Any]]:
        sources = [
            {
                "structure_path": str(self.env.structure_path),
                "table_id": table_id,
                "structure": structure,
            }
            for table_id, structure in self.env.structures.items()
        ]
        sources.extend(getattr(self.env, "related_structures", []))
        deduplicated = []
        seen = set()
        for source in sources:
            structure = source.get("structure") or {}
            key = (
                str(structure.get("sheet") or ""),
                str(source.get("table_id") or structure.get("id") or ""),
                str(structure.get("name") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(source)
        return deduplicated

    def _table_records(
        self,
        sources: list[dict[str, Any]],
        target_names: list[str],
        question: str,
        description: str,
    ) -> list[CommonInfoRecord]:
        selected = self._match_sources(sources, target_names, question, description, fields=("table_id", "name"))
        if not selected:
            selected_ids = self.env.execution_namespace.get("selected_table_ids", [])
            if isinstance(selected_ids, str):
                selected_ids = [selected_ids]
            selected = [source for source in sources if source.get("table_id") in selected_ids]
        if not selected:
            selected = sources
        records = []
        for source in selected:
            structure = source.get("structure") or {}
            headers = self._header_pairs(structure.get("headers") or [])
            records.append(CommonInfoRecord(
                scope="table",
                name=str(structure.get("name") or source.get("table_id") or "Unknown table"),
                description=str(structure.get("description") or ""),
                sheet_name=str(structure.get("sheet") or ""),
                table_names=(str(structure.get("name") or source.get("table_id") or ""),),
                headers=tuple(headers),
            ))
        return records

    def _sheet_records(
        self,
        sources: list[dict[str, Any]],
        target_names: list[str],
        question: str,
        description: str,
        *,
        select_all: bool = False,
    ) -> list[CommonInfoRecord]:
        excluded_sheet_names = getattr(self.env, "excluded_sheet_names", set())
        grouped: dict[str, list[dict[str, Any]]] = {
            str(sheet_name): []
            for sheet_name in self.env.workbook.sheetnames
            if str(sheet_name).strip().casefold() not in excluded_sheet_names
        }
        for source in sources:
            structure = source.get("structure") or {}
            sheet_name = str(structure.get("sheet") or "")
            if sheet_name and sheet_name.strip().casefold() not in excluded_sheet_names:
                grouped.setdefault(sheet_name, []).append(source)

        selected_names = [] if select_all else self._match_names(list(grouped), target_names, question, description)
        if not selected_names:
            selected_names = list(grouped)
        records = []
        for sheet_name in selected_names:
            sheet_sources = grouped.get(sheet_name, [])
            descriptions = []
            table_names = []
            headers: list[tuple[str, str]] = []
            for source in sheet_sources:
                structure = source.get("structure") or {}
                table_name = str(structure.get("name") or source.get("table_id") or "")
                table_description = str(structure.get("description") or "")
                if table_name and table_name not in table_names:
                    table_names.append(table_name)
                if table_description and table_description not in descriptions:
                    descriptions.append(table_description)
                for pair in self._header_pairs(structure.get("headers") or []):
                    if pair not in headers:
                        headers.append(pair)
            records.append(CommonInfoRecord(
                scope="sheet",
                name=sheet_name,
                description=" ".join(descriptions),
                sheet_name=sheet_name,
                table_names=tuple(table_names),
                headers=tuple(headers),
            ))
        return records

    @staticmethod
    def _header_pairs(headers: list[Header]) -> list[tuple[str, str]]:
        pairs = []
        for header in headers:
            pair = (str(header.label or header.id), str(header.description or ""))
            if pair not in pairs:
                pairs.append(pair)
        return pairs

    @staticmethod
    def _render(records: list[CommonInfoRecord]) -> str:
        if not records:
            return "No verified common information matched the requested scope."
        blocks = []
        for record in records:
            title = "Sheet" if record.scope == "sheet" else "Table"
            lines = [f"## {title}: {CommonInfoSubtaskAction._escape(record.name)}"]
            if record.sheet_name and record.scope == "table":
                lines.append(
                    f"- Sheet: {CommonInfoSubtaskAction._escape(record.sheet_name)}"
                )
            description = record.description or "No verified description."
            lines.append(
                f"- Description: {CommonInfoSubtaskAction._escape(description)}"
            )
            if record.table_names and record.scope == "sheet":
                table_names = ", ".join(
                    CommonInfoSubtaskAction._escape(name) for name in record.table_names
                )
                lines.append(f"- Tables: {table_names}")
            lines.extend(["", "| Header | Description |", "| --- | --- |"])
            if record.headers:
                lines.extend(
                    f"| {CommonInfoSubtaskAction._escape(label)} | {CommonInfoSubtaskAction._escape(description)} |"
                    for label, description in record.headers
                )
            else:
                lines.append("| (No verified headers) | |")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _target_names(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @classmethod
    def _match_sources(
        cls,
        sources: list[dict[str, Any]],
        target_names: list[str],
        question: str,
        description: str,
        *,
        fields: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        haystack = cls._normalize(f"{question} {description}")
        targets = [cls._normalize(target) for target in target_names]
        matched = []
        for source in sources:
            structure = source.get("structure") or {}
            values = [source.get(field) if field in source else structure.get(field) for field in fields]
            normalized_values = [cls._normalize(value) for value in values if value]
            if targets and any(target == value or target in value for target in targets for value in normalized_values):
                matched.append(source)
            elif not targets and any(value and value in haystack for value in normalized_values):
                matched.append(source)
        return matched

    @classmethod
    def _match_names(
        cls,
        names: list[str],
        target_names: list[str],
        question: str,
        description: str,
    ) -> list[str]:
        haystack = cls._normalize(f"{question} {description}")
        targets = [cls._normalize(target) for target in target_names]
        matched = []
        for name in names:
            normalized = cls._normalize(name)
            if targets and any(target == normalized or target in normalized for target in targets):
                matched.append(name)
            elif not targets and normalized and normalized in haystack:
                matched.append(name)
        return matched

    @staticmethod
    def _normalize(value: Any) -> str:
        text = unicodedata.normalize("NFKD", str(value or "")).casefold()
        return "".join(character for character in text if character.isalnum())

    @staticmethod
    def _escape(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().replace("|", "\\|")
