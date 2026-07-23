from __future__ import annotations

from dataclasses import dataclass
import re

from TableAgent.llm import BaseLLM, LLMResponse
from TableAgent.QA.language import (
    item_header_for_language,
    ordinal_header_for_language,
    required_answer_language,
)


SIFLEX_FORMATTER_SYSTEM_PROMPT = """You are the SIFLEX answer-formatting agent.
Your only task is to reshape a completed spreadsheet-QA answer into the requested
benchmark presentation type.

Strict rules:
- Preserve every fact, value, unit, row relationship, and the answer language.
- Do not calculate, infer, correct, summarize, omit, or add information.
- Preserve every label explicitly enumerated in the question verbatim. A translation may be added in parentheses, but
  must not replace the original label.
- Preserve each value under the header/label it came from; do not move a semantically similar value to another field.
- Use the draft's meaningful user-facing labels; never expose internal field IDs.
- Return only the reformatted answer, with no commentary or code fence.
- For `table`, return a valid Markdown table.
- For `list`, return a Markdown table with one answer unit per row and an ordinal first column. Use the exact ordinal
  header supplied in the user prompt; number rows consecutively from 1.
- For `form`, return a clearly sectioned document; retain tables when they carry
  row or field relationships.
"""

SIFLEX_FORMATTER_USER_PROMPT = """Question:
{question}

Required SIFLEX answer type: {answer_type}
Required answer language: {answer_language}
Required ordinal column for list answers: {ordinal_header}

Draft answer to reformat:
{draft_answer}

Reformat the draft now. Return only the answer."""


@dataclass(frozen=True)
class SiflexFormatterResult:
    answer: str
    response: LLMResponse
    fallback_used: bool = False


class SiflexAnswerFormatterAgent:
    supported_types = frozenset({"table", "list", "form"})

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @classmethod
    def supports(cls, answer_type: str) -> bool:
        return str(answer_type).strip().lower() in cls.supported_types

    def run(self, *, question: str, answer_type: str, draft_answer: str) -> SiflexFormatterResult:
        normalized_type = str(answer_type).strip().lower()
        answer_language = required_answer_language(question)
        ordinal_header = ordinal_header_for_language(answer_language)
        try:
            response = self.llm.generate(
                prompt=SIFLEX_FORMATTER_USER_PROMPT.format(
                    question=question,
                    answer_type=normalized_type,
                    answer_language=answer_language,
                    ordinal_header=ordinal_header,
                    draft_answer=draft_answer,
                ),
                system_prompt=SIFLEX_FORMATTER_SYSTEM_PROMPT,
            )
        except Exception as exc:
            response = LLMResponse(content=f"ERROR: SIFLEX formatter failed: {exc}")
        formatted = self._clean_response(response.content)
        if not formatted or formatted.upper().startswith("ERROR:"):
            fallback_answer = draft_answer
            if normalized_type == "list":
                fallback_answer = self._ensure_list_ordinal_column(
                    draft_answer,
                    ordinal_header=ordinal_header,
                    item_header=item_header_for_language(answer_language),
                )
            return SiflexFormatterResult(
                answer=fallback_answer,
                response=response,
                fallback_used=True,
            )
        if normalized_type == "table":
            draft_rows = self._markdown_table_rows(draft_answer)
            formatted_rows = self._markdown_table_rows(formatted)
            if draft_rows is not None and (formatted_rows is None or formatted_rows < draft_rows):
                return SiflexFormatterResult(
                    answer=draft_answer,
                    response=response,
                    fallback_used=True,
                )
        if normalized_type == "list":
            formatted = self._ensure_list_ordinal_column(
                formatted,
                ordinal_header=ordinal_header,
                item_header=item_header_for_language(answer_language),
            )
        return SiflexFormatterResult(answer=formatted, response=response)

    @staticmethod
    def _clean_response(content: str) -> str:
        text = str(content or "").strip()
        fenced = re.fullmatch(r"```(?:markdown|md)?\s*\n?(.*?)\n?```", text, flags=re.IGNORECASE | re.DOTALL)
        return fenced.group(1).strip() if fenced else text

    @staticmethod
    def _markdown_table_rows(answer: str) -> int | None:
        lines = [line.strip() for line in str(answer).strip().splitlines() if line.strip()]
        if len(lines) < 2 or not all(line.startswith("|") and line.endswith("|") for line in lines):
            return None
        separator_cells = [cell.strip() for cell in lines[1].strip("|").split("|")]
        if not separator_cells or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator_cells):
            return None
        return len(lines) - 2

    @staticmethod
    def _ensure_list_ordinal_column(answer: str, *, ordinal_header: str, item_header: str) -> str:
        lines = [line.strip() for line in str(answer).strip().splitlines() if line.strip()]
        if len(lines) >= 2 and all(line.startswith("|") and line.endswith("|") for line in lines):
            header_cells = [cell.strip() for cell in lines[0].strip("|").split("|")]
            if header_cells and header_cells[0].casefold() == ordinal_header.casefold():
                return "\n".join(lines)
            output = [
                f"| {ordinal_header} | " + " | ".join(header_cells) + " |",
                "| --- | " + " | ".join("---" for _ in header_cells) + " |",
            ]
            for index, line in enumerate(lines[2:], start=1):
                output.append(f"| {index} | " + line.strip("|").strip() + " |")
            return "\n".join(output)

        items = []
        for line in lines:
            item = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
            if item:
                if len(lines) == 1 and ";" in item:
                    items.extend(part.strip() for part in item.split(";") if part.strip())
                else:
                    items.append(item)
        output = [
            f"| {ordinal_header} | {item_header} |",
            "| --- | --- |",
        ]
        output.extend(f"| {index} | {item} |" for index, item in enumerate(items, start=1))
        return "\n".join(output)
