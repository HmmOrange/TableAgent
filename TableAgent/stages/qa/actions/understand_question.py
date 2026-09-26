from __future__ import annotations

from pathlib import Path
from typing import Any

from TableAgent.stages.qa.prompts.understanding import (
    UNDERSTANDING_SYSTEM_PROMPT,
    UNDERSTANDING_USER_PROMPT_TEMPLATE,
)

MAX_WORKBOOK_CHARS = 40000
MAX_PREVIEW_COLUMNS = 100
MAX_PREVIEW_VALUE_LENGTH = 1000


def workbook_preview(
    workbook: Any,
    workbook_name: str,
    excluded_sheet_names: set[str] | None = None,
    max_chars: int = MAX_WORKBOOK_CHARS,
) -> str:
    """Render sheets as A1-annotated rows, splitting the char budget evenly per sheet."""
    excluded = excluded_sheet_names or set()
    sheets = [
        sheet
        for sheet in workbook.worksheets
        if sheet.title.strip().casefold() not in excluded
    ]
    parts = [f"**Excel File Overview: {workbook_name}**", f"**Total Sheets:** {len(sheets)}"]
    sheet_budget = max_chars // max(len(sheets), 1)
    for sheet in sheets:
        parts.append(f"\n**Sheet: '{sheet.title}'**")
        parts.append(f"- Dimensions: {sheet.max_row} rows x {sheet.max_column} columns")
        rows = []
        used = 0
        for row in sheet.iter_rows(max_col=min(sheet.max_column, MAX_PREVIEW_COLUMNS)):
            line = "| " + " | ".join(
                f"{cell.coordinate}:{_display_value(cell.value)}" for cell in row
            ) + " |"
            remaining = sheet_budget - used
            if len(line) > remaining:
                if remaining > 0:
                    rows.append(line[:remaining])
                break
            rows.append(line)
            used += len(line)
        parts.append(f"- Data Preview ({len(rows)} of {sheet.max_row} rows):")
        parts.extend(rows)
    return "\n".join(parts)


def _display_value(value: Any) -> str:
    text = "" if value is None else str(value)[:MAX_PREVIEW_VALUE_LENGTH]
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


class UnderstandQuestionAction:
    """Clarify a question against raw workbook content before planning."""

    name = "understand_question"

    def __init__(self, env: Any, llm_client: Any):
        self.env = env
        self.llm_client = llm_client

    def run(self, question: str) -> str:
        prompt = UNDERSTANDING_USER_PROMPT_TEMPLATE.format(
            question=question,
            workbook_content=workbook_preview(
                self.env.workbook,
                Path(self.env.workbook_path).name,
                getattr(self.env, "excluded_sheet_names", set()),
            ),
        )
        self.env.logger.log_event(
            "understanding_prompt",
            {"prompt": prompt, "system_prompt": UNDERSTANDING_SYSTEM_PROMPT},
        )
        response = self.llm_client.generate(prompt, system_prompt=UNDERSTANDING_SYSTEM_PROMPT)
        return str(response.content or "").strip()
