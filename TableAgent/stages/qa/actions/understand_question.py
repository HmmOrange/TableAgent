from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from TableAgent.stages.qa.prompts.understanding import (
    UNDERSTANDING_SYSTEM_PROMPT,
    UNDERSTANDING_USER_PROMPT_TEMPLATE,
)


def decisions_block(env: Any, template: str) -> str:
    """Render the question decisions for a later stage, or nothing when none were made."""
    decisions = str(getattr(env, "question_decisions", None) or "").strip()
    return template.format(decisions=decisions) if decisions else ""

MAX_WORKBOOK_CHARS = 40000
MAX_PREVIEW_COLUMNS = 100
MAX_PREVIEW_VALUE_LENGTH = 1000
TAIL_BUDGET_DIVISOR = 4  # a quarter of each sheet's budget is kept for its last rows


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
        rows = _head_and_tail(sheet, sheet_budget)
        shown = sum(1 for line in rows if not line.startswith("[rows "))
        parts.append(f"- Data Preview ({shown} of {sheet.max_row} rows):")
        parts.extend(rows)
    return "\n".join(parts)


def _head_and_tail(sheet: Any, budget: int) -> list[str]:
    """Stream a sheet's rows, keeping whole rows from its start and its end and marking the gap.

    The tail keeps notes, sources, and totals that long sheets place at the bottom. A first row
    longer than the head budget is cut so the tail still fits.
    """
    head_budget = budget - budget // TAIL_BUDGET_DIVISOR
    head: list[str] = []
    tail: deque[str] = deque()
    head_used = tail_used = total = 0
    head_open = True
    for row in sheet.iter_rows(max_col=min(sheet.max_column, MAX_PREVIEW_COLUMNS)):
        total += 1
        line = "| " + " | ".join(f"{cell.coordinate}:{_display_value(cell.value)}" for cell in row) + " |"
        if head_open:
            if head_used + len(line) <= head_budget:
                head.append(line)
                head_used += len(line)
                continue
            head_open = False
            if not head:
                head.append(line[:head_budget])
                head_used = len(head[0])
                continue
        tail.append(line)
        tail_used += len(line)
        while tail and tail_used > budget - head_used:
            tail_used -= len(tail.popleft())
    first_hidden, last_hidden = len(head) + 1, total - len(tail)
    if first_hidden > last_hidden:
        return head + list(tail)
    return head + [f"[rows {first_hidden}-{last_hidden} omitted]"] + list(tail)


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
