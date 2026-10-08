from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from TableAgent.stages.qa.prompts.understanding import (
    UNDERSTANDING_REVISION_TEMPLATE,
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
MATCH_BUDGET_DIVISOR = 4  # with question matches, a quarter goes to matching rows from the middle
QUESTION_STOPWORDS = frozenset(
    "what which when where whose with from that this these those than then there their they them "
    "have does were been being into onto over under about across between among each every total "
    "value values number numbers based table data sheet show shows shown given according provide "
    "calculate compare identify determine find list rank many much more most less least highest "
    "lowest answer year years rate rates".split()
)


def workbook_preview(
    workbook: Any,
    workbook_name: str,
    excluded_sheet_names: set[str] | None = None,
    max_chars: int = MAX_WORKBOOK_CHARS,
    question: str = "",
) -> str:
    """Render sheets as A1-annotated rows, splitting the char budget evenly per sheet.

    A sheet too long for its budget keeps its first and last rows; when a question is given,
    rows from the omitted middle that mention its terms are kept as well.
    """
    excluded = excluded_sheet_names or set()
    sheets = [
        sheet
        for sheet in workbook.worksheets
        if sheet.title.strip().casefold() not in excluded
    ]
    terms = _question_terms(question)
    parts = [f"**Excel File Overview: {workbook_name}**", f"**Total Sheets:** {len(sheets)}"]
    sheet_budget = max_chars // max(len(sheets), 1)
    for sheet in sheets:
        parts.append(f"\n**Sheet: '{sheet.title}'**")
        parts.append(f"- Dimensions: {sheet.max_row} rows x {sheet.max_column} columns")
        rows, matched = _preview_rows(sheet, sheet_budget, terms)
        shown = sum(1 for line in rows if not line.startswith("[rows "))
        note = "; rows from the omitted part that mention the question's terms are included" if matched else ""
        parts.append(f"- Data Preview ({shown} of {sheet.max_row} rows{note}):")
        parts.extend(rows)
    return "\n".join(parts)


def _words(value: Any) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", str(value)).casefold()))


def _question_terms(question: str) -> tuple[list[str], set[str]]:
    """Quoted phrases and distinctive words (and 4-digit numbers) of a question."""
    phrases = [
        _words(phrase)
        for phrase in re.findall(r"[\"“‘']([^\"”’']{3,80})[\"”’']", question or "")
    ]
    words = {
        word for word in _words(question).split()
        if re.fullmatch(r"\d{4}", word) or (len(word) >= 4 and not word.isdigit() and word not in QUESTION_STOPWORDS)
    }
    return [phrase for phrase in phrases if phrase], words


def _preview_rows(sheet: Any, budget: int, terms: tuple[list[str], set[str]]) -> tuple[list[str], bool]:
    """Pick whole rows within the budget: the head, matching middle rows, and the tail.

    Without question matches the head gets three quarters of the budget and the tail the rest.
    A first row longer than the head budget is cut so the tail still fits. Each matching row
    brings the nearest section label row above it, so its context is visible.
    """
    lines: list[str] = []
    texts: list[str] = []
    labels: list[bool] = []
    for row in sheet.iter_rows(max_col=min(sheet.max_column, MAX_PREVIEW_COLUMNS)):
        lines.append("| " + " | ".join(f"{cell.coordinate}:{_cell_text(cell)}" for cell in row) + " |")
        values = [cell.value for cell in row if cell.value not in (None, "")]
        texts.append(" ".join(str(value) for value in values))
        labels.append(len(values) == 1 and isinstance(values[0], str))
    if sum(map(len, lines)) <= budget:
        return lines, False

    scores = _match_scores(texts, terms)
    # A row also counts the match of the section label above it, so a record of a named section ranks first.
    section_score = 0
    for index, is_label in enumerate(labels):
        if is_label:
            section_score = scores[index]
        elif scores[index]:
            scores[index] += section_score
    match_budget = budget // MATCH_BUDGET_DIVISOR if any(scores) else 0
    head_budget = budget - budget // TAIL_BUDGET_DIVISOR - match_budget
    kept: dict[int, str] = {}
    used = 0
    for index, line in enumerate(lines):
        if used + len(line) > head_budget:
            if not kept:
                kept[index] = line[:head_budget]
                used = len(kept[index])
            break
        kept[index] = line
        used += len(line)
    head_end = max(kept)

    matched = False
    remaining = match_budget
    middle = range(head_end + 1, len(lines))
    # Only rows matching at least half as well as the best one; weaker matches are generic words.
    floor = (max((scores[i] for i in middle), default=0) + 1) // 2
    for index in sorted(middle, key=lambda i: (-scores[i], i)):
        if scores[index] <= 0 or scores[index] < floor:
            break
        label = None if labels[index] else next((i for i in range(index - 1, head_end, -1) if labels[i]), None)
        picks = [i for i in (label, index) if i is not None and i not in kept]
        cost = sum(len(lines[i]) for i in picks)
        if cost > remaining:
            continue
        for i in picks:
            kept[i] = lines[i]
        remaining -= cost
        used += cost
        matched = True

    tail_room = budget - used
    for index in range(len(lines) - 1, head_end, -1):
        if index in kept:
            continue
        if len(lines[index]) > tail_room:
            break
        kept[index] = lines[index]
        tail_room -= len(lines[index])

    rows: list[str] = []
    previous = -1
    for index in sorted(kept):
        if index > previous + 1:
            rows.append(f"[rows {previous + 2}-{index} omitted]")
        rows.append(kept[index])
        previous = index
    if previous < len(lines) - 1:
        rows.append(f"[rows {previous + 2}-{len(lines)} omitted]")
    return rows, matched


def _match_scores(texts: list[str], terms: tuple[list[str], set[str]]) -> list[int]:
    """Per row, how strongly it mentions the question: quoted phrases count most, then words."""
    phrases, words = terms
    if not phrases and not words:
        return [0] * len(texts)
    needed = 1 if len(words) <= 1 else 2
    scores = []
    for text in texts:
        normalized = f" {_words(text)} "
        phrase_hits = sum(1 for phrase in phrases if f" {phrase} " in normalized)
        word_hits = sum(1 for word in words if f" {word} " in normalized)
        scores.append(3 * phrase_hits + word_hits if phrase_hits or word_hits >= needed else 0)
    return scores


LEADING_INDENT = re.compile(r"^[\s\u00a0\u2007\u3000·]+")


def _cell_text(cell: Any) -> str:
    """A cell's display text, with a row label's indentation written as `[indent n] `.

    Indentation marks hierarchy but is invisible as text when it is a cell format, and is easy
    to miss as runs of spaces or bullets. Format indents keep Excel's level; leading spaces count
    one level per two characters, and leading bullets one level each.
    """
    value = cell.value
    if not isinstance(value, str):
        return _display_value(value)
    level = int(getattr(getattr(cell, "alignment", None), "indent", 0) or 0)
    leading = LEADING_INDENT.match(value)
    if leading and leading.end() < len(value):
        marks = leading.group(0)
        level = level or (marks.count("·") or (len(marks) + 1) // 2)
        value = value[leading.end():]
    text = _display_value(value)
    return f"[indent {level}] {text}" if level else text


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
        return self._generate(self._prompt(question), "understanding_prompt")

    def revise(self, question: str, decisions: str, answer: str, feedback: str) -> str:
        """A corrected decision sheet after a reviewer found the decisions read the question wrongly."""
        prompt = UNDERSTANDING_REVISION_TEMPLATE.format(
            understanding_prompt=self._prompt(question),
            decisions=decisions,
            answer=answer,
            feedback=feedback,
        )
        return self._generate(prompt, "understanding_revision_prompt")

    def _prompt(self, question: str) -> str:
        return UNDERSTANDING_USER_PROMPT_TEMPLATE.format(
            question=question,
            workbook_content=workbook_preview(
                self.env.workbook,
                Path(self.env.workbook_path).name,
                getattr(self.env, "excluded_sheet_names", set()),
                question=question,
            ),
        )

    def _generate(self, prompt: str, event: str) -> str:
        self.env.logger.log_event(event, {"prompt": prompt, "system_prompt": UNDERSTANDING_SYSTEM_PROMPT})
        response = self.llm_client.generate(prompt, system_prompt=UNDERSTANDING_SYSTEM_PROMPT)
        return str(response.content or "").strip()
