"""Last-resort QA fallback: the selected sheets' cells as text.

When structured QA fails, the other fallbacks only see what the structure
describes. A structure built wrongly for a sheet (a form with a title block,
side notes and a revision table around the real table) leaves no readable
data, while the answer is plainly in the cells. This fallback gives the model
the non-empty cells of the selected sheets, one line per row.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import openpyxl
import yaml
from openpyxl.utils import get_column_letter

# Kept well below the model context: the fallback is one call that must leave room to answer.
SHEET_CELLS_MAX_CHARS = 60_000
# Observations of the failed QA run, sent along only when the cells had to be cut: with the
# whole sheet present they add nothing and their misread values leak into the answer.
SHEET_CELLS_OBSERVATION_CHARS = 8_000
_CELL_MAX_CHARS = 1500
# Rows kept first when a sheet has to be cut: titles and header rows sit at the top.
_LEADING_ROWS = 8
_TERM = re.compile(r"\w{2,}", re.UNICODE)

_PROMPT = (
    "Question:\n{question}\n\n"
    "The structured spreadsheet QA could not read the selected workbook. Below are the non-empty cells of "
    "the selected sheet(s) of {workbook}, one line per row as cell=value; a merged cell appears once, in its "
    "top-left cell, and '...' marks rows left out. The sheet may be laid out as a form: read each value with "
    "the labels in its own row and in the header rows above it. Answer the question using only these cells. "
    "Quote values exactly, keeping units, limits and conditions. If the cells hold only part of what is "
    "asked, give that part and say briefly what is missing; if they hold nothing relevant, say so briefly."
    "\n\n{observations}{cells}\n\nAnswer:"
)
_OBSERVATIONS = (
    "Observations the failed QA computed before stopping. They may come from misread cells; use them only "
    "as hints and trust the cells below when they disagree:\n{text}\n\n"
)


def sheet_cells_fallback_prompt(
    question: str,
    workbook_path: Path | str,
    structure_paths: Iterable[Path | str],
    *,
    max_chars: int = SHEET_CELLS_MAX_CHARS,
    observations: str | None = None,
) -> str | None:
    """A prompt answering ``question`` from the cells of the structures' sheets, or None."""
    try:
        cells, cut = _sheet_cells(
            workbook_path, structure_sheet_names(structure_paths), question, max_chars=max_chars
        )
    except Exception:  # an unreadable workbook leaves the other fallbacks
        return None
    if not cells:
        return None
    return _PROMPT.format(
        question=question,
        workbook=Path(workbook_path).name,
        observations=_OBSERVATIONS.format(text=observations) if observations and cut else "",
        cells=cells,
    )


def structure_sheet_names(structure_paths: Iterable[Path | str]) -> list[str]:
    """Sheet names of the tables in the given structure files, in order, without repeats."""
    names: list[str] = []
    for path in structure_paths:
        try:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        tables = payload.values() if isinstance(payload, dict) else []
        for table in tables:
            sheet = str(table.get("sheet") or "").strip() if isinstance(table, dict) else ""
            if sheet and sheet not in names:
                names.append(sheet)
    return names


def sheet_cells_text(
    workbook_path: Path | str,
    sheet_names: Iterable[str],
    question: str,
    *,
    max_chars: int = SHEET_CELLS_MAX_CHARS,
) -> str:
    """Non-empty cells of ``sheet_names`` as text, cut to ``max_chars``.

    A cut keeps the leading rows of each sheet and the rows sharing the most
    words with the question, then fills the rest in row order.
    """
    return _sheet_cells(workbook_path, sheet_names, question, max_chars=max_chars)[0]


def _sheet_cells(
    workbook_path: Path | str,
    sheet_names: Iterable[str],
    question: str,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    """The cells text and whether rows had to be left out."""
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    try:
        sheets: list[tuple[str, list[tuple[int, str]]]] = []
        for wanted in sheet_names:
            name = _resolve_sheet(workbook.sheetnames, wanted)
            if name is None or any(name == seen for seen, _ in sheets):
                continue
            sheets.append((name, _row_lines(workbook[name])))
    finally:
        workbook.close()
    if not sheets:
        return "", False
    full = _render(sheets, None)
    if len(full) <= max_chars:
        return full, False
    return _render(sheets, _kept_rows(sheets, question, max_chars)), True


def _row_lines(worksheet: Any) -> list[tuple[int, str]]:
    lines = []
    for row_index, row in enumerate(worksheet.iter_rows(values_only=True), start=1):
        cells = [
            f"{get_column_letter(column)}{row_index}={text}"
            for column, value in enumerate(row, start=1)
            if (text := _cell_text(value))
        ]
        if cells:
            lines.append((row_index, f"R{row_index}: " + " | ".join(cells)))
    return lines


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        text = value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat(sep=" ")
    elif isinstance(value, date):
        text = value.isoformat()
    else:
        text = str(value)
    text = " / ".join(part.strip() for part in text.replace("_x000D_", "").splitlines() if part.strip())
    return text[:_CELL_MAX_CHARS] + "…" if len(text) > _CELL_MAX_CHARS else text


def _resolve_sheet(sheet_names: list[str], wanted: str) -> str | None:
    if wanted in sheet_names:
        return wanted
    for key in (lambda name: name.strip().casefold(), _sheet_key):
        matches = [name for name in sheet_names if key(name) == key(wanted)]
        if len(matches) == 1:
            return matches[0]
    return None


def _sheet_key(name: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFC", name).casefold() if ch.isalnum())


def _terms(text: str) -> set[str]:
    # The answer-language line appended to QA questions is not part of the question.
    question = str(text or "").split("\n\n[Answer language]", 1)[0]
    return {term.casefold() for term in _TERM.findall(unicodedata.normalize("NFC", question))}


def _kept_rows(
    sheets: list[tuple[str, list[tuple[int, str]]]], question: str, max_chars: int
) -> set[tuple[str, int]]:
    terms = _terms(question)
    kept: set[tuple[str, int]] = set()
    used = 0

    def keep(key: tuple[str, int], line: str) -> None:
        nonlocal used
        if key not in kept and used + len(line) + 1 <= max_chars:
            kept.add(key)
            used += len(line) + 1

    for name, lines in sheets:
        used += len(name) + 12
        for row_index, line in lines[:_LEADING_ROWS]:
            keep((name, row_index), line)
    scored = sorted(
        (
            (-len(terms & _terms(line)), position, name, row_index, line)
            for position, (name, lines) in enumerate(sheets)
            for row_index, line in lines
        ),
    )
    for score, _, name, row_index, line in scored:
        if score < 0:
            keep((name, row_index), line)
    for name, lines in sheets:
        for row_index, line in lines:
            keep((name, row_index), line)
    return kept


def _render(
    sheets: list[tuple[str, list[tuple[int, str]]]], kept: set[tuple[str, int]] | None
) -> str:
    parts = []
    for name, lines in sheets:
        out = [f"## Sheet: {name}"]
        skipped = False
        for row_index, line in lines:
            if kept is None or (name, row_index) in kept:
                out.append(line)
                skipped = False
            elif not skipped:
                out.append("...")
                skipped = True
        parts.append("\n".join(out))
    return "\n\n".join(parts)
