"""The `sheet` object of the notebook: reads, searches and describes workbook cells.

Reads resolve merged cells to the value a reader sees. `cells` gives the cells of a range with their
formats as a DataFrame, for selecting by format in code.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from openpyxl.utils import get_column_letter

from TableAgent.src.prompts import HEADERS_LOOK_QUESTION
from TableAgent.src.tools.headers import (
    WHITE_FILLS,
    HeaderWalker,
    cell_format,
    colour_name,
    font_colour,
    format_text,
    headers_line,
    plain,
    text_of,
)
from TableAgent.src.tools.look import MAX_LOOK_COLUMNS, MAX_LOOK_ROWS
from TableAgent.utils.excel_utils import cell_to_a1, parse_a1_range

MAX_HEADER_LOOKS = 3  # images `headers` asks a vision model about, per question
HEADER_LOOK_ROWS = 40  # the image runs from up to this many rows above the cell ...
HEADER_LOOK_COLUMNS = 20  # ... and this many columns to its left, the cell included
MAX_FORMAT_CELLS = 300  # filled cells `formats` lists
WHOLE_COLUMNS = re.compile(r"([A-Z]{1,3}):([A-Z]{1,3})")
CELL_COLUMNS = ["ref", "row", "column", "value", "range", "bold", "italic", "fill", "fill_colour", "font", "font_colour", "indent", "formula"]
MAX_NEAR_CELLS = 10  # cells `find` lists when no cell holds all the words
MAX_STACKED_LABELS = 3  # label cells `find` joins down a column (a header split over rows)


class SheetTools:
    """Read, search and describe workbook cells; merged cells read as their visible value."""

    def __init__(
        self,
        workbook: Any,
        look: Callable[[str, str, str], str] | None = None,
        formula_path: Path | None = None,
    ):
        self.workbook = workbook
        self._look = look  # (ref, question, sheet) -> the vision model's answer; None without a vision model
        self._walker = HeaderWalker(self, formula_path)
        self._header_looks = 0
        self._merged: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
        # Read once, before any cell is touched: openpyxl creates every cell it is asked for, which grows the sheet.
        self._sizes = {ws.title: (ws.max_row, ws.max_column) for ws in workbook.worksheets}

    def sheets(self) -> list[str]:
        """Sheet names with their used size, e.g. `Data (233 rows x 19 columns)`."""
        return [f"{ws.title} ({self.size(ws)[0]} rows x {self.size(ws)[1]} columns)" for ws in self.workbook.worksheets]

    def size(self, ws: Any) -> tuple[int, int]:
        """(rows, columns) of a sheet as it was loaded; every cell past them is empty."""
        return self._sizes[ws.title]

    def worksheet(self, sheet: str = "") -> Any:
        """The named sheet, matched ignoring case and spacing; the first sheet when unnamed."""
        names = self.workbook.sheetnames
        if not sheet:
            return self.workbook[names[0]]
        wanted = plain(sheet)
        for name in names:
            if plain(name) == wanted:
                return self.workbook[name]
        if len(names) == 1:
            return self.workbook[names[0]]
        raise ValueError(f"No sheet named {sheet!r}; sheets: {names}")

    def value(self, row: int, col: int, sheet: str = "") -> Any:
        """The visible value at a position: inside a merged range, the range's top-left value; text without its
        outer spaces, which do not show (blank text is empty)."""
        ws = self.worksheet(sheet)
        anchor_row, anchor_col = self._merged_map(ws).get((row, col), (row, col))
        rows, cols = self.size(ws)
        if anchor_row > rows or anchor_col > cols:
            return None
        value = ws.cell(anchor_row, anchor_col).value
        return (value.strip() or None) if isinstance(value, str) else value

    def read(self, ref: str, sheet: str = "") -> list[list[Any]]:
        """Values of an A1 cell or range (e.g. "B3:F9" or "'Data'!B3"), row by row."""
        cells = parse_a1_range(ref, sheet)
        return [
            [self.value(row, col, cells.sheet) for col in range(cells.start_col, cells.end_col + 1)]
            for row in range(cells.start_row, cells.end_row + 1)
        ]

    def find(self, text: Any, sheet: str = "") -> list[tuple[str, Any]]:
        """(A1 ref, value) of every cell whose text contains `text` as whole words, ignoring case and spacing.

        When no cell does: the label cells that do once plural endings are ignored, a label joined with the label
        cells right below it (a header split over rows; value: the joined text); when none does either, up to
        MAX_NEAR_CELLS label cells holding most of its words, most first.
        """
        query = _words(text)
        if not query:
            return []
        ws = self.worksheet(sheet)
        cells = [cell for row in ws.iter_rows() for cell in row if cell.value is not None]
        found = [(cell.coordinate, cell.value) for cell in cells if f" {query} " in f" {_words(cell.value)} "]
        if found:
            return found
        stems = _stems(text)
        found = [(ref, label) for ref, label in _stacked_labels(ws) if f" {stems} " in f" {_stems(label)} "]
        if found:
            return found
        words = set(stems.split())
        shared = [(len(words & set(_stems(cell.value).split())), cell) for cell in cells if isinstance(cell.value, str)]
        near = sorted((item for item in shared if len(words) > 1 and item[0] * 2 >= len(words)), key=lambda item: -item[0])
        return [(cell.coordinate, cell.value) for _, cell in near[:MAX_NEAR_CELLS]]

    def frame(self, sheet: str = "") -> pd.DataFrame:
        """The whole used range as a DataFrame: index = Excel row numbers, columns = column letters."""
        ws = self.worksheet(sheet)
        last_row, last_col = self.size(ws)
        rows = [
            [self.value(row, col, ws.title) for col in range(1, last_col + 1)]
            for row in range(1, last_row + 1)
        ]
        columns = [get_column_letter(col) for col in range(1, last_col + 1)]
        return pd.DataFrame(rows, columns=columns, index=range(1, last_row + 1))

    def headers(self, ref: str, sheet: str = "", look: bool = True) -> dict[str, Any]:
        """What one cell means: its row labels, column headers, sections and block.

        Found by walking the layout in code (tools/headers.py). When the walk is unsure about a filled cell and a
        vision model is available, `vision` adds its answer about an image of the area above and left of the cell,
        widened so that no merged cell is cut.
        """
        cells = parse_a1_range(ref, sheet)
        ws = self.worksheet(cells.sheet)
        found = self._walker.headers(ws, cells.start_row, cells.start_col)
        if (
            look and found["uncertain"] and found["value"] is not None
            and self._look is not None and self._header_looks < MAX_HEADER_LOOKS
        ):
            self._header_looks += 1
            area = self._uncut(
                ws, max(1, cells.start_row - HEADER_LOOK_ROWS + 1), max(1, cells.start_col - HEADER_LOOK_COLUMNS + 1),
                cells.start_row, cells.start_col,
            )
            try:
                found["vision"] = self._look(area, HEADERS_LOOK_QUESTION.format(cell=found["cell"]), ws.title)
            except Exception as exc:  # the image cannot be rendered or the call budget is used up
                found["vision"] = f"unavailable: {exc}"
        return found

    def describe(self, ref: str, sheet: str = "") -> str:
        """One line for `headers(ref)` without the vision model, e.g. `D8 = 35 | row: Alpha | column: 2019`."""
        return headers_line(self.headers(ref, sheet, look=False))

    def cells(self, ref: str, sheet: str = "") -> pd.DataFrame:
        """One row per filled cell of a range, a merged cell once (at its first cell, with its range): ref, row,
        column, value, range, bold, italic, fill, fill_colour, font, font_colour, indent, formula. No fill or a white one is
        None; so is black font; the colour columns name an RGB value ("red", "green", ...)."""
        whole = WHOLE_COLUMNS.fullmatch(ref.strip())
        if whole:  # "C:C" or "C:F": the columns over every row of the sheet
            ref = f"{whole.group(1)}1:{whole.group(2)}{self.size(self.worksheet(sheet))[0]}"
        area = parse_a1_range(ref, sheet)
        ws = self.worksheet(area.sheet)
        covered = self._merged_map(ws)
        ranges = {(merged.min_row, merged.min_col): str(merged) for merged in ws.merged_cells.ranges}
        formulas = self._walker.formula_sheet(ws.title)
        seen: set[tuple[int, int]] = set()
        rows = []
        last_row, last_col = self.size(ws)
        for row in range(area.start_row, min(area.end_row, last_row) + 1):
            for col in range(area.start_col, min(area.end_col, last_col) + 1):
                anchor = covered.get((row, col), (row, col))
                cell = ws.cell(*anchor)
                value = self.value(*anchor, ws.title)
                if anchor in seen or value in (None, ""):
                    continue
                seen.add(anchor)
                bold, fill, indent = cell_format(cell)
                font = font_colour(cell)
                formula = formulas.cell(*anchor).value if formulas is not None else None
                rows.append({
                    "ref": cell_to_a1(*anchor), "row": anchor[0], "column": get_column_letter(anchor[1]), "value": value,
                    "range": ranges.get(anchor), "bold": bold, "italic": bool(cell.font and cell.font.i),
                    "fill": None if fill in WHITE_FILLS else fill, "fill_colour": None if fill in WHITE_FILLS else colour_name(fill),
                    "font": font, "font_colour": colour_name(font), "indent": indent,
                    "formula": formula if isinstance(formula, str) and formula.startswith("=") else None,
                })
        return pd.DataFrame(rows, columns=CELL_COLUMNS)

    def formats(self, ref: str, sheet: str = "") -> str:
        """The filled cells of a range with their format and formula, a line each, e.g. `D9 = 45 | bold | =SUM(D5:D8)`."""
        cells = parse_a1_range(ref, sheet)
        ws = self.worksheet(cells.sheet)
        formulas = self._walker.formula_sheet(ws.title)
        lines = []
        last_row, last_col = self.size(ws)
        for row in range(cells.start_row, min(cells.end_row, last_row) + 1):
            for col in range(cells.start_col, min(cells.end_col, last_col) + 1):
                value = ws.cell(row, col).value
                if value in (None, ""):
                    continue
                if len(lines) == MAX_FORMAT_CELLS:
                    return "\n".join(lines + [f"... cut at {MAX_FORMAT_CELLS} cells"])
                cell = ws.cell(row, col)
                parts = [f"{cell_to_a1(row, col)} = {text_of(value)}", _format_words(cell)]
                formula = formulas.cell(row, col).value if formulas is not None else None
                if isinstance(formula, str) and formula.startswith("="):
                    parts.append(formula)
                lines.append(" | ".join(parts))
        return "\n".join(lines)

    def around(self, ref: str, rows: int = 2, cols: int = 2, sheet: str = "") -> str:
        """The filled cells around one cell, a line per row with their refs; the cell itself is marked *."""
        cells = parse_a1_range(ref, sheet)
        ws = self.worksheet(cells.sheet)
        row, col = cells.start_row, cells.start_col
        lines = []
        for r in range(max(1, row - rows), row + rows + 1):
            parts = [
                f"{'*' if (r, c) == (row, col) else ''}{cell_to_a1(r, c)}={text_of(value)}"
                for c in range(max(1, col - cols), col + cols + 1)
                if (value := self.value(r, c, ws.title)) not in (None, "") or (r, c) == (row, col)
            ]
            lines.append(f"row {r}: " + (" | ".join(parts) if parts else "(empty)"))
        return "\n".join(lines)

    def look(self, ref: str, question: str, sheet: str = "") -> str:
        """Ask a vision model about an image of a cell range, for what values do not show (ticks, shading, spans)."""
        if self._look is None:
            raise ValueError("sheet.look is not available in this run.")
        return self._look(ref, question, self.worksheet(sheet).title)

    def _uncut(self, ws: Any, top: int, left: int, bottom: int, right: int) -> str:
        """The range widened until it cuts no merged cell, within MAX_LOOK_ROWS x MAX_LOOK_COLUMNS from its top left."""
        grown = True
        while grown:
            grown = False
            for merged in ws.merged_cells.ranges:
                inside = merged.min_row <= bottom and merged.max_row >= top and merged.min_col <= right and merged.max_col >= left
                if not inside:
                    continue
                wider = (min(max(bottom, merged.max_row), top + MAX_LOOK_ROWS - 1),
                         min(max(right, merged.max_col), left + MAX_LOOK_COLUMNS - 1))
                if wider != (bottom, right):
                    bottom, right = wider
                    grown = True
        return f"{cell_to_a1(top, left)}:{cell_to_a1(bottom, right)}"

    def _merged_map(self, ws: Any) -> dict[tuple[int, int], tuple[int, int]]:
        if ws.title not in self._merged:
            self._merged[ws.title] = {
                (r, c): (merged.min_row, merged.min_col)
                for merged in ws.merged_cells.ranges
                for r in range(merged.min_row, merged.max_row + 1)
                for c in range(merged.min_col, merged.max_col + 1)
                if (r, c) != (merged.min_row, merged.min_col)
            }
        return self._merged[ws.title]


def _format_words(cell: Any) -> str:
    """Bold, fill and indent, plus italic and a font colour other than the default black."""
    font, words = cell.font, format_text(cell_format(cell))
    named = font_colour(cell)
    extra = (["italic"] if font is not None and font.i else []) + ([f"font {named}"] if named else [])
    return ", ".join(([] if words == "plain" else [words]) + extra) or words


def _stacked_labels(ws: Any) -> list[tuple[str, str]]:
    """Each text cell, and each text cell joined with the up to MAX_STACKED_LABELS - 1 text cells right below it."""
    labels = []
    for column in ws.iter_cols():
        for index, cell in enumerate(column):
            if not isinstance(cell.value, str) or not cell.value.strip():
                continue
            parts = [cell.value]
            for below in column[index + 1:index + MAX_STACKED_LABELS]:
                if not isinstance(below.value, str) or not below.value.strip():
                    break
                parts.append(below.value)
                labels.append((cell.coordinate, " ".join(parts)))
            labels.append((cell.coordinate, cell.value))
    return labels


def _stems(value: Any) -> str:
    """The words of `value` without plural endings: "indexes" and "index" both give "index"."""
    return " ".join(_stem(word) for word in _words(value).split())


def _stem(word: str) -> str:
    if len(word) <= 3:
        return word
    if re.search(r"(s|x|z|ch|sh)es$", word):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def _words(value: Any) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", str(value)).casefold()))
