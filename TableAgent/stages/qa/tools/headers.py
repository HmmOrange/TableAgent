"""What a cell means, found by walking from it up to its column headers and left to its row labels.

The walk reads the layout in code (see `HeaderWalker.headers`):
- column headers: the header rows structure.yaml names, when it names some above the cell; otherwise the
  label cells above the cell's data, walking up.
- row labels: the label cells left of the cell, and keys written once above and carried down (a location
  in column A, then empty or merged cells below it).
- sections: label-only rows above the cell, nearest first, then each parent (a label further left, less
  indented, or bold over plain), each marked when structure.yaml names it as a group.
- block: the run of rows with data shaped like the cell's row, and why it starts and ends.
- uncertain: why the walk may be wrong (no header, no label, a block that starts only at a format change
  where structure.yaml says nothing); `SheetTools.headers` then asks a vision model about an image.

A sheet's layout is read once. Data columns are mostly filled and mostly numbers; key columns are the
label columns left of them. A block ends at a row with no data (a section label or a blank row), a total
computed by an aggregate formula, a row whose label cell is formatted differently, or a row whose data
columns mostly change kind. An empty cell inside the data never ends a block. A format marks structure
only when it is rare next to the other, and a fill is ignored where rows alternate two fills (striped for
reading): only bold and indent then count.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.utils import column_index_from_string, get_column_letter

from TableAgent.utils.excel_utils import cell_to_a1, parse_a1_range

SCAN_COLUMNS = 60
DATA_NUMBER_SHARE = 0.5  # a data column: most of its filled cells are numbers ...
DATA_FILLED_SHARE = 0.25  # ... and it is filled in at least this share of the body rows
MIN_STRIPED_ROWS = 4  # rows alternating two fills for at least this long are striped
ROW_LABELS = 3  # label cells left of a cell that the walk keeps
HEADER_LEVELS = 4  # label cells above a cell that the walk keeps
SECTION_MAX_CELLS = 2  # a section label row holds at most this many filled cells (a merged range counts once)
MAX_SIBLINGS = 20
AGGREGATE = re.compile(r"^=\s*(SUM|SUBTOTAL|AVERAGE|COUNTA?|MIN|MAX)\s*\(", re.IGNORECASE)
RANGE = re.compile(r"\$?([A-Z]{1,3})\$?(\d+)\s*:\s*\$?([A-Z]{1,3})\$?(\d+)")
FORMAT_ONLY = ("formatted differently", "changes the kind")  # block starts that only a format marks


class HeaderWalker:
    """Walks from cells to their headers, labels and sections; `tools` is the SheetTools it serves."""

    def __init__(self, tools: Any, formula_path: Path | None = None):
        self.tools = tools
        self._formula_path = formula_path  # the workbook file, read again with formulas for total rows
        self._formula_workbook: Any = None
        self._sections: dict[str, list[dict[str, Any]]] = {}
        self._layouts: dict[tuple[str, int], Layout] = {}

    def headers(self, ws: Any, row: int, col: int) -> dict[str, Any]:
        """What the walk finds for one cell (see the module docstring)."""
        column_headers, top, source = self.headers_above(ws, row, col)
        row_labels = [
            text_of(value)
            for value in (self.tools.value(row, c, ws.title) for c in range(col - 1, 0, -1))
            if is_label(value)
        ][:ROW_LABELS]
        layout = self.layout(ws, top)
        found_block = block(layout, row) if top < row <= len(layout.values) else None
        found_keys = [
            key for key in self.keys(ws, row, col, top, found_block) if key["from"] != f"{key['column']}{row}"
        ]
        group = self.section_group(ws, row, top)
        structure = self.tools.structure_at(row, col, ws.title)
        uncertain = []
        if not column_headers:
            uncertain.append("no column header found above the cell")
        if not row_labels and not found_keys:
            uncertain.append("no row label found left of the cell")
        if found_block and not structure and any(mark in found_block["starts"] for mark in FORMAT_ONLY):
            uncertain.append(f"structure.yaml does not cover the cell and its block starts only at a format change ({found_block['starts']})")
        return {
            "cell": cell_to_a1(row, col),
            "sheet": ws.title,
            "value": self.tools.value(row, col, ws.title),
            "row_labels": list(reversed(row_labels)),
            "keys": found_keys,
            "column_headers": list(reversed(column_headers)),
            "headers_from": source,
            "sections": group["sections"],
            "siblings": group["siblings"],
            "block": {key: found_block[key] for key in ("rows", "starts", "ends")} if found_block else None,
            "structure": structure,
            "uncertain": uncertain,
        }

    def headers_above(self, ws: Any, row: int, col: int) -> tuple[list[str], int, str]:
        """Header cells above a cell (nearest first), the lowest header row, and where they come from.

        The header rows structure.yaml names are read in the cell's column, since a header may be a
        number (a year) that looks like data. Without them, label cells are collected walking up past
        the data, up to the next number.
        """
        rows = [r for r in self._header_rows(ws.title) if r < row]
        if rows:
            values = [self.tools.value(r, col, ws.title) for r in sorted(rows, reverse=True)]
            headers = [text_of(value) for value in values if value not in (None, "")]
            headers = [text for index, text in enumerate(headers) if index == 0 or text != headers[index - 1]]
            return headers, max(rows), "structure.yaml"
        headers: list[str] = []
        top = 0
        for r in range(row - 1, 0, -1):
            value = self.tools.value(r, col, ws.title)
            if is_label(value):
                headers.append(text_of(value))
                top = r
                if len(headers) == HEADER_LEVELS:
                    break
            elif value is not None and headers:
                break
        return headers, top, "the layout"

    def section_group(self, ws: Any, row: int, top: int) -> dict[str, Any]:
        """The section rows above a row (its own, then each parent) and the siblings of its own, with row spans."""
        sections = [section for section in self.section_rows(ws) if section["row"] > top]

        def span(start: int, level: tuple[int, ...]) -> str:
            end = next((s["row"] for s in sections if s["row"] > start and s["level"] <= level), ws.max_row + 1)
            return f"{start + 1}-{end - 1}"

        def shown(section: dict[str, Any]) -> dict[str, Any]:
            found = {"label": section["label"], "row": section["row"], "rows": span(section["row"], section["level"])}
            named = self.tools.structure_at(section["row"], section["col"], ws.title, label=True)
            return {**found, "structure": named} if named else found

        above = [section for section in sections if section["row"] < row]
        if not above:
            return {"sections": [], "siblings": []}
        own = above[-1]
        parents = []
        for section in reversed(above[:-1]):
            if section["level"] < (parents[-1]["level"] if parents else own["level"]):
                parents.append(section)
        parent = parents[0] if parents else None
        parent_end = int(span(parent["row"], parent["level"]).split("-")[1]) if parent else ws.max_row
        siblings = [
            shown(s) for s in sections
            if s["level"] == own["level"] and s["row"] != own["row"] and (parent is None or parent["row"] < s["row"] <= parent_end)
        ]
        return {"sections": [shown(own), *(shown(s) for s in parents)], "siblings": siblings[:MAX_SIBLINGS]}

    def section_rows(self, ws: Any) -> list[dict[str, Any]]:
        """Every section label row of a sheet, top to bottom: a row holding only a label or two.

        Raw cells are read, so a label merged across the row counts once. `level` is the first filled
        column, the indent there, and bold before plain: a smaller level is a parent.
        """
        if ws.title not in self._sections:
            found = []
            for r, values in enumerate(ws.iter_rows(max_col=SCAN_COLUMNS, values_only=True), start=1):
                filled = [(c, value) for c, value in enumerate(values, start=1) if value not in (None, "")]
                if not filled or len(filled) > SECTION_MAX_CELLS or not all(is_label(value) for _, value in filled):
                    continue
                first = ws.cell(r, filled[0][0])
                found.append({
                    "row": r,
                    "col": first.column,
                    "label": " / ".join(text_of(value) for _, value in filled),
                    "level": (first.column, int(first.alignment.indent or 0), 0 if first.font and first.font.b else 1),
                })
            self._sections[ws.title] = found
        return self._sections[ws.title]

    def keys(self, ws: Any, row: int, col: int, top: int, found_block: dict[str, Any] | None) -> list[dict[str, Any]]:
        """The key values a row shares with other rows of its block, each named by its column header."""
        if found_block is None:
            return []
        found = keys(self.layout(ws, top), row, found_block["first"], found_block["last"])
        for key in found:
            column = parse_a1_range(key["from"]).start_col
            header = self.headers_above(ws, top + 1, column)[0]
            key["header"] = header[0] if header else key["column"]
        return found

    def layout(self, ws: Any, top: int) -> Layout:
        if (ws.title, top) not in self._layouts:
            self._layouts[(ws.title, top)] = read_layout(ws, top, self.formula_sheet(ws.title))
        return self._layouts[(ws.title, top)]

    def formula_sheet(self, title: str) -> Any:
        """The sheet read with formulas instead of their values, or None without the workbook file."""
        if self._formula_path is not None and self._formula_workbook is None:
            self._formula_workbook = openpyxl.load_workbook(self._formula_path)
        return self._formula_workbook[title] if self._formula_workbook is not None else None

    def _header_rows(self, sheet_title: str) -> set[int]:
        """Rows holding the labels of structure.yaml headers on a sheet."""
        return {
            r
            for table in self.tools.structure
            for entry in table.entries
            if entry.kind == "header" and entry.label_range is not None and plain(entry.sheet) == plain(sheet_title)
            for r in range(entry.label_range.start_row, entry.label_range.end_row + 1)
        }


def headers_line(found: dict[str, Any]) -> str:
    """One line for a walk, e.g. `D8 = 35 | row: Alpha | column: 2019 | section: South (row 7)`."""
    parts = [f"{found['cell']} = {text_of(found['value']) or '(empty)'}"]
    if found["row_labels"]:
        parts.append("row: " + " > ".join(found["row_labels"]))
    for key in found["keys"]:
        parts.append(f"key: {key['header']} = {text_of(key['value'])} (from {key['from']})")
    if found["column_headers"]:
        parts.append("column: " + " > ".join(found["column_headers"]))
    for section in found["sections"]:
        named = "".join(f", {entry}" for entry in section.get("structure", []))
        parts.append(f"section: {section['label']} (row {section['row']}{named})")
    for entry in found["structure"]:
        parts.append(f"structure {entry}")
    if found["uncertain"]:
        parts.append("unsure: " + "; ".join(found["uncertain"]))
    return " | ".join(parts)


# ---------------------------------------------------------------------------------------------------- layout

@dataclass
class Layout:
    ws: Any
    top: int  # the lowest header row; the body starts below it
    values: list[list[Any]]  # raw values[r - 1][c - 1]: a merged range holds its value in its first cell only
    totals: dict[int, tuple[int, int]] = field(default_factory=dict)  # row -> rows its aggregate formula covers
    data_columns: list[int] = field(default_factory=list)
    key_columns: list[int] = field(default_factory=list)
    format_counts: dict[int, dict[tuple[Any, ...], int]] = field(default_factory=dict)  # per column
    striped: dict[int, set[int]] = field(default_factory=dict)  # per column, rows whose fill only stripes

    def format(self, row: int, col: int) -> tuple[Any, ...]:
        """(bold, fill, indent) of a cell; the fill is left out where the rows are striped."""
        bold, fill, indent = cell_format(self.ws.cell(row, col))
        return (bold, None if row in self.striped_rows(col) else fill, indent)

    def striped_rows(self, col: int) -> set[int]:
        """Body rows of a column inside a run that alternates two fills, at least MIN_STRIPED_ROWS long."""
        if col not in self.striped:
            striped: set[int] = set()
            run: list[int] = []
            fills: dict[int, Any] = {}
            for r in range(self.top + 1, len(self.values) + 1):
                fills[r] = cell_format(self.ws.cell(r, col))[1]
                if run and fills[r] != fills[r - 1] and (len(run) < 2 or fills[r] == fills[r - 2]):
                    run.append(r)
                    continue
                if len(run) >= MIN_STRIPED_ROWS:
                    striped.update(run)
                run = [r]
            if len(run) >= MIN_STRIPED_ROWS:
                striped.update(run)
            self.striped[col] = striped
        return self.striped[col]

    def format_count(self, col: int, fmt: tuple[Any, ...]) -> int:
        """How many filled body cells of a column have this format."""
        if col not in self.format_counts:
            counts: dict[tuple[Any, ...], int] = {}
            for r in range(self.top + 1, len(self.values) + 1):
                if _kind(self.value(r, col)) != ".":
                    found = self.format(r, col)
                    counts[found] = counts.get(found, 0) + 1
            self.format_counts[col] = counts
        return self.format_counts[col].get(fmt, 0)

    def value(self, row: int, col: int) -> Any:
        if row < 1 or row > len(self.values) or col > len(self.values[row - 1]):
            return None
        return self.values[row - 1][col - 1]

    def label_column(self, row: int) -> int | None:
        """The first label column filled in a row."""
        columns = self.key_columns or range(1, len(self.values[row - 1]) + 1)
        return next((c for c in columns if _kind(self.value(row, c)) != "."), None)


def read_layout(ws: Any, top: int, formula_ws: Any = None) -> Layout:
    """The layout of a sheet whose header ends at row `top`; `formula_ws` is the same sheet read with formulas."""
    width = min(ws.max_column, SCAN_COLUMNS)
    values = [list(row) for row in ws.iter_rows(max_col=width, values_only=True)]
    body = values[top:]
    data_columns = []
    for c in range(1, width + 1):
        filled = [row[c - 1] for row in body if _kind(row[c - 1]) != "."]
        numbers = [value for value in filled if _kind(value) == "n"]
        if body and filled and len(numbers) >= DATA_NUMBER_SHARE * len(filled) and len(filled) >= DATA_FILLED_SHARE * len(body):
            data_columns.append(c)
    first_data = data_columns[0] if data_columns else width + 1
    key_columns = [c for c in range(1, first_data) if any(_kind(row[c - 1]) != "." for row in body)]
    layout = Layout(ws, top, values, data_columns=data_columns, key_columns=key_columns)
    if formula_ws is not None:
        layout.totals = _aggregate_rows(formula_ws, width)
    return layout


def block(layout: Layout, row: int) -> dict[str, Any]:
    """The run of rows with the shape of `row`: {"rows": "a-b", "starts": why, "ends": why}."""
    start = row
    while not (reason_above := _boundary(layout, start - 1, row)):
        start -= 1
    end = row
    while not (reason_below := _boundary(layout, end + 1, row)):
        end += 1
    return {"rows": f"{start}-{end}", "starts": reason_above, "ends": reason_below, "first": start, "last": end}


def keys(layout: Layout, row: int, first: int, last: int) -> list[dict[str, Any]]:
    """For each key column, the value `row` belongs to (its own, or carried down) and the rows sharing it."""
    found = []
    for c in layout.key_columns:
        source = next((r for r in range(row, first - 1, -1) if _kind(layout.value(r, c)) != "."), None)
        if source is None:
            continue
        value = layout.value(source, c)
        start = source
        for r in range(source - 1, first - 1, -1):
            other = layout.value(r, c)
            if _kind(other) != "." and other != value:
                break
            if other == value:
                start = r
        end = row
        while end < last and (_kind(layout.value(end + 1, c)) == "." or layout.value(end + 1, c) == value):
            end += 1
        if end == start and source == row:
            continue  # a value of this row alone, not a group
        following = layout.value(end + 1, c) if end < last else None
        found.append({
            "column": get_column_letter(c),
            "value": value,
            "from": f"{get_column_letter(c)}{source}",
            "rows": f"{start}-{end}",
            "next": following,
        })
    return found


def _boundary(layout: Layout, r: int, row: int) -> str:
    """Why row `r` is not in the block of `row`, or an empty string."""
    if r <= layout.top:
        return f"row {r} is the header" if r >= 1 else "start of sheet"
    if r > len(layout.values):
        return "end of sheet"
    columns = layout.data_columns or range(1, len(layout.values[r - 1]) + 1)
    kinds = [_kind(layout.value(r, c)) for c in columns]
    if all(kind == "." for kind in kinds):
        label = next((layout.value(r, c) for c in range(1, len(layout.values[r - 1]) + 1)
                      if _kind(layout.value(r, c)) != "."), None)
        return f"row {r} has no data" + (f" (label {str(label).strip()!r})" if label is not None else " (blank)")
    if r in layout.totals:
        covered = layout.totals[r]
        return f"row {r} totals rows {covered[0]}-{covered[1]} by formula"
    column = layout.label_column(row) or layout.label_column(r)
    if column is not None:
        here, there = layout.format(r, column), layout.format(row, column)
        if _kind(layout.value(r, column)) != "." and here != there:
            rarer, commoner = sorted((layout.format_count(column, here), layout.format_count(column, there)))
            if 2 * rarer < commoner:
                return f"row {r} is formatted differently ({format_text(here)})"
    if layout.data_columns:
        pairs = [
            (_kind(layout.value(r, c)), _kind(layout.value(row, c)))
            for c in layout.data_columns
            if _kind(layout.value(r, c)) != "." and _kind(layout.value(row, c)) != "."
        ]
        if pairs and 2 * sum(a != b for a, b in pairs) > len(pairs):
            return f"row {r} changes the kind of most data columns"
    return ""


def _aggregate_rows(ws: Any, width: int) -> dict[int, tuple[int, int]]:
    totals: dict[int, tuple[int, int]] = {}
    for row in ws.iter_rows(max_col=width):
        for cell in row:
            if not (isinstance(cell.value, str) and AGGREGATE.match(cell.value)):
                continue
            for first_col, first_row, last_col, last_row in RANGE.findall(cell.value):
                same_column = column_index_from_string(first_col) == column_index_from_string(last_col) == cell.column
                if same_column and int(last_row) < cell.row:
                    totals.setdefault(cell.row, (int(first_row), int(last_row)))
    return totals


def _kind(value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "."
    if isinstance(value, bool):
        return "t"
    if isinstance(value, (int, float)) or hasattr(value, "year"):
        return "n"
    return "t"


def cell_format(cell: Any) -> tuple[Any, ...]:
    """(bold, fill, indent) of a cell."""
    fill = None
    if cell.fill is not None and cell.fill.fill_type:
        color = cell.fill.fgColor
        fill = {"rgb": color.rgb, "theme": f"theme {color.theme}", "indexed": f"indexed {color.indexed}"}.get(color.type)
    return (bool(cell.font and cell.font.b), fill, int(cell.alignment.indent or 0))


def format_text(fmt: tuple[Any, ...]) -> str:
    bold, fill, indent = fmt
    parts = (["bold"] if bold else []) + ([f"fill {fill}"] if fill else []) + ([f"indent {indent}"] if indent else [])
    return ", ".join(parts) or "plain"


# ---------------------------------------------------------------------------------------------------- values

def is_label(value: Any) -> bool:
    """Text that is not a number, or a date: what headers and row labels are made of."""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return True
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        float(value.replace(",", "").replace("%", "").strip())
    except ValueError:
        return True
    return False


def text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime.datetime) and value.time() == datetime.time():
        return value.date().isoformat()
    return " ".join(str(value).split())


def plain(name: str) -> str:
    return " ".join(str(name).split()).casefold()
