"""What a cell means, found by walking from it up to its column headers and left to its row labels.

The walk reads the layout in code (see `HeaderWalker.headers`):
- column headers: the label cells above the cell's data, walking up; a row of numbers typed plain above
  formatted data is a header row too; when the header right above the data is merged across several
  columns, its span is reported ("spans"): those columns all share the label.
- row labels: the label cells left of the cell (a code written as text, such as '3' or '01001', labels its row too;
  right of the label columns, the row's labels there after the nearest label beyond them), and keys written once
  above and carried down (a location in column A, then empty or merged cells below it).
- sections: label-only rows above the cell, nearest first, then each parent (a label further left, less
  indented, or bold over plain).
- block: the run of rows with data shaped like the cell's row, and why it starts and ends.
- legend: a label cell, not bold, that alone among labels has the cell's (non-white) fill while numbers have it too:
  cells marked with that colour belong to that label.
- uncertain: why the walk may be wrong (no header, or no label while cells left of the cell are filled);
  `SheetTools.headers` then asks a vision model about an image.

The header rows are found from the data: the body starts below the lowest header row above the first number of the
cell's row (or of the nearest row with a number), and a cell of the body takes its column's headers from there. The
walk up counts a merged header once, passes section rows between the data and its header, and takes a row of numbers
written as text, above numbers, for a header row; a run of section rows of one level is one section. A placeholder in
the data (a mark without letter or digit, such as `-` or `..`, or an Excel error value) stands for a number. Where no
number is near the cell, a label repeated by the cell above or below it is data.

A sheet's layout is read once. Data columns are mostly filled and mostly numbers; key columns are the
label columns left of them, and a column of numbers written once per group of rows is a key column too. A block ends at a row with no data (a section label or a blank row), a total
computed by an aggregate formula, a row whose label cell is formatted differently, or a row whose data
columns mostly change kind. An empty cell inside the data never ends a block. A format marks structure
only when it is rare next to the other, and a fill is ignored where rows alternate two fills (striped for
reading): only bold and indent then count.
"""

from __future__ import annotations

import colorsys
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
NEARBY_ROWS = 20  # rows searched for a number when a cell's own row has none
PLAIN_FORMAT = "General"
NUMBER_HEADER_LOOKDOWN = 10  # rows below a number searched for the next number of its column
KEY_FILLED_SHARE = 0.7  # a key column written once per group is filled in under this share of the body
MIN_LEGEND_CELLS = 1  # numbers sharing a label's fill for that label to be a legend
COLOUR_HUES = ((15, "red"), (45, "orange"), (70, "yellow"), (170, "green"), (200, "cyan"), (260, "blue"), (300, "purple"), (345, "pink"), (361, "red"))
WHITE_FILLS = {"theme 0", "FFFFFFFF", "00FFFFFF"}  # a white fill paints the background, it marks nothing
PLACEHOLDER = re.compile(r"^(\W+|#(N/A|REF!|DIV/0!|VALUE!|NAME\?|NUM!|NULL!|SPILL!|CALC!))$", re.IGNORECASE)
AGGREGATE = re.compile(r"^=\s*(SUM|SUBTOTAL|AVERAGE|COUNTA?|MIN|MAX)\s*\(", re.IGNORECASE)
RANGE = re.compile(r"\$?([A-Z]{1,3})\$?(\d+)\s*:\s*\$?([A-Z]{1,3})\$?(\d+)")


class HeaderWalker:
    """Walks from cells to their headers, labels and sections; `tools` is the SheetTools it serves."""

    def __init__(self, tools: Any, formula_path: Path | None = None):
        self.tools = tools
        self._formula_path = formula_path  # the workbook file, read again with formulas for total rows
        self._formula_workbook: Any = None
        self._sections: dict[str, list[dict[str, Any]]] = {}
        self._layouts: dict[tuple[str, int], Layout] = {}
        self._legends: dict[str, dict[str, str]] = {}
        self._number_rows: dict[tuple[str, int], bool] = {}
        self._text_number_rows: dict[tuple[str, int], bool] = {}

    def headers(self, ws: Any, row: int, col: int) -> dict[str, Any]:
        """What the walk finds for one cell (see the module docstring)."""
        column_headers, top, header_rows = self.column_headers(ws, row, col)
        left = [self.tools.value(row, c, ws.title) for c in range(col - 1, 0, -1)]
        row_labels = [text_of(value) for value in left if is_row_label(value)][:ROW_LABELS]
        layout = self.layout(ws, top)
        key_columns = layout.key_columns if top < row else []
        if key_columns and col > key_columns[-1]:
            # Right of the label columns, the row's labels there after the nearest label beyond them: the labels
            # nearest a cell far right are often text data of other columns.
            nearest = next((value for value in left[: col - 1 - key_columns[-1]] if is_row_label(value)), None)
            keyed = [self.tools.value(row, c, ws.title) for c in reversed(key_columns)]
            row_labels = ([text_of(nearest)] if nearest is not None else []) + [text_of(v) for v in keyed if is_row_label(v)]
            row_labels = row_labels[:ROW_LABELS]
        found_block = block(layout, row) if top < row <= len(layout.values) else None
        found_keys = [
            key for key in self.keys(ws, row, col, top, found_block) if key["from"] != f"{key['column']}{row}"
        ]
        group = self.section_group(ws, row, top)
        uncertain = []
        if not column_headers:
            uncertain.append("no column header found above the cell")
        if not row_labels and not found_keys and any(_kind(v) != "." and not is_placeholder(v) for v in left):
            uncertain.append("no row label found left of the cell")
        found = {
            "cell": cell_to_a1(row, col),
            "sheet": ws.title,
            "value": self.tools.value(row, col, ws.title),
            "row_labels": list(reversed(row_labels)),
            "keys": found_keys,
            "column_headers": list(reversed(column_headers)),
            "spans": self.header_spans(ws, header_rows, col),
            "sections": group["sections"],
            "siblings": group["siblings"],
            "block": {key: found_block[key] for key in ("rows", "starts", "ends")} if found_block else None,
            "uncertain": uncertain,
        }
        legend = self.legends(ws).get(cell_format(ws.cell(row, col))[1])
        if legend:
            found["legend"] = legend
        return found

    def column_headers(self, ws: Any, row: int, col: int) -> tuple[list[str], int, list[int]]:
        """Header cells above a cell (nearest first), the header row its body starts below, and the header rows."""
        top = self.data_top(ws, row)
        if top:  # a cell of the body takes the headers of its column; a cell of the header band, those above it
            headers, _, rows = self.headers_above(ws, top + 1 if top < row else row, col)
            return headers, top, rows
        # No number near the cell: a label repeated in its column is data, and a cell of such data is in the body,
        # which starts below its lowest header.
        headers, highest, rows = self.headers_above(ws, row, col, repeats_are_data=True)
        return headers, (rows[0] if rows and self.repeated(ws, row, col) else highest), rows

    def data_top(self, ws: Any, row: int) -> int:
        """The lowest header row above the data of a row, or 0: found walking up from the row's first number (or the
        first number of the nearest row below or above that has one), past numbers and section rows."""
        last_row = self.tools.size(ws)[0]
        nearby = [*range(row + 1, min(last_row, row + NEARBY_ROWS) + 1), *range(row - 1, max(0, row - NEARBY_ROWS), -1)]
        for r in [row, *nearby]:
            col = self.first_number(ws, r)
            if col:
                rows = self.headers_above(ws, r, col, past_sections=True)[2]
                return rows[0] if rows else 0
        return 0

    def first_number(self, ws: Any, row: int) -> int | None:
        """The column of the first number of a row that is data (not a number of a header row)."""
        for c in range(1, min(self.tools.size(ws)[1], SCAN_COLUMNS) + 1):
            value = self.tools.value(row, c, ws.title)
            if _is_number(value) and not self.number_header(ws, row, value):
                return c
        return None

    def _same_merged(self, ws: Any, row: int, other: int, col: int) -> bool:
        """Whether two cells of a column belong to one merged range."""
        anchors = self.tools._merged_map(ws)
        return anchors.get((row, col), (row, col)) == anchors.get((other, col), (other, col))

    def repeated(self, ws: Any, row: int, col: int) -> bool:
        """Whether a label cell has the same value as the cell above or below it (raw cells: a merged range is not)."""
        value = ws.cell(row, col).value
        return is_label(value) and value in (ws.cell(row - 1, col).value if row > 1 else None, ws.cell(row + 1, col).value)

    def text_number_header(self, ws: Any, row: int, value: Any) -> bool:
        """A number written as text in a header row: a row of at least two filled cells and no numbers, with numbers
        somewhere below it (in a table of numbers written as text, such rows are data)."""
        if not isinstance(value, str) or is_label(value) or not value.strip():
            return False
        key = (ws.title, row)
        if key not in self._text_number_rows:
            values = [v for v in (ws.cell(row, c).value for c in range(1, min(self.tools.size(ws)[1], SCAN_COLUMNS) + 1))
                      if _kind(v) != "."]
            last_row, last_col = self.tools.size(ws)
            below = ws.iter_rows(min_row=row + 1, max_row=last_row, max_col=min(last_col, SCAN_COLUMNS), values_only=True)
            self._text_number_rows[key] = (len(values) >= 2 and not any(_is_number(v) for v in values)
                                           and any(_is_number(v) for line in below for v in line))
        return self._text_number_rows[key]

    def headers_above(
        self, ws: Any, row: int, col: int, past_sections: bool = False, repeats_are_data: bool = False,
    ) -> tuple[list[str], int, list[int]]:
        """Header cells above a cell (nearest first), the top header row, and the rows of the headers.

        Label cells, and numbers of a header row (`number_header` and `text_number_header`), are collected walking up
        past the data (placeholders included), up to the next number; a merged header counts once. `past_sections` passes
        the section rows met before the first header (rows between data and its header); with `repeats_are_data` a
        label repeated by the cell above or below it is data.
        """
        sections = {section["row"] for section in self.section_rows(ws)} if past_sections else set()
        headers: list[str] = []
        rows: list[int] = []
        for r in range(row - 1, 0, -1):
            value = self.tools.value(r, col, ws.title)
            if not headers and r in sections and ws.cell(r, col).value is None:
                continue  # a section row merged across the column, between the data and its header
            if rows and self._same_merged(ws, r, rows[-1], col):
                continue  # the same merged header again
            label = is_label(value) and not is_placeholder(value) and not (repeats_are_data and self.repeated(ws, r, col))
            if label or self.number_header(ws, r, value) or self.text_number_header(ws, r, value):
                headers.append(text_of(value))
                rows.append(r)
                if len(headers) == HEADER_LEVELS:
                    break
            elif value is not None and headers:
                break
        return headers, (rows[-1] if rows else 0), rows

    def number_header(self, ws: Any, row: int, value: Any) -> bool:
        """A number in a header row: the row's numbers (at least two) are the first numbers of their columns, each written plain
        (General) while the next number below it is formatted: headers are typed as labels, data is formatted."""
        if _kind(value) != "n" or hasattr(value, "year"):
            return False
        key = (ws.title, row)
        if key not in self._number_rows:
            pairs = []
            last_row, last_col = self.tools.size(ws)
            for c in range(1, min(last_col, SCAN_COLUMNS) + 1):
                if _kind(v := ws.cell(row, c).value) != "n" or hasattr(v, "year"):
                    continue
                below = next((ws.cell(r, c) for r in range(row + 1, min(last_row, row + NUMBER_HEADER_LOOKDOWN) + 1)
                              if _kind(ws.cell(r, c).value) == "n"), None)
                if any(_kind(ws.cell(r, c).value) == "n" for r in range(1, row)):
                    pairs.append(False)  # a number above it: the row is in the body, not the header band
                elif below is not None:
                    pairs.append(ws.cell(row, c).number_format == PLAIN_FORMAT != below.number_format)
            self._number_rows[key] = len(pairs) >= 2 and all(pairs)
        return self._number_rows[key]

    def header_spans(self, ws: Any, rows: list[int], col: int) -> list[str]:
        """The nearest header when it is merged across several columns, e.g. `Sales: B3:F3 (5 columns)`."""
        merged = self.span_range(ws, rows, col)
        if merged is None:
            return []
        return [f"{text_of(self.tools.value(rows[0], col, ws.title))}: {merged.coord} ({merged.max_col - merged.min_col + 1} columns)"]

    def span_range(self, ws: Any, rows: list[int], col: int) -> Any:
        """The merged range of the header right above the data, when its columns all share that header.

        Only the header right above the data counts, only beside other headers in its row, and only when no row
        above tells its columns apart: they then all share its label (replicates, a group read as one).
        """
        if not rows:
            return None
        r = rows[0]
        merged = next((m for m in ws.merged_cells.ranges if m.min_row <= r <= m.max_row and m.min_col <= col <= m.max_col), None)
        if merged is None or merged.max_col == merged.min_col:
            return None
        if not any(cell.value is not None for cell in ws[r] if not merged.min_col <= cell.column <= merged.max_col):
            return None  # a section or title row merged across the table, not the header of a column group
        for other in range(1, r):
            values = {text_of(ws.cell(other, c).value) for c in range(merged.min_col, merged.max_col + 1) if ws.cell(other, c).value is not None}
            if len(values) > 1:
                return None  # another header row tells these columns apart
        return merged

    def legends(self, ws: Any) -> dict[str, str]:
        """Fill → its legend, e.g. `Transformed Luminal (J4, fill FFFF0000)` (see `legend_cells`)."""
        if ws.title not in self._legends:
            self._legends[ws.title] = {
                fill: f"{text_of(cell.value)} ({cell.coordinate}, fill {fill})" for fill, cell in legend_cells(ws).items()
            }
        return self._legends[ws.title]

    def section_group(self, ws: Any, row: int, top: int) -> dict[str, Any]:
        """The section rows above a row (its own, then each parent) and the siblings of its own, with row spans."""
        sections = _stacked([section for section in self.section_rows(ws) if section["row"] > top])
        last_row = self.tools.size(ws)[0]

        def span(start: int, level: tuple[int, ...]) -> str:
            end = next((s["row"] for s in sections if s["row"] > start and s["level"] <= level), last_row + 1)
            return f"{start + 1}-{end - 1}"

        def shown(section: dict[str, Any]) -> dict[str, Any]:
            return {"label": section["label"], "row": section["row"], "rows": span(section["last"], section["level"])}

        above = [section for section in sections if section["row"] < row]
        if not above:
            return {"sections": [], "siblings": []}
        own = above[-1]
        parents = []
        for section in reversed(above[:-1]):
            if section["level"] < (parents[-1]["level"] if parents else own["level"]):
                parents.append(section)
        parent = parents[0] if parents else None
        parent_end = int(span(parent["row"], parent["level"]).split("-")[1]) if parent else last_row
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
            last_row, last_col = self.tools.size(ws)
            for r, values in enumerate(ws.iter_rows(max_row=last_row, max_col=min(last_col, SCAN_COLUMNS), values_only=True), start=1):
                filled = [(c, value) for c, value in enumerate(values, start=1) if value not in (None, "")]
                if not filled or len(filled) > SECTION_MAX_CELLS or not all(
                    is_label(value) and not is_placeholder(value) for _, value in filled
                ):
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
            self._layouts[(ws.title, top)] = read_layout(ws, top, self.tools.size(ws), self.formula_sheet(ws.title))
        return self._layouts[(ws.title, top)]

    def formula_sheet(self, title: str) -> Any:
        """The sheet read with formulas instead of their values, or None without the workbook file."""
        if self._formula_path is not None and self._formula_workbook is None:
            self._formula_workbook = openpyxl.load_workbook(self._formula_path)
        return self._formula_workbook[title] if self._formula_workbook is not None else None


def _stacked(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Section rows with consecutive section rows of one level joined into one, its label the rows' labels."""
    joined: list[dict[str, Any]] = []
    for section in sections:
        last = joined[-1] if joined else None
        if last and section["row"] == last["last"] + 1 and section["level"] == last["level"]:
            last.update(label=f"{last['label']} / {section['label']}", last=section["row"])
        else:
            joined.append({**section, "last": section["row"]})
    return joined


def headers_line(found: dict[str, Any]) -> str:
    """One line for a walk, e.g. `D8 = 35 | row: Alpha | column: 2019 | section: South (row 7)`."""
    parts = [f"{found['cell']} = {text_of(found['value']) or '(empty)'}"]
    if found["row_labels"]:
        parts.append("row: " + " > ".join(found["row_labels"]))
    for key in found["keys"]:
        parts.append(f"key: {key['header']} = {text_of(key['value'])} (from {key['from']})")
    if found["column_headers"]:
        parts.append("column: " + " > ".join(found["column_headers"]))
    for span in found.get("spans", []):
        parts.append(f"header spans {span}")
    if found.get("legend"):
        parts.append(f"legend: {found['legend']}")
    for section in found["sections"]:
        parts.append(f"section: {section['label']} (row {section['row']})")
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

    def kind(self, row: int, col: int) -> str:
        """`_kind` of a cell, a placeholder counting as a number."""
        value = self.value(row, col)
        return "n" if is_placeholder(value) else _kind(value)

    def label_column(self, row: int) -> int | None:
        """The first label column filled in a row."""
        columns = self.key_columns or range(1, len(self.values[row - 1]) + 1)
        return next((c for c in columns if _kind(self.value(row, c)) != "."), None)


def read_layout(ws: Any, top: int, size: tuple[int, int], formula_ws: Any = None) -> Layout:
    """The layout of a sheet of `size` (rows, columns) whose header ends at row `top`; `formula_ws` is the same
    sheet read with formulas. A placeholder in the data counts as a number."""
    width = min(size[1], SCAN_COLUMNS)
    values = [list(row) for row in ws.iter_rows(max_row=size[0], max_col=width, values_only=True)]
    body = values[top:]
    data_columns = []
    for c in range(1, width + 1):
        filled = [row[c - 1] for row in body if _kind(row[c - 1]) != "."]
        numbers = [value for value in filled if _kind(value) == "n" or is_placeholder(value)]
        if _carried_key(ws, top, c, len(body)):
            continue  # a key written once per group of rows, in its own number format: not data
        if body and filled and len(numbers) >= DATA_NUMBER_SHARE * len(filled) and len(filled) >= DATA_FILLED_SHARE * len(body):
            data_columns.append(c)
    first_data = data_columns[0] if data_columns else width + 1
    key_columns = [c for c in range(1, first_data) if any(_kind(row[c - 1]) != "." for row in body)]
    layout = Layout(ws, top, values, data_columns=data_columns, key_columns=key_columns)
    if formula_ws is not None:
        layout.totals = _aggregate_rows(formula_ws, width)
    return layout


def legend_cells(ws: Any) -> dict[str, Any]:
    """Fill → the cell of its legend: the one plain label with that fill when numbers have it too (white fills excluded)."""
    labels: dict[str, list[Any]] = {}
    numbers: dict[str, int] = {}
    for line in ws.iter_rows():
        for cell in line:
            fill = cell_format(cell)[1] if cell.value is not None else None
            if fill is None or fill in WHITE_FILLS:
                continue
            if is_label(cell.value):
                if not (cell.font and cell.font.b):
                    labels.setdefault(fill, []).append(cell)
            elif isinstance(cell.value, (int, float)):
                numbers[fill] = numbers.get(fill, 0) + 1
    return {fill: cells[0] for fill, cells in labels.items() if len(cells) == 1 and numbers.get(fill, 0) >= MIN_LEGEND_CELLS}


def _carried_key(ws: Any, top: int, col: int, body_rows: int) -> bool:
    """A column of numbers written once per group of rows (section labels may sit in it too): filled in under
    KEY_FILLED_SHARE of the body, each number beside a label of the column right of it, which is filled in more rows:
    the column the groups are written out in."""
    rows = range(top + 1, top + body_rows + 1)
    filled = [r for r in rows if _kind(ws.cell(r, col).value) != "."]
    if len(filled) < 2 or len(filled) >= KEY_FILLED_SHARE * body_rows:
        return False
    numbers = [r for r in filled if _kind(ws.cell(r, col).value) == "n" and not hasattr(ws.cell(r, col).value, "year")]
    if len(numbers) < 2:
        return False
    labels = [r for r in rows if is_label(ws.cell(r, col + 1).value)]
    return len(labels) > len(numbers) and all(r in labels for r in numbers)


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
    kinds = [layout.kind(r, c) for c in columns]
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
            (layout.kind(r, c), layout.kind(row, c))
            for c in layout.data_columns
            if layout.kind(r, c) != "." and layout.kind(row, c) != "."
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


def font_colour(cell: Any) -> str | None:
    """The font colour of a cell when it is not the default black."""
    color = cell.font.color if cell.font is not None else None
    named = None if color is None else {"rgb": color.rgb, "theme": f"theme {color.theme}", "indexed": f"indexed {color.indexed}"}.get(color.type)
    return named if named not in ("FF000000", "theme 1") else None


def colour_name(value: Any) -> str | None:
    """A basic colour word for an ARGB or RGB hex value ("FFFF0000" -> "red"); None for theme and indexed colours."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8}", value):
        return None
    red, green, blue = (int(value[-6:][i:i + 2], 16) / 255 for i in (0, 2, 4))
    hue, saturation, brightness = colorsys.rgb_to_hsv(red, green, blue)
    if brightness < 0.2:
        return "black"
    if saturation < 0.15:
        return "white" if brightness > 0.9 else "grey"
    return next(name for limit, name in COLOUR_HUES if hue * 360 < limit)


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


def is_placeholder(value: Any) -> bool:
    """A mark written in place of a number: text without letter or digit (`-`, `..`, `*`), or an Excel error value."""
    return isinstance(value, str) and bool(value.strip()) and bool(PLACEHOLDER.match(value.strip()))


def is_row_label(value: Any) -> bool:
    """A label of a row: a label, or a code written as text ('3', '01001'), not a placeholder."""
    return (is_label(value) or (isinstance(value, str) and bool(value.strip()))) and not is_placeholder(value)


def _is_number(value: Any) -> bool:
    return _kind(value) == "n" and not hasattr(value, "year")


def text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime.datetime) and value.time() == datetime.time():
        return value.date().isoformat()
    return " ".join(str(value).split())


def plain(name: str) -> str:
    return " ".join(str(name).split()).casefold()
