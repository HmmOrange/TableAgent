from __future__ import annotations
import re
from typing import Any, List, Tuple
from TableAgent.schema.range import CellRange

def col_name_to_num(col_str: str) -> int:
    col_str = col_str.upper()
    col_num = 0
    for char in col_str:
        col_num = col_num * 26 + (ord(char) - ord('A') + 1)
    return col_num

def col_num_to_name(col_num: int) -> str:
    name = ""
    while col_num > 0:
        col_num, remainder = divmod(col_num - 1, 26)
        name = chr(65 + remainder) + name
    return name

def parse_a1_cell(cell_str: str) -> Tuple[int, int]:
    m = re.match(r"^([A-Za-z]+)(\d+)$", cell_str.strip())
    if not m:
        raise ValueError(f"Invalid A1 cell format: {cell_str}")
    col_str, row_str = m.groups()
    return int(row_str), col_name_to_num(col_str)

def parse_a1_range(range_str: str, sheet: str = "") -> CellRange:
    range_str = range_str.strip()
    if ":" in range_str:
        parts = range_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid A1 range format: {range_str}")
        start_row, start_col = parse_a1_cell(parts[0])
        end_row, end_col = parse_a1_cell(parts[1])
        # Ensure correct ordering in case user provided bottom-right to top-left
        min_row = min(start_row, end_row)
        max_row = max(start_row, end_row)
        min_col = min(start_col, end_col)
        max_col = max(start_col, end_col)
        return CellRange(min_row, min_col, max_row, max_col, sheet)
    else:
        row, col = parse_a1_cell(range_str)
        return CellRange(row, col, row, col, sheet)

def cell_to_a1(row: int, col: int) -> str:
    return f"{col_num_to_name(col)}{row}"

def range_to_a1(r: CellRange) -> str:
    if r.start_row == r.end_row and r.start_col == r.end_col:
        return cell_to_a1(r.start_row, r.start_col)
    return f"{cell_to_a1(r.start_row, r.start_col)}:{cell_to_a1(r.end_row, r.end_col)}"

def read_excel_range(
    sheet_obj: Any,
    cell_range: CellRange,
    *,
    expand_merged: bool = False,
) -> List[List[Any]]:
    """Read cell values from an openpyxl sheet object for a given CellRange."""
    values = []
    for row in sheet_obj.iter_rows(
        min_row=cell_range.start_row,
        max_row=cell_range.end_row,
        min_col=cell_range.start_col,
        max_col=cell_range.end_col
    ):
        values.append([cell.value for cell in row])
    if not expand_merged or not values or not hasattr(sheet_obj, "merged_cells"):
        return values

    # openpyxl stores a merged range's value only in its top-left cell. Expand
    # that value in memory so QA sees the same grouping users see in Excel.
    # Ordinary blank cells outside real merged ranges remain untouched.
    for merged in sheet_obj.merged_cells.ranges:
        start_row = max(cell_range.start_row, merged.min_row)
        end_row = min(cell_range.end_row, merged.max_row)
        start_col = max(cell_range.start_col, merged.min_col)
        end_col = min(cell_range.end_col, merged.max_col)
        if start_row > end_row or start_col > end_col:
            continue
        anchor_value = sheet_obj.cell(row=merged.min_row, column=merged.min_col).value
        for row in range(start_row, end_row + 1):
            for column in range(start_col, end_col + 1):
                values[row - cell_range.start_row][column - cell_range.start_col] = anchor_value
    return values
