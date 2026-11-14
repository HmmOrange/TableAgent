"""A1 addresses: parse a cell or range ("B3", "'Data'!B3:F9") and write a cell's address."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CellRange:
    start_row: int  # 1-indexed
    start_col: int  # 1-indexed
    end_row: int    # 1-indexed
    end_col: int    # 1-indexed
    sheet: str = ""

    def __post_init__(self):
        if self.start_row > self.end_row:
            raise ValueError(f"start_row ({self.start_row}) cannot be greater than end_row ({self.end_row})")
        if self.start_col > self.end_col:
            raise ValueError(f"start_col ({self.start_col}) cannot be greater than end_col ({self.end_col})")

    def __repr__(self) -> str:
        sheet_prefix = f"'{self.sheet}'!" if self.sheet else ""
        return f"CellRange({sheet_prefix}{self.start_row}:{self.start_col} to {self.end_row}:{self.end_col})"


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
    if "!" in range_str:
        # Excel-style "Sheet1!A1:B2" or "'Table 1'!A1"; the sheet named in the address wins.
        sheet_part, range_str = range_str.rsplit("!", 1)
        sheet = sheet_part.strip().strip("'").replace("''", "'") or sheet
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
