from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass(frozen=True)
class Cell:
    row: int  # 1-indexed
    col: int  # 1-indexed

    def __repr__(self) -> str:
        return f"Cell(row={self.row}, col={self.col})"

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

    def contains(self, cell: Cell) -> bool:
        return (self.start_row <= cell.row <= self.end_row and
                self.start_col <= cell.col <= self.end_col)

    def intersection(self, other: CellRange) -> Optional[CellRange]:
        """Compute the intersection of two ranges. Returns None if they do not intersect."""
        if self.sheet and other.sheet and self.sheet != other.sheet:
            return None
        
        start_row = max(self.start_row, other.start_row)
        end_row = min(self.end_row, other.end_row)
        start_col = max(self.start_col, other.start_col)
        end_col = min(self.end_col, other.end_col)

        if start_row <= end_row and start_col <= end_col:
            return CellRange(start_row, start_col, end_row, end_col, self.sheet or other.sheet)
        return None

    def union(self, other: CellRange) -> List[CellRange]:
        """
        Merge two ranges if they overlap or are adjacent and share the same dimension/orientation.
        Otherwise returns both.
        """
        if self.sheet and other.sheet and self.sheet != other.sheet:
            return [self, other]

        # Check if they share the exact same row bounds
        if self.start_row == other.start_row and self.end_row == other.end_row:
            # Overlap or adjacent columns
            if max(self.start_col, other.start_col) <= min(self.end_col, other.end_col) + 1:
                return [CellRange(
                    self.start_row,
                    min(self.start_col, other.start_col),
                    self.end_row,
                    max(self.end_col, other.end_col),
                    self.sheet or other.sheet
                )]

        # Check if they share the exact same column bounds
        if self.start_col == other.start_col and self.end_col == other.end_col:
            # Overlap or adjacent rows
            if max(self.start_row, other.start_row) <= min(self.end_row, other.end_row) + 1:
                return [CellRange(
                    min(self.start_row, other.start_row),
                    self.start_col,
                    max(self.end_row, other.end_row),
                    self.end_col,
                    self.sheet or other.sheet
                )]

        return [self, other]

    def difference(self, other: CellRange) -> List[CellRange]:
        """
        Subtract the other range from this range, returning a list of remaining sub-ranges.
        """
        intersect = self.intersection(other)
        if not intersect:
            return [self]

        # If the intersection covers self completely
        if (intersect.start_row == self.start_row and intersect.end_row == self.end_row and
                intersect.start_col == self.start_col and intersect.end_col == self.end_col):
            return []

        sub_ranges = []

        # 1D column-wise difference if they have the same row bounds
        if self.start_row == other.start_row and self.end_row == other.end_row:
            if self.start_col < intersect.start_col:
                sub_ranges.append(CellRange(self.start_row, self.start_col, self.end_row, intersect.start_col - 1, self.sheet))
            if intersect.end_col < self.end_col:
                sub_ranges.append(CellRange(self.start_row, intersect.end_col + 1, self.end_row, self.end_col, self.sheet))
            return sub_ranges

        # 1D row-wise difference if they have the same col bounds
        if self.start_col == other.start_col and self.end_col == other.end_col:
            if self.start_row < intersect.start_row:
                sub_ranges.append(CellRange(self.start_row, self.start_col, intersect.start_row - 1, self.end_col, self.sheet))
            if intersect.end_row < self.end_row:
                sub_ranges.append(CellRange(intersect.end_row + 1, self.start_col, self.end_row, self.end_col, self.sheet))
            return sub_ranges

        # General 2D rectangle difference (split into 4 possible rects)
        # Top part
        if self.start_row < intersect.start_row:
            sub_ranges.append(CellRange(self.start_row, self.start_col, intersect.start_row - 1, self.end_col, self.sheet))
        # Bottom part
        if intersect.end_row < self.end_row:
            sub_ranges.append(CellRange(intersect.end_row + 1, self.start_col, self.end_row, self.end_col, self.sheet))
        # Left part
        if self.start_col < intersect.start_col:
            sub_ranges.append(CellRange(intersect.start_row, self.start_col, intersect.end_row, intersect.start_col - 1, self.sheet))
        # Right part
        if intersect.end_col < self.end_col:
            sub_ranges.append(CellRange(intersect.start_row, intersect.end_col + 1, intersect.end_row, self.end_col, self.sheet))

        return sub_ranges


@dataclass(frozen=True)
class AxisSelection:
    """
    Sparse selection along one table axis.

    `axis="row"` means positions are absolute worksheet row numbers.
    `axis="col"` means positions are absolute worksheet column numbers.
    """
    axis: str
    positions: Tuple[int, ...]
    sheet: str = ""
    source_range: Optional[CellRange] = None

    def __post_init__(self):
        if self.axis not in {"row", "col"}:
            raise ValueError("AxisSelection.axis must be 'row' or 'col'.")
        normalized = tuple(sorted({int(pos) for pos in self.positions}))
        object.__setattr__(self, "positions", normalized)

    def __repr__(self) -> str:
        sheet_prefix = f"'{self.sheet}'!" if self.sheet else ""
        return f"AxisSelection({sheet_prefix}{self.axis}s={list(self.positions)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AxisSelection):
            return False
        return self.axis == other.axis and self.positions == other.positions and self.sheet == other.sheet

    def __hash__(self) -> int:
        return hash((self.axis, self.positions, self.sheet))

    def intersection(self, other: AxisSelection) -> AxisSelection:
        self._assert_compatible(other)
        positions = tuple(sorted(set(self.positions).intersection(other.positions)))
        return AxisSelection(self.axis, positions, self.sheet or other.sheet, self.source_range or other.source_range)

    def union(self, other: AxisSelection) -> AxisSelection:
        self._assert_compatible(other)
        positions = tuple(sorted(set(self.positions).union(other.positions)))
        return AxisSelection(self.axis, positions, self.sheet or other.sheet, self.source_range or other.source_range)

    def difference(self, other: AxisSelection) -> AxisSelection:
        self._assert_compatible(other)
        positions = tuple(sorted(set(self.positions).difference(other.positions)))
        return AxisSelection(self.axis, positions, self.sheet or other.sheet, self.source_range or other.source_range)

    def to_ranges(self, target_range: CellRange) -> List[CellRange]:
        """Project the sparse selection onto a target field/range."""
        if self.sheet and target_range.sheet and self.sheet != target_range.sheet:
            return []
        sheet = target_range.sheet or self.sheet
        ranges: List[CellRange] = []
        if self.axis == "row":
            for row in self.positions:
                if target_range.start_row <= row <= target_range.end_row:
                    ranges.append(CellRange(row, target_range.start_col, row, target_range.end_col, sheet))
        else:
            for col in self.positions:
                if target_range.start_col <= col <= target_range.end_col:
                    ranges.append(CellRange(target_range.start_row, col, target_range.end_row, col, sheet))
        return ranges

    def _assert_compatible(self, other: AxisSelection) -> None:
        if self.axis != other.axis:
            raise ValueError(f"Cannot combine {self.axis!r} selection with {other.axis!r} selection.")
        if self.sheet and other.sheet and self.sheet != other.sheet:
            raise ValueError(f"Cannot combine selections from different sheets: {self.sheet!r}, {other.sheet!r}.")
