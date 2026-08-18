from dataclasses import dataclass, field
from typing import List, Optional

from TableAgent.domain.ranges import CellRange


def _a1(cell_range: Optional[CellRange]) -> Optional[str]:
    if cell_range is None:
        return None
    start = _cell(cell_range.start_row, cell_range.start_col)
    if cell_range.start_row == cell_range.end_row and cell_range.start_col == cell_range.end_col:
        return start
    return f"{start}:{_cell(cell_range.end_row, cell_range.end_col)}"


def _cell(row: int, col: int) -> str:
    name = ""
    while col > 0:
        col, remainder = divmod(col - 1, 26)
        name = chr(65 + remainder) + name
    return f"{name}{row}"


@dataclass
class Header:
    id: str
    label: str
    description: str
    orientation: str  # 'column', 'column_group', 'row', 'row_group'
    header_range: Optional[CellRange]
    data_range: Optional[CellRange]
    sub_headers: List["Header"] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Header(id={self.id!r}, label={self.label!r}, "
            f"description={self.description!r}, orientation={self.orientation!r}, "
            f"data_range={_a1(self.data_range)!r}, sub_headers={len(self.sub_headers)})"
        )
