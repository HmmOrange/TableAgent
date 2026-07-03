from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from TableAgent.schema.range import CellRange

@dataclass
class Header:
    id: str
    label: str
    description: str
    orientation: str  # 'column', 'column_group', 'row', 'row_group'
    header_range: Optional[CellRange]
    data_range: Optional[CellRange]
    sub_headers: List[Header] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Header(id='{self.id}', label='{self.label}', orientation='{self.orientation}', sub_headers={len(self.sub_headers)})"
