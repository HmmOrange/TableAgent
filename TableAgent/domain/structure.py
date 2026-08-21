from dataclasses import dataclass, field
from typing import List, Optional

from TableAgent.domain.ranges import CellRange


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
            "Header("
            f"id={self.id!r}, label={self.label!r}, description={self.description!r}, "
            f"orientation={self.orientation!r}, header_range={self.header_range!r}, "
            f"data_range={self.data_range!r}, sub_headers={len(self.sub_headers)})"
        )
