from dataclasses import dataclass
from typing import Optional

from TableAgent.domain.ranges import CellRange


@dataclass
class StructureGroup:
    id: str
    label: str
    group_range: Optional[CellRange]
    data_range: Optional[CellRange]
    axis: str
    description: str
