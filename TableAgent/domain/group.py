from dataclasses import dataclass
from typing import Optional

from TableAgent.domain.ranges import CellRange


@dataclass(frozen=True)
class StructureGroup:
    """A semantic row/column group declared in structure.yaml."""

    id: str
    label: str
    description: str
    axis: str
    group_range: Optional[CellRange]
    data_range: Optional[CellRange]

    def __repr__(self) -> str:
        return f"StructureGroup(id='{self.id}', label='{self.label}', axis='{self.axis}')"
