from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal

@dataclass
class SubTask:
    id: str
    description: str
    layer: Literal["table_inspect", "inspect", "synthesis"]
    category: Literal["normal", "common_info"] = "normal"
    depends_on: List[str] = field(default_factory=list)
    status: Literal["pending", "running", "success", "failed"] = "pending"
    code_attempt: Optional[str] = None
    observation: Optional[str] = None
    assigned_agent: Optional[str] = None
    metadata: Optional[dict] = None

    def __repr__(self) -> str:
        return (
            f"SubTask(id='{self.id}', layer='{self.layer}', category='{self.category}', depends_on={self.depends_on}, "
            f"status='{self.status}', metadata={self.metadata})"
        )
