from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalSample:
    index: int
    sample_id: str
    table_id: str
    table_content: str
    question: str
    answer: list[Any]
    sample_path: str = ""
    table_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
