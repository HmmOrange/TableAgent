from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
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

class BaseDataset(ABC):
    name: str

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def load_samples(self, *, limit: int = 0) -> list[EvalSample]:
        pass

    @abstractmethod
    def table_ids(self, samples: list[EvalSample]) -> list[str]:
        pass


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path

