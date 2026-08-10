from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from typing import Any


_WORKBOOK_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}


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


def has_workbook_sources(sample: EvalSample) -> bool:
    """Return whether every configured source path is a supported workbook."""
    paths = [
        part.strip()
        for part in str(sample.table_path or "").split(";")
        if part.strip()
    ]
    return bool(paths) and all(
        PureWindowsPath(path).suffix.lower() in _WORKBOOK_SUFFIXES
        for path in paths
    )


__all__ = ["EvalSample", "has_workbook_sources"]
