from __future__ import annotations

from pathlib import Path

from TableAgent.pipeline.sample import EvalSample
from TableAgent.utils.paths import safe_name


def original_workbook_for_sample(sample: EvalSample, current: Path) -> Path | None:
    """Resolve a structure workbook back to the original workbook used for QA."""
    raw = sample.raw if isinstance(sample.raw, dict) else {}
    path_map = raw.get("original_workbook_paths")
    if isinstance(path_map, dict):
        mapped = path_map.get(str(current.resolve()))
        if mapped:
            original = Path(str(mapped))
            if original.is_file():
                return original

    values = [
        Path(value.strip())
        for value in str(sample.table_path or "").split(";")
        if value.strip()
    ]
    current_name = safe_name(current.name)
    for value in values:
        if value.is_file() and (
            value.name == current.name or safe_name(value.name) == current_name
        ):
            return value
    return values[0] if len(values) == 1 and values[0].is_file() else None
