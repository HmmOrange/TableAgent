from __future__ import annotations

from pathlib import Path
import yaml


def write_relations(output_dict: dict, output_path: str | Path) -> None:
    """
    Writes the relations dictionary to a YAML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(output_dict, f, sort_keys=False, allow_unicode=True)
