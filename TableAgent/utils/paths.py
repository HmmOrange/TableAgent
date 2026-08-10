import re
from pathlib import Path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)) or "item"


def display_path(path: Path) -> str:
    return str(path).replace("\\", "/")


__all__ = ["display_path", "safe_name"]
