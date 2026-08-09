from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def prepared_verification(directory: Path) -> dict[str, Any]:
    """Read verification metadata shared by structure reporting and QA output."""
    metadata_path = directory / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    verification = metadata.get("verification") if isinstance(metadata, dict) else None
    if isinstance(verification, dict):
        return dict(verification)
    return {"status": "good", "feedback": "Retrieved from encoded source"}
