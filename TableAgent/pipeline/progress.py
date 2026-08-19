"""Formatting for pipeline progress events."""

from __future__ import annotations

from typing import Any


_LABELS = {
    "prepare": "prepare",
    "prepare_extract": "prepare:extract",
    "prepare_metadata": "prepare:metadata",
    "prepare_cached": "prepare:cached",
    "prepare_error": "prepare:error",
    "prepare_layout": "prepare:layout",
    "prepare_done": "prepare:done",
    "retrieval": "retrieve",
    "rerank": "rerank",
    "render": "render",
    "layout": "layout",
    "verify": "verify",
    "structure_done": "structure:done",
    "directions": "directions",
    "qa": "qa",
    "answer": "answer",
    "done": "done",
}

_LAYOUT_FIELDS = (
    ("range", "range"),
    ("iteration", "iter"),
    ("direction", "dir"),
    ("workbook", "book"),
    ("sheet", "sheet"),
    ("sample", "sample"),
)

_DEFAULT_FIELDS = (
    ("sample", "sample"),
    ("workbook", "book"),
    ("sheet", "sheet"),
    ("table", "table"),
    ("range", "range"),
    ("iteration", "iter"),
    ("direction", "dir"),
    ("status", "status"),
)


def format_progress(stage: str, fields: dict[str, Any]) -> str:
    parts = [_LABELS.get(stage, stage)]
    ordered_fields = (
        _LAYOUT_FIELDS
        if stage in {"prepare_layout", "prepare_done", "render", "layout", "verify"}
        else _DEFAULT_FIELDS
    )
    for key, label in ordered_fields:
        value = fields.get(key)
        if value is not None and value != "":
            parts.append(f"{label}={value}")
    return " | ".join(parts)
