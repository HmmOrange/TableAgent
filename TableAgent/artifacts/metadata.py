from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import openpyxl

from .schema import SummaryGenerator


def build_workbook_metadata(
    source_path: Path,
    workbook_name: str,
    output_path: Path,
    *,
    schema_path: Path | None = None,
    summarizer: SummaryGenerator | None = None,
) -> dict[str, Any]:
    sheet_names, author, date_created, date_modified = _read_workbook_properties(source_path)
    description = ""
    if schema_path is not None and schema_path.is_file():
        if summarizer is None:
            raise ValueError("A summary LLM is required to describe an existing schema")
        description = summarizer.workbook_description(schema_path.read_text(encoding="utf-8"))
    payload = {
        "name": workbook_name,
        "description": description,
        "sheet_names": sheet_names,
        "author": author,
        "date_created": date_created,
        "date_modified": date_modified,
        "size_bytes": source_path.stat().st_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _read_workbook_properties(
    path: Path,
) -> tuple[list[str], str | None, str | None, str | None]:
    if path.suffix.lower() == ".xls":
        import xlrd

        workbook = xlrd.open_workbook(path, on_demand=True)
        try:
            return list(workbook.sheet_names()), _text_or_none(workbook.user_name), None, None
        finally:
            workbook.release_resources()

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        properties = workbook.properties
        return (
            list(workbook.sheetnames),
            _text_or_none(properties.creator),
            _iso_or_none(properties.created),
            _iso_or_none(properties.modified),
        )
    finally:
        workbook.close()


def _text_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _iso_or_none(value: Any) -> str | None:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return _text_or_none(value)


__all__ = ["build_workbook_metadata"]
