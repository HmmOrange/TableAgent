from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from datasets.base import EvalSample


@dataclass(frozen=True)
class WorkbookConversion:
    path: Path
    source_format: str
    sheet_names: list[str]


def sample_to_xlsx(sample: EvalSample, output_path: str | Path) -> WorkbookConversion:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    workbook.remove(workbook.active)
    source_format = _write_sample(workbook, sample)
    workbook.save(output)

    return WorkbookConversion(
        path=output,
        source_format=source_format,
        sheet_names=workbook.sheetnames,
    )


def _write_sample(workbook: Workbook, sample: EvalSample) -> str:
    raw_tables = sample.raw.get("tables")
    if isinstance(raw_tables, list) and raw_tables:
        tables = [_table_payload_to_rows(table) for table in raw_tables]
        _write_combined_tables(workbook, tables)
        for index, table in enumerate(tables, start=1):
            sheet = workbook.create_sheet(_sheet_title(f"Table {index}"))
            _write_rows(sheet, table.rows, table.merged_regions)
        return "json:tables"

    content = sample.table_content.strip()
    if content.startswith("<"):
        table = _html_to_rows(content)
        sheet = workbook.create_sheet(_sheet_title(sample.table_id or "Table"))
        _write_rows(sheet, table.rows, table.merged_regions)
        return "html"

    try:
        payload = json.loads(sample.table_content)
    except json.JSONDecodeError:
        rows = _plain_text_to_rows(sample.table_content)
        sheet = workbook.create_sheet(_sheet_title(sample.table_id or "Table"))
        _write_rows(sheet, rows, [])
        return "text"

    source_format = _write_json_payload(workbook, payload, title=sample.table_id or "Table")
    return source_format


@dataclass(frozen=True)
class _TableRows:
    rows: list[list[Any]]
    merged_regions: list[dict[str, int]]
    title: str = ""


def _table_payload_to_rows(payload: Any) -> _TableRows:
    if isinstance(payload, str):
        if "<table" in payload.lower():
            return _html_to_rows(payload)
        return _TableRows(_plain_text_to_rows(payload), [])
    if isinstance(payload, list):
        return _TableRows(_rectangularize(payload), [])
    if isinstance(payload, dict):
        if "texts" in payload:
            return _hitab_json_to_rows(payload)
        for key in ("rows", "data", "table", "values"):
            value = payload.get(key)
            if isinstance(value, str) and "<table" in value.lower():
                return _html_to_rows(value)
            if isinstance(value, list):
                return _TableRows(_rectangularize(value), [])
        return _TableRows([[key, value] for key, value in payload.items()], [])
    return _TableRows([[payload]], [])


def _write_json_payload(workbook: Workbook, payload: Any, *, title: str) -> str:
    if isinstance(payload, dict) and "texts" in payload:
        table = _hitab_json_to_rows(payload)
        sheet = workbook.create_sheet(_sheet_title(table.title or title))
        _write_rows(sheet, table.rows, table.merged_regions)
        return "hitab-json"

    if isinstance(payload, dict) and isinstance(payload.get("tables"), list):
        tables = [_table_payload_to_rows(table) for table in payload["tables"]]
        _write_combined_tables(workbook, tables)
        for index, table in enumerate(tables, start=1):
            sheet = workbook.create_sheet(_sheet_title(f"Table {index}"))
            _write_rows(sheet, table.rows, table.merged_regions)
        return "json:tables"

    table = _table_payload_to_rows(payload)
    sheet = workbook.create_sheet(_sheet_title(table.title or title))
    _write_rows(sheet, table.rows, table.merged_regions)
    return "json"


def _hitab_json_to_rows(payload: dict[str, Any]) -> _TableRows:
    return _TableRows(
        rows=_rectangularize(payload.get("texts") or []),
        merged_regions=[
            {
                "first_row": int(region.get("first_row", 0)),
                "last_row": int(region.get("last_row", region.get("first_row", 0))),
                "first_column": int(region.get("first_column", 0)),
                "last_column": int(region.get("last_column", region.get("first_column", 0))),
            }
            for region in payload.get("merged_regions") or []
        ],
        title=str(payload.get("title") or ""),
    )


def _html_to_rows(content: str) -> _TableRows:
    soup = BeautifulSoup(content, "html.parser")
    table = soup.find("table")
    if table is None:
        return _TableRows(_plain_text_to_rows(soup.get_text("\n", strip=True)), [])

    caption = table.find("caption")
    title = caption.get_text(" ", strip=True) if caption else ""
    rows: list[list[Any]] = []
    occupied: set[tuple[int, int]] = set()
    merged_regions: list[dict[str, int]] = []

    for row_index, tr in enumerate(table.find_all("tr")):
        while len(rows) <= row_index:
            rows.append([])
        col_index = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (row_index, col_index) in occupied:
                _ensure_cell(rows, row_index, col_index)
                col_index += 1

            rowspan = max(1, _safe_int(cell.get("rowspan"), 1))
            colspan = max(1, _safe_int(cell.get("colspan"), 1))
            _ensure_cell(rows, row_index, col_index)
            rows[row_index][col_index] = cell.get_text(" ", strip=True)

            if rowspan > 1 or colspan > 1:
                merged_regions.append(
                    {
                        "first_row": row_index,
                        "last_row": row_index + rowspan - 1,
                        "first_column": col_index,
                        "last_column": col_index + colspan - 1,
                    }
                )

            for row_offset in range(rowspan):
                for col_offset in range(colspan):
                    target = (row_index + row_offset, col_index + col_offset)
                    _ensure_cell(rows, target[0], target[1])
                    if row_offset or col_offset:
                        occupied.add(target)

            col_index += colspan

    return _TableRows(_rectangularize(rows), merged_regions, title=title)


def _write_combined_tables(workbook: Workbook, tables: list[_TableRows]) -> None:
    sheet = workbook.create_sheet("Combined")
    cursor = 1
    for index, table in enumerate(tables, start=1):
        sheet.cell(row=cursor, column=1, value=table.title or f"Table {index}")
        cursor += 1
        _write_rows(sheet, table.rows, table.merged_regions, start_row=cursor)
        cursor += len(table.rows) + 2


def _write_rows(
    sheet: Worksheet,
    rows: list[list[Any]],
    merged_regions: list[dict[str, int]],
    *,
    start_row: int = 1,
) -> None:
    rows = _rectangularize(rows)
    for row_index, row in enumerate(rows, start=start_row):
        for column_index, value in enumerate(row, start=1):
            sheet.cell(row=row_index, column=column_index, value=_cell_value(value))

    for region in merged_regions:
        first_row = start_row + int(region["first_row"])
        last_row = start_row + int(region["last_row"])
        first_col = 1 + int(region["first_column"])
        last_col = 1 + int(region["last_column"])
        if first_row != last_row or first_col != last_col:
            sheet.merge_cells(
                start_row=first_row,
                start_column=first_col,
                end_row=last_row,
                end_column=last_col,
            )


def _plain_text_to_rows(content: str) -> list[list[str]]:
    lines = [line for line in content.splitlines() if line.strip()]
    if any("|" in line for line in lines):
        return [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines
            if not re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*", line)
        ]
    return [[line] for line in lines]


def _rectangularize(rows: Any) -> list[list[Any]]:
    normalized = []
    for row in rows or []:
        if isinstance(row, dict):
            normalized.append([f"{key}: {value}" for key, value in row.items()])
        elif isinstance(row, (list, tuple)):
            normalized.append(list(row))
        else:
            normalized.append([row])
    width = max((len(row) for row in normalized), default=0)
    return [row + [""] * (width - len(row)) for row in normalized]


def _ensure_cell(rows: list[list[Any]], row_index: int, col_index: int) -> None:
    while len(rows) <= row_index:
        rows.append([])
    while len(rows[row_index]) <= col_index:
        rows[row_index].append("")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sheet_title(value: str) -> str:
    title = re.sub(r"[\[\]:*?/\\]", "_", str(value).strip())[:31]
    return title or "Table"


def _cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)
