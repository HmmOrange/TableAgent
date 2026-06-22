from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from openpyxl.utils import get_column_letter
from openpyxl.utils.cell import column_index_from_string


@dataclass(frozen=True)
class SheetMetadata:
    sheet_name: str
    used_range: str | None
    merged_ranges: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


class ExStructMetadataExtractor:
    """Run ExStruct once per workbook and derive the layout metadata contract."""

    def __init__(self, mode: str = "light"):
        self.mode = mode

    def extract(self, workbook_path: Path) -> dict[str, Any]:
        import exstruct

        workbook = exstruct.extract(str(workbook_path), mode=self.mode, alpha_col=True)
        if hasattr(workbook, "model_dump"):
            return workbook.model_dump(mode="json")
        return workbook.to_dict()

    def sheet_metadata(
        self,
        workbook_path: Path,
        workbook_payload: dict[str, Any],
        sheet_name: str,
    ) -> SheetMetadata:
        sheet = (workbook_payload.get("sheets") or {}).get(sheet_name) or {}
        return SheetMetadata(
            sheet_name=sheet_name,
            used_range=_used_range_from_rows(sheet.get("rows") or []),
            merged_ranges=[str(value) for value in sheet.get("merged_ranges") or []],
        )


def _used_range_from_rows(rows: list[dict[str, Any]]) -> str | None:
    coordinates: list[tuple[int, int]] = []
    for row in rows:
        try:
            row_index = int(row["r"])
        except (KeyError, TypeError, ValueError):
            continue
        cells = row.get("c") or {}
        if not isinstance(cells, dict):
            continue
        for column, value in cells.items():
            if value in (None, ""):
                continue
            try:
                column_index = (
                    int(column)
                    if str(column).isdigit()
                    else column_index_from_string(str(column))
                )
            except (TypeError, ValueError):
                continue
            coordinates.append((row_index, column_index))

    if not coordinates:
        return None
    min_row = min(row for row, _ in coordinates)
    max_row = max(row for row, _ in coordinates)
    min_col = min(column for _, column in coordinates)
    max_col = max(column for _, column in coordinates)
    return f"{get_column_letter(min_col)}{min_row}:{get_column_letter(max_col)}{max_row}"
