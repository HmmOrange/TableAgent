from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from TableAgent.QA.operators.base_operator import BaseOperator
from TableAgent.schema.header import Header
from TableAgent.utils.excel_utils import cell_to_a1
from TableAgent.utils.structure_utils import flatten_headers
from TableAgent.utils.workbook_media import WorkbookMedia, index_workbook_media, read_media_bytes

logger = logging.getLogger(__name__)

MEDIA_COLUMN = "media"
_LIST_COLUMNS = [
    "id", "token", "sheet", "cells", "table_id", "row", "header", "text_above",
    "text_left", "nearest_table", "side", "same_as", "name", "viewable",
]
# Vision calls one answer may make; each sends a whole picture to the model.
MAX_DESCRIBE_CALLS = 4
# Cells looked at around a picture for nearby labels.
_LABEL_SEARCH_ROWS = 3
_LABEL_SEARCH_COLS = 3
_LABEL_MAX_CHARS = 80
_DESCRIBE_PROMPT = """Look at this picture from the spreadsheet sheet "{sheet}" (cells {cells}).
Describe what it shows so a question about it can be answered. Transcribe any visible text.
For a flowchart or diagram, list every box and the arrows between them in order, with branch labels.
Report only what is visible; do not guess hidden details.
{question_line}"""


@dataclass(frozen=True)
class _TableArea:
    table_id: str
    sheet: str
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    headers: tuple[Header, ...]
    # Row spans of the headers' data ranges; a table can hold several blocks.
    data_rows: tuple[tuple[int, int], ...]

    def has_data_row(self, row: int) -> bool:
        return any(start <= row <= end for start, end in self.data_rows)


class MediaOperator(BaseOperator):
    """Pictures placed on the loaded sheets, read from the workbook package."""

    name = "media"
    description = (
        "Pictures placed on the loaded sheets. A picture reaches the user only as its token "
        "`[[media:<id>]]` copied verbatim into `final_answer` where the picture should appear; it is shown as the image, so "
        "never call it a token or an id in the text, never write image URLs or Markdown image syntax yourself, and never "
        "invent ids. "
        f"DataFrames read from table rows (`read_table_as_dataframe`, `read_range_as_dataframe`, `read_sheet_as_dataframe`) "
        f"get a `{MEDIA_COLUMN}` column when those rows hold pictures: `<header label>: <tokens>` for the pictures on that "
        "row, or None. A cell that looks empty may hold a picture. A picture under a header is part of that header's "
        "value: when `final_answer` reports that header for the row, include its tokens even if the cell text is empty. "
        "When the answer reports rows whose pictures show what the row describes (a part, a defect, a product, a person), "
        "put those tokens next to the row; leave them out of counts and totals. At most 8 tokens per answer. "
        "Pictures outside tables are listed by `operators.list_media`, with the cells they cover, the text written just "
        "above and left of them, and the nearest table with the picture's side of it. Identical pictures share a "
        "`same_as` group. Use `operators.describe_media` only when the question needs what a picture shows and the "
        "workbook text does not say it; its text explains a picture but never replaces it: when the picture is the value "
        "asked for, the token must still be in `final_answer`. A token's id names the sheet position and the cell under "
        "the picture (`1_C30` is sheet 1, cell C30): check that it is on the row you report."
    )
    examples = (
        "operators.list_media(sheet='', table_id=None) -> pandas.DataFrame  # columns: id, token, sheet, cells, table_id, row, header, text_above, text_left, nearest_table, side, same_as, name, viewable",
        "operators.find_same_media(media_id) -> pandas.DataFrame  # identical picture elsewhere in the workbook, any sheet",
        "operators.describe_media(media_id, question='') -> str  # vision model reads the picture; at most 4 calls",
    )

    def __init__(self, env: Any):
        super().__init__(env)
        self._media: list[WorkbookMedia] | None = None
        self._areas: list[_TableArea] | None = None
        self._descriptions: dict[tuple[str, str], str] = {}
        self._describe_calls = 0

    # Index ---------------------------------------------------------------

    def _table_areas(self) -> list[_TableArea]:
        if self._areas is not None:
            return self._areas
        structures: list[tuple[str, dict[str, Any]]] = list(getattr(self.env, "structures", {}).items())
        structures.extend(
            (str(item.get("table_id") or ""), item.get("structure") or {})
            for item in getattr(self.env, "related_structures", []) or []
            if isinstance(item, dict)
        )
        areas: list[_TableArea] = []
        seen: set[str] = set()
        for table_id, structure in structures:
            if not table_id or table_id in seen or not isinstance(structure, dict):
                continue
            seen.add(table_id)
            try:
                headers = tuple(flatten_headers(structure.get("headers") or []))
            except Exception:
                continue
            ranges = [
                cell_range
                for header in headers
                for cell_range in (header.header_range, header.data_range)
                if cell_range is not None
            ]
            if not ranges:
                continue
            start_row = min(cell_range.start_row for cell_range in ranges)
            areas.append(
                _TableArea(
                    table_id=table_id,
                    sheet=ranges[0].sheet,
                    start_row=start_row,
                    start_col=min(cell_range.start_col for cell_range in ranges),
                    end_row=max(cell_range.end_row for cell_range in ranges),
                    end_col=max(cell_range.end_col for cell_range in ranges),
                    headers=headers,
                    data_rows=tuple(
                        sorted(
                            {
                                (header.data_range.start_row, header.data_range.end_row)
                                for header in headers
                                if header.data_range is not None
                            }
                        )
                    ),
                )
            )
        self._areas = areas
        return areas

    def all_media(self) -> list[WorkbookMedia]:
        """Every picture of the workbook; empty when none or unreadable."""
        if self._media is not None:
            return self._media
        workbook_path = getattr(self.env, "workbook_path", "")
        media: list[WorkbookMedia] = []
        if workbook_path:
            try:
                media = index_workbook_media(workbook_path)
            except Exception:
                logger.warning("Workbook pictures could not be read", exc_info=True)
        self._media = media
        return media

    def loaded_media(self) -> list[WorkbookMedia]:
        """Pictures on the sheets of the loaded tables."""
        sheets = {area.sheet.casefold() for area in self._table_areas() if area.sheet}
        return [item for item in self.all_media() if item.sheet.casefold() in sheets]

    def has_media(self) -> bool:
        return bool(self.loaded_media())

    def _media_by_id(self, media_id: str) -> WorkbookMedia:
        key = str(media_id).strip()
        if key.startswith("[[media:") and key.endswith("]]"):
            key = key[len("[[media:"):-2]
        for item in self.all_media():
            if item.media_id == key:
                return item
        raise KeyError(f"Unknown media id {media_id!r}; use ids from operators.list_media().")

    def _area_of(self, item: WorkbookMedia) -> _TableArea | None:
        for area in self._table_areas():
            if (
                area.sheet.casefold() == item.sheet.casefold()
                and area.start_row <= item.center_row <= area.end_row
                and area.start_col <= item.center_col <= area.end_col
            ):
                return area
        return None

    @staticmethod
    def _header_label(area: _TableArea, row: int, column: int) -> str | None:
        """Label of the narrowest header over ``column`` among the nearest header rows above ``row``."""
        candidates = [
            header
            for header in area.headers
            if header.header_range is not None
            and header.header_range.start_col <= column <= header.header_range.end_col
            and header.header_range.end_row < row
        ]
        if not candidates:
            return None
        nearest = max(header.header_range.end_row for header in candidates)  # type: ignore[union-attr]
        candidates = [header for header in candidates if header.header_range.end_row == nearest]  # type: ignore[union-attr]
        candidates.sort(key=lambda header: header.header_range.end_col - header.header_range.start_col)  # type: ignore[union-attr]
        return candidates[0].label

    # Table column ------------------------------------------------------------

    def tokens_for_rows(
        self,
        sheet: str,
        rows: list[int],
        *,
        start_col: int,
        end_col: int,
    ) -> list[Optional[str]] | None:
        """Tokens of the pictures over each sheet row read; None when no row has one.

        A picture counts for a row when its center lies on a data row of a
        table that overlaps the columns read, so header logos stay out.
        """
        if not rows or not sheet:
            return None
        # Row -> header label -> tokens, in drawing order.
        by_row: dict[int, dict[str, list[str]]] = {}
        for item in self.all_media():
            if item.sheet.casefold() != sheet.casefold():
                continue
            area = self._area_of(item)
            if area is None or not area.has_data_row(item.center_row):
                continue
            if area.end_col < start_col or area.start_col > end_col:
                continue
            label = self._header_label(area, item.center_row, item.center_col) or ""
            by_row.setdefault(item.center_row, {}).setdefault(label, []).append(item.token)
        if not any(row in by_row for row in rows):
            return None
        return [
            "; ".join(
                f"{label}: {' '.join(tokens)}" if label else " ".join(tokens)
                for label, tokens in by_row[row].items()
            )
            if row in by_row
            else None
            for row in rows
        ]

    # Agent-facing operators ----------------------------------------------------

    def _cell_text(self, sheet: str, row: int, col: int) -> str:
        if row < 1 or col < 1:
            return ""
        try:
            value = self.env.get_sheet(sheet).cell(row=row, column=col).value
        except Exception:
            return ""
        if value is None:
            return ""
        return " ".join(str(value).split())[:_LABEL_MAX_CHARS]

    def _text_above(self, item: WorkbookMedia) -> str:
        for row in range(item.from_row - 1, max(0, item.from_row - 1 - _LABEL_SEARCH_ROWS), -1):
            values = [
                text
                for col in range(max(1, item.from_col - 1), item.to_col + 1)
                if (text := self._cell_text(item.sheet, row, col))
            ]
            if values:
                return f"{cell_to_a1(row, item.from_col)}: " + " | ".join(values)
        return ""

    def _text_left(self, item: WorkbookMedia) -> str:
        for col in range(item.from_col - 1, max(0, item.from_col - 1 - _LABEL_SEARCH_COLS), -1):
            values = [
                text
                for row in range(item.from_row, item.to_row + 1)
                if (text := self._cell_text(item.sheet, row, col))
            ]
            if values:
                return f"{cell_to_a1(item.from_row, col)}: " + " | ".join(values)
        return ""

    def _nearest_table(self, item: WorkbookMedia) -> tuple[str | None, str | None]:
        best: tuple[int, str, str] | None = None
        for area in self._table_areas():
            if area.sheet.casefold() != item.sheet.casefold():
                continue
            row_gap = max(area.start_row - item.to_row, item.from_row - area.end_row, 0)
            col_gap = max(area.start_col - item.to_col, item.from_col - area.end_col, 0)
            if row_gap == 0 and col_gap == 0:
                side = "inside"
            elif col_gap >= row_gap:
                side = "right" if item.from_col > area.end_col else "left"
            else:
                side = "below" if item.from_row > area.end_row else "above"
            distance = row_gap + col_gap
            if best is None or distance < best[0]:
                best = (distance, area.table_id, side)
        return (best[1], best[2]) if best else (None, None)

    def _same_as(self) -> dict[str, str]:
        groups: dict[str, list[str]] = {}
        for item in self.all_media():
            groups.setdefault(item.sha256, []).append(item.media_id)
        return {
            media_id: ids[0]
            for ids in groups.values()
            if len(ids) > 1
            for media_id in ids
        }

    def _row_record(self, item: WorkbookMedia, same_as: dict[str, str]) -> dict[str, Any]:
        area = self._area_of(item)
        in_rows = area is not None and area.has_data_row(item.center_row)
        nearest_table, side = (area.table_id, "inside") if area is not None else self._nearest_table(item)
        return {
            "id": item.media_id,
            "token": item.token,
            "sheet": item.sheet,
            "cells": item.cells,
            "table_id": area.table_id if area is not None else None,
            "row": item.center_row if in_rows else None,
            "header": self._header_label(area, item.center_row, item.center_col) if area is not None else None,
            "text_above": None if in_rows else self._text_above(item) or None,
            "text_left": None if in_rows else self._text_left(item) or None,
            "nearest_table": nearest_table,
            "side": side,
            "same_as": same_as.get(item.media_id),
            "name": item.description or item.name or None,
            "viewable": item.previewable,
        }

    def list_media(self, sheet: str = "", table_id: Optional[str] = None) -> pd.DataFrame:
        """Pictures with their position and the text around them (loaded sheets unless ``sheet`` is given)."""
        same_as = self._same_as()
        records = []
        for item in self.all_media() if sheet else self.loaded_media():
            if sheet and item.sheet.casefold() != str(sheet).strip().casefold():
                continue
            record = self._row_record(item, same_as)
            if table_id and record["table_id"] != table_id and record["nearest_table"] != table_id:
                continue
            records.append(record)
        # object dtype keeps whole row numbers and None instead of floats and NaN.
        return pd.DataFrame(records, columns=_LIST_COLUMNS, dtype=object)

    def find_same_media(self, media_id: str) -> pd.DataFrame:
        """Other placements of the identical picture (same image bytes)."""
        item = self._media_by_id(media_id)
        same_as = self._same_as()
        records = [
            self._row_record(other, same_as)
            for other in self.all_media()
            if other.sha256 == item.sha256 and other.media_id != item.media_id
        ]
        # object dtype keeps whole row numbers and None instead of floats and NaN.
        return pd.DataFrame(records, columns=_LIST_COLUMNS, dtype=object)

    def describe_media(self, media_id: str, question: str = "") -> str:
        """What a picture shows, read by the vision model; cached per picture and question."""
        item = self._media_by_id(media_id)
        if not item.previewable:
            return f"{item.location} is a .{item.file_type} picture that cannot be read."
        key = (item.sha256, " ".join(str(question).split()))
        if key in self._descriptions:
            return self._descriptions[key]
        client = getattr(self.env, "vision_client", None)
        generate_with_image = getattr(client, "generate_with_image", None)
        if not callable(generate_with_image):
            return "No vision model is available to read pictures."
        if self._describe_calls >= MAX_DESCRIBE_CALLS:
            return f"Picture reading limit reached ({MAX_DESCRIBE_CALLS} per answer)."
        self._describe_calls += 1
        prompt = _DESCRIBE_PROMPT.format(
            sheet=item.sheet,
            cells=item.cells,
            question_line=f"Question to support: {question}" if str(question).strip() else "",
        ).strip()
        with tempfile.TemporaryDirectory(prefix="table-agent-media-") as temp_text:
            image_path = Path(temp_text) / f"{item.media_id}.{item.file_type}"
            image_path.write_bytes(read_media_bytes(self.env.workbook_path, item.media_path))
            response = generate_with_image(prompt=prompt, image_path=image_path)
        text = str(getattr(response, "content", response) or "").strip()
        self._descriptions[key] = text
        return text
