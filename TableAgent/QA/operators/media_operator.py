from __future__ import annotations

import io
import logging
import math
import re
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from TableAgent.QA.operators.base_operator import BaseOperator
from TableAgent.schema.header import Header
from TableAgent.utils.excel_utils import cell_to_a1
from TableAgent.utils.workbook_charts import chart_summary, read_chart, split_reference
from TableAgent.utils.structure_utils import flatten_headers
from TableAgent.utils.workbook_media import (
    KIND_CHART,
    KIND_PICTURE,
    WorkbookMedia,
    index_workbook_media,
    read_media_bytes,
)

logger = logging.getLogger(__name__)

MEDIA_COLUMN = "media"
# What tells pictures apart comes first: a wide printout elides the middle columns.
_LIST_COLUMNS = [
    "id", "cells", "kind", "text_above", "text_left", "row", "header", "table_id",
    "nearest_table", "side", "same_as", "name", "sheet", "viewable", "token",
]
# A picture mostly made of white background is a drawing (diagram, flowchart, chart, logo),
# not a photo; photos rarely have even a fifth of near-white pixels.
_DRAWING_WHITE_SHARE = 0.5
_NEAR_WHITE = 225
_WORD = re.compile(r"\w+")
_CELL_ADDRESS = re.compile(r"\b[A-Z]{1,3}\d+:\s")
# Vision calls one answer may make; each sends a whole picture to the model.
MAX_DESCRIBE_CALLS = 4
# Cells looked at around a picture for nearby labels.
_LABEL_SEARCH_ROWS = 3
_LABEL_SEARCH_COLS = 3
_LABEL_MAX_CHARS = 40
_LABELS_PER_SIDE = 2
# Length kept of each value written on a picture's table row.
_ROW_VALUE_MAX_CHARS = 160
# Rows of `list_media` that still fit a cell's printed output (about 2000 characters).
_LIST_PRINT_LIMIT = 8
# Cells of one chart reference read when a chart has no saved values.
_CHART_RANGE_MAX_CELLS = 5000
# Default names Excel gives inserted pictures ("Picture 12", "그림 3"); they say nothing.
_DEFAULT_PICTURE_NAME = re.compile(r"^[^\W\d_]+\s*\d+$")
_DESCRIBE_PROMPT = """Look at this picture from the spreadsheet sheet "{sheet}" (cells {cells}).
Describe what it shows so a question about it can be answered. Transcribe any visible text.
For a flowchart or diagram, list every box and the arrows between them in order, with branch labels.
Report only what is visible; do not guess hidden details.
{question_line}"""


# Text kept per cell when a picture list is printed.
_PRINT_COLWIDTH = 32
# A list cut down to this many columns prints them all.
_PRINT_ALL_COLUMNS = 8


class MediaFrame(pd.DataFrame):
    """A picture list that prints all the columns picked for it, with long text cut short.

    pandas otherwise hides middle columns to fit 80 characters, so a list printed
    with `id, kind, text_above, token` would show only `id ... token`. A list with
    every column keeps the plain printout, which still fits the output preview.
    """

    @property
    def _constructor(self):
        return MediaFrame

    def __repr__(self) -> str:
        if len(self.columns) > _PRINT_ALL_COLUMNS:
            return super().__repr__()
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 10_000,
            "display.max_colwidth", _PRINT_COLWIDTH,
        ):
            return super().__repr__()


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
    """Pictures, charts and diagrams placed on the loaded sheets, read from the workbook package."""

    name = "media"
    description = (
        "Pictures placed on the loaded sheets, and the Excel charts and diagrams (shapes, flowcharts, labels drawn on "
        "photos) drawn there; all of them are listed and shown the same way, so `picture` below covers charts and "
        "diagrams. A picture reaches the user only as its token "
        "`[[media:<id>]]` copied verbatim into `final_answer` where the picture should appear; it is shown as the image, so "
        "never call it a token or an id in the text, never write image URLs or Markdown image syntax yourself, and never "
        "invent ids. "
        f"DataFrames read from table rows (`read_table_as_dataframe`, `read_range_as_dataframe`, `read_sheet_as_dataframe`) "
        f"get a `{MEDIA_COLUMN}` column when those rows hold pictures: `<header label>: <tokens>` for the pictures on that "
        "row, or None. Such DataFrames are indexed by sheet row number, the same number as `row` in "
        "`list_media`: use `df.loc[row]`, never convert it; to find the rows that hold pictures, filter the "
        f"`{MEDIA_COLUMN}` column. "
        "A cell that looks empty may hold a picture. A picture under a header is part of that header's "
        "value: an empty cell of that header's column shows the picture's tokens, and when `final_answer` reports that "
        "header for the row, include its tokens. "
        "When the answer reports rows whose pictures show what the row describes (a part, a defect, a product, a person), "
        "put those tokens next to the row; leave them out of counts and totals. At most 8 tokens per answer. "
        "Pictures outside tables are listed by `operators.list_media`, with the cells they cover, their `kind` "
        "(`chart`: an Excel chart; `diagram`: Excel shapes drawn together, such as a flowchart or labels on a photo; for "
        "image files, `drawing`: mostly white background, as diagrams, flowcharts, charts and logos are; `photo` "
        "otherwise), the text "
        "written just above and left of them, and the nearest table with the picture's side of it, in sheet order. That "
        "text is only the closest written cell: often the picture's caption, but it may title a neighbouring section or "
        "another picture. `operators.find_media(query)` ranks pictures by how well that text matches words you take from "
        "the question. Print a list in a step of its own with only the columns you need; never use an id you have not "
        "seen printed. Identical pictures share a `same_as` group. Use `operators.describe_media` only when the question "
        "needs what a picture shows and the workbook text does not say it; it reads at most 4 image files, so pick the "
        "candidates by their text and kind first and read the best one first. For a `chart` it returns the plotted series "
        "and values with their cells, and for a `diagram` its text and arrows; those do not count toward the 4. When the "
        "question asks to show, draw or see a chart or diagram, put its token in `final_answer`. Its text explains a "
        "picture but never "
        "replaces it: when the picture is the value asked for, the token must still be in `final_answer`. For a picture on "
        "a table row it also lists the values on the same row, which may or may not be about this picture; it also "
        "names where an identical picture is placed. When it shows that a picture is not the one asked for, check the next "
        "candidate before concluding there is none. A token's id names the sheet position and the cell under "
        "the picture (`1_C30` is sheet 1, cell C30): check that it is on the row you report."
    )
    examples = (
        "operators.list_media(sheet='', table_id=None) -> pandas.DataFrame  # columns: " + ", ".join(_LIST_COLUMNS),
        "operators.find_media(query, sheet='') -> pandas.DataFrame  # every picture with list_media columns plus score, best match first (score 0: no word matched)",
        "operators.find_same_media(media_id) -> pandas.DataFrame  # identical picture elsewhere in the workbook, any sheet",
        "operators.describe_media(media_id, question='') -> str  # vision model reads an image once (at most 4); a chart or diagram is read from the workbook",
    )

    def __init__(self, env: Any):
        super().__init__(env)
        self._media: list[WorkbookMedia] | None = None
        self._areas: list[_TableArea] | None = None
        self._descriptions: dict[str, str] = {}
        self._describe_calls = 0
        self._merged: dict[str, list[Any]] = {}
        self._caption_cells: dict[str, dict[str, str]] = {}
        self._kinds: dict[str, str | None] = {}
        # Pictures the vision model read for this answer, in call order.
        self.described: list[str] = []

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

    def describe(self) -> str:  # type: ignore[override]
        """The operator text plus how many pictures outside tables each loaded sheet holds.

        Only sheets with such pictures get a line: pictures on table rows already
        come with the rows, so those sheets keep the plain operator text.
        """
        outside: dict[str, int] = {}
        for item in self.loaded_media():
            if not self._on_table_row(item):
                outside[item.sheet] = outside.get(item.sheet, 0) + 1
        lines = [f"- {sheet}: {count}" for sheet, count in outside.items()]
        text = super().describe()
        if lines:
            text += "\nPictures, charts and diagrams outside table rows, listed by `list_media`, per loaded sheet:\n" + "\n".join(lines)
            if any(count > _LIST_PRINT_LIMIT for count in outside.values()):
                text += (
                    f"\nWith more than {_LIST_PRINT_LIMIT} pictures, filter `list_media` (by its text, cells or rows) "
                    "before printing it; a full printout is cut in the middle."
                )
        return text

    def _media_by_id(self, media_id: str) -> WorkbookMedia:
        key = str(media_id).strip()
        if key.startswith("[[media:") and key.endswith("]]"):
            key = key[len("[[media:"):-2]
        for item in self.all_media():
            if item.media_id == key:
                return item
        raise KeyError(f"Unknown media id {media_id!r}; use ids from operators.list_media().")

    def _merged_ranges(self, sheet: str) -> list[Any]:
        if sheet not in self._merged:
            try:
                self._merged[sheet] = list(self.env.get_sheet(sheet).merged_cells.ranges)
            except Exception:
                self._merged[sheet] = []
        return self._merged[sheet]

    def _record_row(self, item: WorkbookMedia) -> int:
        """The sheet row a picture belongs to.

        A picture over a merged cell belongs to that cell, which a table reads on
        the merge's first row: a defect photo in a value merged over rows 2-3
        goes with row 2, where the record's text is.
        """
        row, col = item.center_row, item.center_col
        for merged in self._merged_ranges(item.sheet):
            if merged.min_row <= row <= merged.max_row and merged.min_col <= col <= merged.max_col:
                return merged.min_row
        return row

    def _area_of(self, item: WorkbookMedia) -> _TableArea | None:
        for area in self._table_areas():
            if (
                area.sheet.casefold() == item.sheet.casefold()
                and area.start_row <= item.center_row <= area.end_row
                and area.start_col <= item.center_col <= area.end_col
            ):
                return area
        return None

    def _on_table_row(self, item: WorkbookMedia) -> bool:
        """Whether a picture is a value of a table row; charts and diagrams never are."""
        if item.kind != KIND_PICTURE:
            return False
        area = self._area_of(item)
        return area is not None and area.has_data_row(self._record_row(item))

    @staticmethod
    def _header_label(area: _TableArea, row: int, column: int) -> str | None:
        header = MediaOperator._header_for(area, row, column)
        return header.label if header is not None else None

    @staticmethod
    def _header_for(area: _TableArea, row: int, column: int) -> Header | None:
        """The narrowest header over ``column`` among the nearest header rows above ``row``."""
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
        return candidates[0]

    # Table column ------------------------------------------------------------

    def _row_pictures(self, sheet: str, start_col: int, end_col: int):
        """``(row, header, picture)`` for pictures on table data rows overlapping the columns read."""
        for item in self.all_media():
            if item.sheet.casefold() != sheet.casefold() or not self._on_table_row(item):
                continue
            area = self._area_of(item)
            row = self._record_row(item)
            if area is None:
                continue
            if area.end_col < start_col or area.start_col > end_col:
                continue
            yield row, self._header_for(area, row, item.center_col), item

    def tokens_by_header(self, sheet: str, rows: list[int], *, start_col: int, end_col: int) -> dict[int, dict[str, list[str]]]:
        """Sheet row -> header id -> tokens of the pictures under that header on that row."""
        wanted = set(rows)
        found: dict[int, dict[str, list[str]]] = {}
        if not sheet or not wanted:
            return found
        for row, header, item in self._row_pictures(sheet, start_col, end_col):
            if row in wanted and header is not None:
                found.setdefault(row, {}).setdefault(header.id, []).append(item.token)
        return found

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
        for row, header, item in self._row_pictures(sheet, start_col, end_col):
            label = header.label if header is not None else ""
            by_row.setdefault(row, {}).setdefault(label, []).append(item.token)
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

    def _cell_text(self, sheet: str, row: int, col: int) -> tuple[str, str] | None:
        """``(address, text)`` of a cell that holds text.

        Merged cells are not followed back to their top-left value: a section
        title merged across a wide row is not a label of every picture under it.
        """
        if row < 1 or col < 1:
            return None
        try:
            value = self.env.get_sheet(sheet).cell(row=row, column=col).value
        except Exception:
            return None
        text = " ".join(str(value).split()) if value is not None else ""
        if not text:
            return None
        if len(text) > _LABEL_MAX_CHARS:
            text = text[: _LABEL_MAX_CHARS - 1] + "…"
        return cell_to_a1(row, col), text

    def _labels(self, sheet: str, cells: list[tuple[int, int]], *, skip: set[str] = frozenset()) -> list[tuple[str, str]]:
        found: dict[str, str] = {}
        for row, col in cells:
            label = self._cell_text(sheet, row, col)
            if label is not None and label[0] not in skip:
                found.setdefault(*label)
        return list(found.items())

    @staticmethod
    def _join_labels(labels: list[tuple[str, str]]) -> str:
        text = " | ".join(f"{address}: {value}" for address, value in labels[:_LABELS_PER_SIDE])
        return text + (" | …" if len(labels) > _LABELS_PER_SIDE else "")

    def _labels_above(self, item: WorkbookMedia) -> list[tuple[str, str]]:
        for row in range(item.from_row - 1, max(0, item.from_row - 1 - _LABEL_SEARCH_ROWS), -1):
            cells = [(row, col) for col in range(max(1, item.from_col - 1), item.to_col + 1)]
            if labels := self._labels(item.sheet, cells):
                return labels
        return []

    def _captions(self, sheet: str) -> dict[str, str]:
        """Cell address -> id of the picture the cell is written just above."""
        if sheet not in self._caption_cells:
            self._caption_cells[sheet] = {
                address: item.media_id
                for item in self.all_media()
                if item.sheet == sheet
                for address, _ in self._labels_above(item)
            }
        return self._caption_cells[sheet]

    def _text_above(self, item: WorkbookMedia) -> str:
        return self._join_labels(self._labels_above(item))

    def _text_left(self, item: WorkbookMedia) -> str:
        # A caption written just above another picture belongs to that picture, not to
        # every picture whose rows it happens to share.
        taken = {address for address, owner in self._captions(item.sheet).items() if owner != item.media_id}
        for col in range(item.from_col - 1, max(0, item.from_col - 1 - _LABEL_SEARCH_COLS), -1):
            cells = [(row, col) for row in range(item.from_row, item.to_row + 1)]
            if labels := self._labels(item.sheet, cells, skip=taken):
                return self._join_labels(labels)
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
        row = self._record_row(item)
        in_rows = self._on_table_row(item)
        nearest_table, side = (area.table_id, "inside") if area is not None else self._nearest_table(item)
        return {
            "id": item.media_id,
            "token": item.token,
            "sheet": item.sheet,
            "cells": item.cells,
            "table_id": area.table_id if area is not None else None,
            "row": row if in_rows else None,
            "header": self._header_label(area, row, item.center_col) if area is not None else None,
            "text_above": None if in_rows else self._text_above(item) or None,
            "text_left": None if in_rows else self._text_left(item) or None,
            "nearest_table": nearest_table,
            "side": side,
            "same_as": same_as.get(item.media_id),
            "name": item.description or (None if _DEFAULT_PICTURE_NAME.match(item.name or "") else item.name) or None,
            "viewable": item.previewable,
            "kind": self._kind(item),
        }

    def _kind(self, item: WorkbookMedia) -> str | None:
        """``"chart"``/``"diagram"``; for an image ``"drawing"`` when mostly white background, else ``"photo"``.

        None when the image cannot be read.
        """
        if item.kind != KIND_PICTURE:
            return item.kind
        if not item.previewable:
            return None
        if item.sha256 not in self._kinds:
            kind = None
            try:
                from PIL import Image

                with Image.open(io.BytesIO(read_media_bytes(self.env.workbook_path, item.media_path))) as image:
                    image = image.convert("RGB")
                    image.thumbnail((96, 96))
                    data = image.tobytes()
                pixels = len(data) // 3
                if pixels:
                    white = sum(1 for start in range(0, pixels * 3, 3) if min(data[start:start + 3]) >= _NEAR_WHITE)
                    kind = "drawing" if white / pixels >= _DRAWING_WHITE_SHARE else "photo"
            except Exception:
                logger.debug("Picture kind could not be read", exc_info=True)
            self._kinds[item.sha256] = kind
        return self._kinds[item.sha256]

    def list_media(self, sheet: str = "", table_id: Optional[str] = None) -> pd.DataFrame:
        """Pictures with their position and the text around them (loaded sheets unless ``sheet`` is given)."""
        same_as = self._same_as()
        items = self.all_media() if sheet else self.loaded_media()
        sheet_order = {name: index for index, name in reversed(list(enumerate(item.sheet for item in items)))}
        records = []
        # Sheet reading order: top to bottom, then left to right.
        for item in sorted(items, key=lambda item: (sheet_order[item.sheet], item.from_row, item.from_col)):
            if sheet and item.sheet.casefold() != str(sheet).strip().casefold():
                continue
            record = self._row_record(item, same_as)
            if table_id and record["table_id"] != table_id and record["nearest_table"] != table_id:
                continue
            records.append(record)
        # object dtype keeps whole row numbers and None instead of floats and NaN.
        return MediaFrame(records, columns=_LIST_COLUMNS, dtype=object)

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
        return MediaFrame(records, columns=_LIST_COLUMNS, dtype=object)

    @staticmethod
    def _words(text: Any) -> set[str]:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return set()
        plain = _CELL_ADDRESS.sub(" ", str(text))
        return {word for word in _WORD.findall(unicodedata.normalize("NFC", plain).casefold()) if len(word) > 1}

    def find_media(self, query: str, sheet: str = "") -> pd.DataFrame:
        """Every picture, ranked by how well the text around it matches ``query``.

        Words written just above a picture, its header and its name count twice
        as much as words to its left; words found around many pictures count
        less. Pictures with no matching word stay at the bottom with score 0, so
        a query worded differently from the sheet still lists every candidate.
        """
        listed = self.list_media(sheet=sheet)
        if listed.empty:
            return listed.assign(score=pd.Series(dtype=float))
        query_words = self._words(query)
        fields = [
            (self._words(record["text_above"]) | self._words(record["header"]) | self._words(record["name"]),
             self._words(record["text_left"]))
            for _, record in listed.iterrows()
        ]
        counts: dict[str, int] = {}
        for strong, weak in fields:
            for word in strong | weak:
                counts[word] = counts.get(word, 0) + 1
        scores = []
        for strong, weak in fields:
            score = 0.0
            for word in query_words:
                weight = 2.0 if word in strong else 1.0 if word in weak else 0.0
                if weight:
                    score += weight * math.log(1 + len(fields) / counts[word])
            scores.append(round(score, 2))
        ranked = listed.assign(score=scores).sort_values("score", ascending=False, kind="stable")
        return ranked[["score"] + _LIST_COLUMNS]

    def _copies_note(self, item: WorkbookMedia) -> str:
        copies = [other for other in self.all_media() if other.sha256 == item.sha256 and other.media_id != item.media_id]
        if not copies:
            return ""
        return "\nThe identical picture is also placed at: " + ", ".join(
            f"{other.location} (id {other.media_id})" for other in copies
        )

    def _row_values(self, item: WorkbookMedia) -> tuple[_TableArea, int, list[tuple[str, str]]] | None:
        """The values written on the table row a picture belongs to, with their header labels.

        A merged value counts once, on its first column, and a value merged down
        from an earlier row is read from its top-left cell.
        """
        area = self._area_of(item)
        row = self._record_row(item)
        if area is None or not self._on_table_row(item):
            return None
        sheet = self.env.get_sheet(item.sheet)
        merged = self._merged_ranges(item.sheet)
        values: list[tuple[str, str]] = []
        for col in range(area.start_col, area.end_col + 1):
            source = (row, col)
            for span in merged:
                if span.min_row <= row <= span.max_row and span.min_col <= col <= span.max_col:
                    source = None if col != span.min_col else (span.min_row, span.min_col)
                    break
            if source is None:
                continue
            value = sheet.cell(row=source[0], column=source[1]).value
            text = " ".join(str(value).split()) if value is not None else ""
            if not text:
                continue
            if len(text) > _ROW_VALUE_MAX_CHARS:
                text = text[: _ROW_VALUE_MAX_CHARS - 1] + "…"
            values.append((self._header_label(area, row, col) or cell_to_a1(row, col), text))
        return area, row, values

    def _row_note(self, item: WorkbookMedia) -> str:
        try:
            found = self._row_values(item)
        except Exception:
            logger.debug("Row values of a picture could not be read", exc_info=True)
            return ""
        if not found or not found[2]:
            return ""
        area, row, values = found
        return (
            f"\nValues on the same table row ({area.table_id}, sheet row {row}): "
            + "; ".join(f"{label}: {text}" for label, text in values)
        )

    def describe_media(self, media_id: str, question: str = "") -> str:
        """What a picture shows, read by the vision model once per picture.

        Asking again about the same picture (or an identical copy) returns the
        first reading, so the call budget covers distinct pictures.
        """
        item = self._media_by_id(media_id)
        if item.kind != KIND_PICTURE:
            # Read from the workbook itself: exact, and no vision call.
            self._note_described(item)
            heading = f"{item.kind.capitalize()} at {item.location}"
            if not item.previewable:
                heading += " (a chart type the chat cannot draw; its token shows only its location)"
            text = self._chart_text(item) if item.kind == KIND_CHART else item.text
            return f"{heading}:\n{text}" if text else f"{heading}: nothing could be read from it."
        if not item.previewable:
            return f"{item.location} is a .{item.file_type} picture that cannot be read."
        key = item.sha256
        if key in self._descriptions:
            self._note_described(item)
            return self._descriptions[key] + self._row_note(item) + self._copies_note(item)
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
        self._note_described(item)
        return text + self._row_note(item) + self._copies_note(item)

    def _read_chart_range(self, formula: str) -> list[object]:
        reference = split_reference(formula)
        if reference is None:
            return []
        sheet_name, cells = reference
        sheet = self.env.get_sheet(sheet_name)
        values: list[object] = []
        for row in sheet[cells] if ":" in cells else ((sheet[cells],),):
            for cell in row if isinstance(row, tuple) else (row,):
                values.append(cell.value)
                if len(values) >= _CHART_RANGE_MAX_CELLS:
                    return values
        return values

    def _chart_text(self, item: WorkbookMedia) -> str:
        """The chart's series, from its saved values or, without them, from its cells."""
        try:
            with zipfile.ZipFile(self.env.workbook_path) as archive:
                chart = read_chart(archive, item.media_path, read_range=self._read_chart_range)
        except Exception:
            logger.debug("Chart could not be read again from its cells", exc_info=True)
            return item.text
        return chart_summary(chart) if chart is not None else item.text

    def _note_described(self, item: WorkbookMedia) -> None:
        if item.media_id not in self.described:
            self.described.append(item.media_id)

    def pictures_left_out(self, answer: str) -> list[str]:
        """Tokens of the pictures an answer explains in words but does not show.

        Only when the answer shows no picture at all and every picture the vision
        model read lies on the same table row: the answer then reports that row's
        pictures, so they belong in it. Pictures read while searching (on several
        rows, or outside tables) are left alone.
        """
        if not self.described or not str(answer).strip() or "[[media:" in str(answer):
            return []
        rows = set()
        items = []
        for media_id in self.described:
            item = self._media_by_id(media_id)
            area = self._area_of(item)
            row = self._record_row(item)
            if area is None or not self._on_table_row(item):
                return []
            rows.add((area.table_id, row))
            items.append(item)
        if len(rows) != 1:
            return []
        return [item.token for item in items]
