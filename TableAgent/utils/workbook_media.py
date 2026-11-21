"""Pictures placed on the sheets of an XLSX workbook.

openpyxl drops most drawings, so pictures are read from the package parts:
each worksheet's drawing part lists anchored pictures, and the drawing's
relationships point at the image files under ``xl/media/``.

Positions follow Excel's default geometry (seven pixels per width unit,
96 dpi row heights), the same rule the document preview uses, so a picture is
reported on the cells a user sees it over.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import zipfile
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from TableAgent.utils.excel_utils import cell_to_a1

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
_R_ID = f"{{{_NS['r']}}}id"
_R_EMBED = f"{{{_NS['r']}}}embed"
_DRAWING_REL = "/relationships/drawing"
EMU_PER_PIXEL = 9525
PIXELS_PER_POINT = 96 / 72
# Image types a browser shows; others (EMF, WMF, WDP) are reported but not displayed.
PREVIEWABLE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})
MAX_PREVIEWABLE_BYTES = 50 * 1024 * 1024
MEDIA_TOKEN = re.compile(r"\[\[media:([A-Za-z0-9_]+)\]\]")


@dataclass(frozen=True)
class WorkbookMedia:
    """One picture placement; rows and columns are 1-based like openpyxl."""

    media_id: str
    sheet: str
    media_path: str
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    center_row: int
    center_col: int
    width_px: float
    height_px: float
    name: str
    description: str
    sha256: str
    previewable: bool

    @property
    def token(self) -> str:
        return f"[[media:{self.media_id}]]"

    @property
    def cells(self) -> str:
        start = cell_to_a1(self.from_row, self.from_col)
        end = cell_to_a1(self.to_row, self.to_col)
        return start if start == end else f"{start}:{end}"

    @property
    def location(self) -> str:
        return f"{self.sheet}!{self.cells}"

    @property
    def file_type(self) -> str:
        return posixpath.splitext(self.media_path)[1].lower().lstrip(".")


class _Axis:
    """Pixel offsets along rows or columns with a default size and exceptions."""

    def __init__(self, default_px: float, sizes: Mapping[int, float]) -> None:
        self.default_px = default_px
        self.indexes = sorted(sizes)
        self.deltas: list[float] = []
        total = 0.0
        for index in self.indexes:
            total += sizes[index] - default_px
            self.deltas.append(total)

    def start(self, index: int) -> float:
        """Offset of the 0-based line ``index``."""
        position = bisect_left(self.indexes, index)
        extra = self.deltas[position - 1] if position else 0.0
        return index * self.default_px + extra

    def locate(self, offset: float) -> tuple[int, float]:
        """0-based line containing ``offset`` and the offset inside it."""
        offset = max(0.0, offset)
        low, high = 0, max(1, int(offset / max(self.default_px, 1.0)) + 1)
        while self.start(high) <= offset:
            high *= 2
        while low < high:
            middle = (low + high + 1) // 2
            if self.start(middle) <= offset:
                low = middle
            else:
                high = middle - 1
        return low, offset - self.start(low)


def _column_width_px(width: float) -> float:
    return max(0.0, round(width * 7))


def _relationships(archive: zipfile.ZipFile, part_path: str) -> dict[str, tuple[str, str]]:
    directory, name = posixpath.split(part_path)
    try:
        root = ElementTree.fromstring(archive.read(posixpath.join(directory, "_rels", f"{name}.rels")))
    except (KeyError, ElementTree.ParseError):
        return {}
    targets: dict[str, tuple[str, str]] = {}
    for relation in root.findall("rel:Relationship", _NS):
        target = relation.get("Target", "")
        if relation.get("TargetMode") == "External" or not target:
            continue
        resolved = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(posixpath.join(directory, target))
        )
        targets[relation.get("Id", "")] = (resolved, relation.get("Type", ""))
    return targets


def _sheet_parts(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    except (KeyError, ElementTree.ParseError):
        return []
    targets = _relationships(archive, "xl/workbook.xml")
    sheets = workbook.find("main:sheets", _NS)
    parts = []
    for sheet in sheets if sheets is not None else []:
        target = targets.get(sheet.get(_R_ID, ""))
        if target:
            parts.append((sheet.get("name", ""), target[0]))
    return parts


def _sheet_geometry(archive: zipfile.ZipFile, worksheet_path: str) -> tuple[_Axis, _Axis]:
    default_row_pt, default_col_width = 15.0, 64 / 7
    col_sizes: dict[int, float] = {}
    row_sizes: dict[int, float] = {}
    try:
        stream = archive.open(worksheet_path)
    except KeyError:
        return _Axis(default_row_pt * PIXELS_PER_POINT, {}), _Axis(_column_width_px(default_col_width), {})
    with stream:
        for _, element in ElementTree.iterparse(stream, events=("end",)):
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "sheetFormatPr":
                try:
                    default_row_pt = float(element.get("defaultRowHeight") or 15)
                    if element.get("defaultColWidth"):
                        default_col_width = float(element.get("defaultColWidth") or 0)
                    elif element.get("baseColWidth"):
                        base = float(element.get("baseColWidth") or 8)
                        default_col_width = -(-(base * 7 + 5) // 8) * 8 / 7
                except ValueError:
                    pass
            elif tag == "col":
                try:
                    first, last = int(element.get("min", "0")), int(element.get("max", "0"))
                    width = 0.0 if element.get("hidden") in {"1", "true"} else float(element.get("width") or default_col_width)
                except ValueError:
                    continue
                # A trailing <col> often covers every remaining column; only explicit sizes matter.
                for column in range(first, min(last, first + 16384) + 1):
                    col_sizes[column - 1] = _column_width_px(width)
            elif tag == "row":
                try:
                    index = int(element.get("r", "0")) - 1
                except ValueError:
                    index = -1
                if index >= 0:
                    if element.get("hidden") in {"1", "true"}:
                        row_sizes[index] = 0.0
                    elif element.get("ht"):
                        try:
                            row_sizes[index] = float(element.get("ht") or 0) * PIXELS_PER_POINT
                        except ValueError:
                            pass
                element.clear()
            elif tag == "c":
                element.clear()
    return (
        _Axis(default_row_pt * PIXELS_PER_POINT, row_sizes),
        _Axis(_column_width_px(default_col_width), col_sizes),
    )


def _marker_px(anchor: ElementTree.Element, tag: str, rows: _Axis, cols: _Axis) -> tuple[float, float] | None:
    marker = anchor.find(f"xdr:{tag}", _NS)
    if marker is None:
        return None

    def number(name: str) -> int:
        node = marker.find(f"xdr:{name}", _NS)
        try:
            return int(node.text or 0) if node is not None else 0
        except ValueError:
            return 0

    x = cols.start(number("col")) + number("colOff") / EMU_PER_PIXEL
    y = rows.start(number("row")) + number("rowOff") / EMU_PER_PIXEL
    return x, y


def _anchor_rect(anchor: ElementTree.Element, rows: _Axis, cols: _Axis) -> tuple[float, float, float, float] | None:
    kind = anchor.tag.rsplit("}", 1)[-1]
    extent = anchor.find("xdr:ext", _NS)
    width = int(extent.get("cx", "0")) / EMU_PER_PIXEL if extent is not None else 0.0
    height = int(extent.get("cy", "0")) / EMU_PER_PIXEL if extent is not None else 0.0
    if kind == "absoluteAnchor":
        position = anchor.find("xdr:pos", _NS)
        if position is None:
            return None
        x = int(position.get("x", "0")) / EMU_PER_PIXEL
        y = int(position.get("y", "0")) / EMU_PER_PIXEL
        return x, y, x + width, y + height
    start = _marker_px(anchor, "from", rows, cols)
    if start is None:
        return None
    if kind == "twoCellAnchor":
        end = _marker_px(anchor, "to", rows, cols)
        if end is not None:
            return start[0], start[1], max(start[0], end[0]), max(start[1], end[1])
    if kind in {"twoCellAnchor", "oneCellAnchor"}:
        return start[0], start[1], start[0] + width, start[1] + height
    return None


def index_workbook_media(
    workbook_path: str | Path,
    *,
    sheets: Iterable[str] | None = None,
) -> list[WorkbookMedia]:
    """Every picture of the workbook (or of ``sheets``), in sheet and drawing order.

    An id names the sheet position and the cell under the picture's center
    (``1_C30``; ``1_C30_2`` for a second picture there), so it reads as a
    location and is the same whether or not ``sheets`` filters the result.
    """
    path = Path(workbook_path)
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        return []
    wanted = {name.casefold() for name in sheets} if sheets is not None else None
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile):
        return []
    media: list[WorkbookMedia] = []
    hashes: dict[str, str] = {}
    with archive:
        for sheet_number, (sheet_name, worksheet_path) in enumerate(_sheet_parts(archive), start=1):
            if wanted is not None and sheet_name.casefold() not in wanted:
                continue
            drawings = [
                target
                for target, relation_type in _relationships(archive, worksheet_path).values()
                if relation_type.endswith(_DRAWING_REL)
            ]
            if not drawings:
                continue
            rows, cols = _sheet_geometry(archive, worksheet_path)
            ids_in_cell: dict[str, int] = {}
            for drawing_path in drawings:
                try:
                    root = ElementTree.fromstring(archive.read(drawing_path))
                except (KeyError, ElementTree.ParseError):
                    continue
                targets = _relationships(archive, drawing_path)
                for anchor in root:
                    rect = _anchor_rect(anchor, rows, cols)
                    if rect is None:
                        continue
                    # Pictures grouped with shapes share the group's anchor.
                    for picture in anchor.iter(f"{{{_NS['xdr']}}}pic"):
                        blip = picture.find(".//a:blip", _NS)
                        target = targets.get(blip.get(_R_EMBED, "")) if blip is not None else None
                        if not target or not target[0].startswith("xl/media/"):
                            continue
                        properties = picture.find("xdr:nvPicPr/xdr:cNvPr", _NS)
                        if properties is not None and properties.get("hidden") in {"1", "true"}:
                            continue
                        media_path = target[0]
                        try:
                            size = archive.getinfo(media_path).file_size
                        except KeyError:
                            continue
                        if media_path not in hashes:
                            hashes[media_path] = hashlib.sha256(archive.read(media_path)).hexdigest()
                        left, top, right, bottom = rect
                        start_row, _ = rows.locate(top)
                        start_col, _ = cols.locate(left)
                        end_row, _ = rows.locate(max(top, bottom - 1))
                        end_col, _ = cols.locate(max(left, right - 1))
                        center_row, _ = rows.locate((top + bottom) / 2)
                        center_col, _ = cols.locate((left + right) / 2)
                        suffix = posixpath.splitext(media_path)[1].lower()
                        base_id = f"{sheet_number}_{cell_to_a1(center_row + 1, center_col + 1)}"
                        ids_in_cell[base_id] = ids_in_cell.get(base_id, 0) + 1
                        media.append(
                            WorkbookMedia(
                                media_id=base_id if ids_in_cell[base_id] == 1 else f"{base_id}_{ids_in_cell[base_id]}",
                                sheet=sheet_name,
                                media_path=media_path,
                                from_row=start_row + 1,
                                from_col=start_col + 1,
                                to_row=end_row + 1,
                                to_col=end_col + 1,
                                center_row=center_row + 1,
                                center_col=center_col + 1,
                                width_px=round(right - left, 1),
                                height_px=round(bottom - top, 1),
                                name=(properties.get("name", "") if properties is not None else "").strip(),
                                description=" ".join(
                                    value.strip()
                                    for value in (
                                        properties.get("title", "") if properties is not None else "",
                                        properties.get("descr", "") if properties is not None else "",
                                    )
                                    if value and value.strip()
                                ),
                                sha256=hashes[media_path],
                                previewable=suffix in PREVIEWABLE_SUFFIXES and size <= MAX_PREVIEWABLE_BYTES,
                            )
                        )
    return media


def read_media_bytes(workbook_path: str | Path, media_path: str) -> bytes:
    with zipfile.ZipFile(workbook_path) as archive:
        return archive.read(media_path)


def render_media_tokens(
    text: str,
    media: Iterable[WorkbookMedia],
    *,
    url_for: Callable[[WorkbookMedia], str | None],
) -> str:
    """Replace ``[[media:<id>]]`` tokens with Markdown images.

    A picture the browser cannot show, or one without a URL, keeps only its
    location; an unknown id is dropped.
    """
    if "[[media:" not in text:
        return text
    by_id = {item.media_id: item for item in media}

    def replace(match: re.Match[str]) -> str:
        item = by_id.get(match.group(1))
        if item is None:
            return ""
        alt = re.sub(r"[\[\]()]", " ", item.location).strip()
        url = url_for(item) if item.previewable else None
        if not url:
            return f"`{item.location} ({item.file_type})`"
        return f"![{alt}]({url})"

    rendered = MEDIA_TOKEN.sub(replace, text)
    # A dropped token leaves no trailing spaces behind.
    return re.sub(r"[ \t]+(?=\n|$)", "", rendered)
