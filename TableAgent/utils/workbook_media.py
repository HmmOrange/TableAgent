"""Pictures, charts and diagrams placed on the sheets of an XLSX workbook.

openpyxl drops most drawings, so they are read from the package parts: each
worksheet's drawing part lists anchored pictures, whose relationships point at
the image files under ``xl/media/``, charts (``xl/charts/chartN.xml``), and
shapes. Shapes that belong together (boxes joined by arrows, labels drawn on a
photo) are reported as one diagram covering their joint area.

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
from typing import Any
from pathlib import Path
from xml.etree import ElementTree

from TableAgent.utils.excel_utils import cell_to_a1
from TableAgent.utils.workbook_charts import chart_summary, read_chart

_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
}
_R_ID = f"{{{_NS['r']}}}id"
_R_EMBED = f"{{{_NS['r']}}}embed"
_DRAWING_REL = "/relationships/drawing"
EMU_PER_PIXEL = 9525
PIXELS_PER_POINT = 96 / 72
# Image types a browser shows; others (EMF, WMF, WDP) are reported but not displayed.
PREVIEWABLE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"})
MAX_PREVIEWABLE_BYTES = 50 * 1024 * 1024
KIND_PICTURE = "picture"
KIND_CHART = "chart"
KIND_DIAGRAM = "diagram"
# Shapes this close (pixels) are drawn together: a connector ends on the box it points at.
DIAGRAM_GAP_PX = 6.0
# Text of one diagram box kept in the diagram's summary.
_DIAGRAM_TEXT_MAX_CHARS = 120
MEDIA_TOKEN = re.compile(r"\[\[media:([A-Za-z0-9_]+)\]\]")
# An id written into the text without the token brackets, optionally in backticks.
MEDIA_MENTION = re.compile(r"`?\bmedia:([A-Za-z0-9_]+)`?")
# A bare picture id in the text ("2_C18", "1_P22_2"): sheet number, cell, optional copy number.
MEDIA_BARE_ID = re.compile(r"(?<![\w/:.\-])(\d+_[A-Z]{1,3}\d+(?:_\d+)?)(?!\w)")


def mentions_media(text: str) -> bool:
    """Whether ``text`` may hold picture tokens or picture ids to render."""
    text = str(text or "")
    return "media:" in text or MEDIA_BARE_ID.search(text) is not None


@dataclass(frozen=True)
class WorkbookMedia:
    """One picture, chart or diagram placement; rows and columns are 1-based like openpyxl.

    ``media_path`` is the image part of a picture and the chart part of a chart;
    a diagram is drawn from the shapes of its cells, so it has none.
    """

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
    kind: str = KIND_PICTURE
    # Chart data or diagram boxes and arrows, as text the agent can read.
    text: str = ""

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
        if self.kind != KIND_PICTURE:
            return self.kind
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


@dataclass
class _ShapePart:
    shape_id: str
    text: str
    connector: bool
    # A box, symbol or line rather than a plain text box.
    drawn: bool
    start_id: str = ""
    end_id: str = ""
    # Arrowheads drawn at the connector's start and end.
    head_arrow: bool = False
    tail_arrow: bool = False


@dataclass
class _DrawingUnit:
    """One anchor taking part in diagram grouping: a shape anchor or a picture."""

    rect: tuple[float, float, float, float]
    drawing: str
    parts: list[_ShapePart]
    # Set for a picture: the id it is listed under.
    media_id: str = ""
    # Ends of a connector anchored on its own, for arrows not glued to a box.
    ends: tuple[tuple[float, float], tuple[float, float]] | None = None


def _hidden(properties: ElementTree.Element | None) -> bool:
    return properties is not None and properties.get("hidden") in {"1", "true"}


def _paragraph_text(node: ElementTree.Element | None) -> str:
    if node is None:
        return ""
    paragraphs = []
    for paragraph in node.iter(f"{{{_NS['a']}}}p"):
        text = " ".join("".join(run.text or "" for run in paragraph.iter(f"{{{_NS['a']}}}t")).split())
        if text:
            paragraphs.append(text)
    text = " ".join(paragraphs)
    return text if len(text) <= _DIAGRAM_TEXT_MAX_CHARS else text[: _DIAGRAM_TEXT_MAX_CHARS - 1] + "…"


def _shape_parts(anchor: ElementTree.Element) -> list[_ShapePart]:
    """Visible shapes and connectors of an anchor; empty text boxes draw nothing and are left out."""
    parts: list[_ShapePart] = []
    for shape in anchor.iter(f"{{{_NS['xdr']}}}sp"):
        properties = shape.find("xdr:nvSpPr/xdr:cNvPr", _NS)
        if _hidden(properties):
            continue
        flags = shape.find("xdr:nvSpPr/xdr:cNvSpPr", _NS)
        text_box = flags is not None and flags.get("txBox") in {"1", "true"}
        text = _paragraph_text(shape.find("xdr:txBody", _NS))
        if text_box and not text:
            continue
        parts.append(
            _ShapePart(
                shape_id=properties.get("id", "") if properties is not None else "",
                text=text,
                connector=False,
                drawn=not text_box,
            )
        )
    for connector in anchor.iter(f"{{{_NS['xdr']}}}cxnSp"):
        properties = connector.find("xdr:nvCxnSpPr/xdr:cNvPr", _NS)
        if _hidden(properties):
            continue
        start = connector.find("xdr:nvCxnSpPr/xdr:cNvCxnSpPr/a:stCxn", _NS)
        end = connector.find("xdr:nvCxnSpPr/xdr:cNvCxnSpPr/a:endCxn", _NS)
        head = connector.find("xdr:spPr/a:ln/a:headEnd", _NS)
        tail = connector.find("xdr:spPr/a:ln/a:tailEnd", _NS)
        parts.append(
            _ShapePart(
                shape_id=properties.get("id", "") if properties is not None else "",
                text="",
                connector=True,
                drawn=True,
                start_id=start.get("id", "") if start is not None else "",
                end_id=end.get("id", "") if end is not None else "",
                head_arrow=head is not None and head.get("type", "none") != "none",
                tail_arrow=tail is not None and tail.get("type", "none") != "none",
            )
        )
    return parts


def _connector_ends(
    anchor: ElementTree.Element,
    rect: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Start and end of a connector that is the anchor's only content (flips swap the corners)."""
    connector = anchor.find("xdr:cxnSp", _NS)
    if connector is None:
        return None
    transform = connector.find("xdr:spPr/a:xfrm", _NS)
    flip_h = transform is not None and transform.get("flipH") in {"1", "true"}
    flip_v = transform is not None and transform.get("flipV") in {"1", "true"}
    left, top, right, bottom = rect
    start = (right if flip_h else left, bottom if flip_v else top)
    end = (left if flip_h else right, top if flip_v else bottom)
    return start, end


def _near(first: tuple[float, float, float, float], second: tuple[float, float, float, float], gap: float) -> bool:
    return (
        first[0] - gap <= second[2]
        and second[0] - gap <= first[2]
        and first[1] - gap <= second[3]
        and second[1] - gap <= first[3]
    )


def _group_units(units: list[_DrawingUnit]) -> list[list[_DrawingUnit]]:
    """Units drawn together: shapes that touch, and the pictures they are drawn on.

    Two pictures side by side stay apart unless a shape joins them.
    """
    parent = list(range(len(units)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    shapes = [index for index, unit in enumerate(units) if not unit.media_id]
    for index in shapes:
        for other in range(len(units)):
            if other != index and _near(units[index].rect, units[other].rect, DIAGRAM_GAP_PX):
                parent[root(index)] = root(other)
    groups: dict[int, list[_DrawingUnit]] = {}
    for index, unit in enumerate(units):
        groups.setdefault(root(index), []).append(unit)
    return list(groups.values())


def _is_diagram(members: list[_DrawingUnit]) -> bool:
    """Boxes or lines drawn together, or shapes drawn on a picture; a lone note or plain text boxes are not."""
    parts = [part for unit in members for part in unit.parts]
    pictures = [unit for unit in members if unit.media_id]
    if not pictures:
        return len(parts) >= 2 and any(part.drawn for part in parts)
    if any(part.drawn for part in parts):
        return True
    # Only text boxes: a label written on a picture, not a caption beside it.
    return any(
        _near(unit.rect, picture.rect, -1.0)
        for unit in members
        if unit.parts
        for picture in pictures
    )


def _diagram_summary(members: list[_DrawingUnit], pictures: Mapping[str, WorkbookMedia]) -> tuple[str, str]:
    """``(title, text)`` of a diagram: its text in reading order, its arrows and the pictures it covers."""
    ordered = sorted(members, key=lambda unit: (round(unit.rect[1]), round(unit.rect[0])))
    names: dict[tuple[str, str], str] = {}
    texts: list[str] = []
    shapes = connectors = 0
    for unit in ordered:
        for part in unit.parts:
            if part.connector:
                connectors += 1
                continue
            shapes += 1
            if part.text:
                names[(unit.drawing, part.shape_id)] = part.text
                if part.text not in texts:
                    texts.append(part.text)
    text_units = [unit for unit in members if len(unit.parts) == 1 and unit.parts[0].text and not unit.parts[0].connector]

    def text_at(point: tuple[float, float]) -> str:
        x, y = point
        hits = [
            unit
            for unit in text_units
            if _near(unit.rect, (x, y, x, y), DIAGRAM_GAP_PX * 2)
        ]
        if not hits:
            return ""
        best = min(hits, key=lambda unit: (unit.rect[2] - unit.rect[0]) * (unit.rect[3] - unit.rect[1]))
        return best.parts[0].text

    arrows: list[str] = []
    for unit in ordered:
        for part in unit.parts:
            if not part.connector:
                continue
            start = names.get((unit.drawing, part.start_id), "")
            end = names.get((unit.drawing, part.end_id), "")
            if unit.ends is not None:
                start = start or text_at(unit.ends[0])
                end = end or text_at(unit.ends[1])
            if not start or not end or start == end:
                continue
            if part.head_arrow and not part.tail_arrow:
                arrows.append(f"{end} → {start}")
            elif part.head_arrow and part.tail_arrow:
                arrows.append(f"{start} ↔ {end}")
            elif part.tail_arrow:
                arrows.append(f"{start} → {end}")
            else:
                arrows.append(f"{start} — {end}")
    covered = [pictures[unit.media_id] for unit in ordered if unit.media_id in pictures]
    counts = [f"{shapes} shape{'s' if shapes != 1 else ''}"]
    if connectors:
        counts.append(f"{connectors} connector{'s' if connectors != 1 else ''}")
    if covered:
        counts.append(f"{len(covered)} picture{'s' if len(covered) != 1 else ''}")
    lines = [f"Diagram drawn with {', '.join(counts)}."]
    if texts:
        lines.append("Text on it, top to bottom: " + "; ".join(f"{index}. {text}" for index, text in enumerate(texts, start=1)))
    else:
        lines.append("It has no text.")
    if arrows:
        lines.append("Arrows and lines between boxes: " + "; ".join(arrows))
    if covered:
        lines.append(
            "Pictures it is drawn on: "
            + ", ".join(f"{item.media_id} ({item.location})" for item in covered)
            + "; describe_media reads what they show."
        )
    title = " / ".join(texts[:3])
    if len(title) > 80:
        title = title[:79] + "…"
    return title, "\n".join(lines)


def _placement(rect: tuple[float, float, float, float], rows: _Axis, cols: _Axis) -> dict[str, Any]:
    """Cells under a pixel rectangle: 1-based first, last and center row and column."""
    left, top, right, bottom = rect
    start_row, _ = rows.locate(top)
    start_col, _ = cols.locate(left)
    end_row, _ = rows.locate(max(top, bottom - 1))
    end_col, _ = cols.locate(max(left, right - 1))
    center_row, _ = rows.locate((top + bottom) / 2)
    center_col, _ = cols.locate((left + right) / 2)
    return {
        "from_row": start_row + 1,
        "from_col": start_col + 1,
        "to_row": end_row + 1,
        "to_col": end_col + 1,
        "center_row": center_row + 1,
        "center_col": center_col + 1,
        "width_px": round(right - left, 1),
        "height_px": round(bottom - top, 1),
    }


def index_workbook_media(
    workbook_path: str | Path,
    *,
    sheets: Iterable[str] | None = None,
) -> list[WorkbookMedia]:
    """Every picture, chart and diagram of the workbook (or of ``sheets``).

    An id names the sheet position and the cell under the item's center
    (``1_C30``; ``1_C30_2`` for a second item there), so it reads as a
    location and is the same whether or not ``sheets`` filters the result.
    A sheet lists its pictures first, in drawing order, then its charts, then
    its diagrams, so adding chart and diagram support kept every picture id.
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

            def next_id(place: dict[str, Any]) -> str:
                base_id = f"{sheet_number}_{cell_to_a1(place['center_row'], place['center_col'])}"
                ids_in_cell[base_id] = ids_in_cell.get(base_id, 0) + 1
                return base_id if ids_in_cell[base_id] == 1 else f"{base_id}_{ids_in_cell[base_id]}"

            charts: list[tuple[tuple[float, float, float, float], ElementTree.Element, str]] = []
            units: list[_DrawingUnit] = []
            pictures: dict[str, WorkbookMedia] = {}
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
                    frame = anchor.find("xdr:graphicFrame", _NS)
                    chart_node = frame.find(".//c:chart", _NS) if frame is not None else None
                    if chart_node is not None:
                        target = targets.get(chart_node.get(_R_ID, ""))
                        if target and not _hidden(frame.find("xdr:nvGraphicFramePr/xdr:cNvPr", _NS)):
                            charts.append((rect, frame, target[0]))
                        continue
                    # Pictures grouped with shapes share the group's anchor.
                    for picture in anchor.iter(f"{{{_NS['xdr']}}}pic"):
                        blip = picture.find(".//a:blip", _NS)
                        target = targets.get(blip.get(_R_EMBED, "")) if blip is not None else None
                        if not target or not target[0].startswith("xl/media/"):
                            continue
                        properties = picture.find("xdr:nvPicPr/xdr:cNvPr", _NS)
                        if _hidden(properties):
                            continue
                        media_path = target[0]
                        try:
                            size = archive.getinfo(media_path).file_size
                        except KeyError:
                            continue
                        if media_path not in hashes:
                            hashes[media_path] = hashlib.sha256(archive.read(media_path)).hexdigest()
                        place = _placement(rect, rows, cols)
                        suffix = posixpath.splitext(media_path)[1].lower()
                        item = WorkbookMedia(
                            media_id=next_id(place),
                            sheet=sheet_name,
                            media_path=media_path,
                            **place,
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
                        media.append(item)
                        pictures[item.media_id] = item
                        units.append(_DrawingUnit(rect=rect, drawing=drawing_path, parts=[], media_id=item.media_id))
                    parts = _shape_parts(anchor)
                    if parts:
                        units.append(
                            _DrawingUnit(
                                rect=rect,
                                drawing=drawing_path,
                                parts=parts,
                                ends=_connector_ends(anchor, rect),
                            )
                        )

            for rect, frame, chart_path in charts:
                chart = read_chart(archive, chart_path)
                if chart is None:
                    continue
                properties = frame.find("xdr:nvGraphicFramePr/xdr:cNvPr", _NS)
                place = _placement(rect, rows, cols)
                media.append(
                    WorkbookMedia(
                        media_id=next_id(place),
                        sheet=sheet_name,
                        media_path=chart_path,
                        **place,
                        name=(properties.get("name", "") if properties is not None else "").strip(),
                        description=chart.label
                        or (properties.get("descr", "") if properties is not None else "").strip(),
                        sha256=hashlib.sha256(archive.read(chart_path)).hexdigest(),
                        previewable=chart.drawable,
                        kind=KIND_CHART,
                        text=chart_summary(chart),
                    )
                )

            if any(not unit.media_id for unit in units):
                for members in _group_units(units):
                    if not _is_diagram(members):
                        continue
                    rect = (
                        min(unit.rect[0] for unit in members),
                        min(unit.rect[1] for unit in members),
                        max(unit.rect[2] for unit in members),
                        max(unit.rect[3] for unit in members),
                    )
                    place = _placement(rect, rows, cols)
                    title, text = _diagram_summary(members, pictures)
                    media_id = next_id(place)
                    media.append(
                        WorkbookMedia(
                            media_id=media_id,
                            sheet=sheet_name,
                            media_path="",
                            **place,
                            name="",
                            description=title,
                            sha256=hashlib.sha256(f"diagram\n{sheet_name}\n{media_id}\n{text}".encode()).hexdigest(),
                            previewable=True,
                            kind=KIND_DIAGRAM,
                            text=text,
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

    Charts and diagrams are images too: ``url_for`` points them at a route
    that redraws them, and the alt text carries their title.

    A picture the browser cannot show, or one without a URL, keeps only its
    location; an unknown id is dropped. A known id mentioned without the token
    brackets (``(media:1_C30)`` or a bare ``1_C30``) is a reference in the
    text, not a picture, so it reads as the picture's location.
    """
    if not mentions_media(text):
        return text
    by_id = {item.media_id: item for item in media}

    def replace(match: re.Match[str]) -> str:
        item = by_id.get(match.group(1))
        if item is None:
            return ""
        label = f"{item.description} — {item.location}" if item.kind != KIND_PICTURE and item.description else item.location
        alt = " ".join(re.sub(r"[\[\]()]", " ", label).split())
        url = url_for(item) if item.previewable else None
        if not url:
            return f"`{item.location} ({item.file_type})`"
        return f"![{alt}]({url})"

    def mention(match: re.Match[str]) -> str:
        item = by_id.get(match.group(1))
        return item.location if item is not None else match.group(0)

    rendered = MEDIA_TOKEN.sub(replace, text)
    rendered = MEDIA_MENTION.sub(mention, rendered)
    rendered = MEDIA_BARE_ID.sub(mention, rendered)
    # A dropped token leaves no trailing spaces behind.
    return re.sub(r"[ \t]+(?=\n|$)", "", rendered)
