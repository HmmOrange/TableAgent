"""Native Excel charts read as data.

An Excel chart is not a picture: ``xl/charts/chartN.xml`` names the chart
type, the cells each series plots and a cache of the plotted values. Reading
that cache tells the agent exactly what a chart shows, with no vision model.
"""

from __future__ import annotations

import datetime as dt
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from xml.etree import ElementTree

_NS = {
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
# Plot types the browser redraws (the document preview's chart renderer).
DRAWABLE_CHART_TYPES = frozenset({
    "barChart", "bar3DChart", "lineChart", "line3DChart", "areaChart", "area3DChart",
    "pieChart", "pie3DChart", "ofPieChart", "doughnutChart", "scatterChart",
})
_PLOT_TAGS = DRAWABLE_CHART_TYPES | {
    "radarChart", "bubbleChart", "stockChart", "surfaceChart", "surface3DChart",
}
# Points of one series written into a summary; a longer series is cut short.
SUMMARY_MAX_POINTS = 60
# A number format that shows a date or time ("m/d;@", "yyyy-mm-dd", "[$-409]d-mmm").
_DATE_FORMAT = re.compile(r"(?<![\\\"])[ymdhs]", re.IGNORECASE)


@dataclass(frozen=True)
class ChartSeries:
    name: str
    categories: tuple[str, ...]
    values: tuple[str, ...]
    category_ref: str = ""
    value_ref: str = ""
    category_format: str = ""


@dataclass(frozen=True)
class WorkbookChart:
    path: str
    types: tuple[str, ...]
    title: str
    series: tuple[ChartSeries, ...] = field(default_factory=tuple)
    format_code: str = ""

    @property
    def drawable(self) -> bool:
        return bool(self.types) and all(kind in DRAWABLE_CHART_TYPES for kind in self.types)

    @property
    def label(self) -> str:
        """The title, else the names of the series it plots."""
        if self.title:
            return self.title
        return ", ".join(dict.fromkeys(series.name for series in self.series if series.name))


def _rich_text(node: ElementTree.Element | None) -> str:
    if node is None:
        return ""
    paragraphs = []
    for paragraph in node.iter(f"{{{_NS['a']}}}p"):
        text = "".join(run.text or "" for run in paragraph.iter(f"{{{_NS['a']}}}t"))
        if text.strip():
            paragraphs.append(text.strip())
    return " ".join(paragraphs)


def _first(node: ElementTree.Element, *paths: str) -> ElementTree.Element | None:
    for path in paths:
        found = node.find(path, _NS)
        if found is not None:
            return found
    return None


def _points(reference: ElementTree.Element | None) -> tuple[tuple[str, ...], str, str]:
    """Cached points of a ``c:cat``/``c:val``/``c:tx`` reference, its formula and format."""
    if reference is None:
        return (), "", ""
    formula = reference.find(".//c:f", _NS)
    cache = None
    for tag in ("numCache", "strCache", "numLit", "strLit"):
        cache = reference.find(f".//c:{tag}", _NS)
        if cache is not None:
            break
    if cache is None:
        # Multi-level categories keep each level's labels; the lowest level is the label shown.
        level = reference.find(".//c:multiLvlStrCache/c:lvl", _NS)
        cache = level
    if cache is None:
        return (), (formula.text or "").strip() if formula is not None else "", ""
    try:
        count = int(cache.findtext("c:ptCount", default="", namespaces={"c": _NS["c"]}) or 0)
    except ValueError:
        count = 0
    by_index: dict[int, str] = {}
    for point in cache.findall("c:pt", _NS):
        try:
            by_index[int(point.get("idx", "0"))] = (point.findtext("c:v", default="", namespaces=_NS) or "").strip()
        except ValueError:
            continue
    count = max(count, max(by_index, default=-1) + 1)
    format_code = (cache.findtext("c:formatCode", default="", namespaces=_NS) or "").strip()
    return (
        tuple(by_index.get(index, "") for index in range(count)),
        (formula.text or "").strip() if formula is not None else "",
        format_code,
    )


# Values of a chart reference such as 'Loss Rate'!$B$2:$B$32, read from the cells.
RangeReader = Callable[[str], list[object]]


def split_reference(formula: str) -> tuple[str, str] | None:
    """``(sheet, cells)`` of a one-area chart reference; None for anything else."""
    reference = formula.strip().lstrip("(").rstrip(")")
    if "!" not in reference or "," in reference:
        return None
    sheet, cells = reference.rsplit("!", 1)
    sheet = sheet.strip()
    if sheet.startswith("'") and sheet.endswith("'"):
        sheet = sheet[1:-1].replace("''", "'")
    cells = cells.replace("$", "").strip()
    return (sheet, cells) if sheet and cells else None


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        return value.strftime("%Y-%m-%d") if value.time() == dt.time() else value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, dt.date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _from_cells(points: tuple[str, ...], formula: str, read_range: RangeReader | None) -> tuple[str, ...]:
    """The cached points, else the referenced cells (charts saved without a value cache)."""
    if points or not formula or read_range is None:
        return points
    try:
        return tuple(_cell_text(value) for value in read_range(formula))
    except Exception:
        return points


def read_chart(
    archive: zipfile.ZipFile,
    chart_path: str,
    *,
    read_range: RangeReader | None = None,
) -> WorkbookChart | None:
    """The chart part as types, title and series; None when it is not a chart.

    A chart written without a value cache (by openpyxl and some report
    tools) is read from its cells when ``read_range`` is given.
    """
    try:
        root = ElementTree.fromstring(archive.read(chart_path))
    except (KeyError, ElementTree.ParseError):
        return None
    chart = root.find("c:chart", _NS)
    plot = chart.find("c:plotArea", _NS) if chart is not None else None
    if chart is None or plot is None:
        return None
    types: list[str] = []
    series: list[ChartSeries] = []
    format_code = ""
    for group in plot:
        tag = group.tag.rsplit("}", 1)[-1]
        if tag not in _PLOT_TAGS:
            continue
        if tag not in types:
            types.append(tag)
        for ser in group.findall("c:ser", _NS):
            name_points, name_ref, _ = _points(ser.find("c:tx", _NS))
            name_points = _from_cells(name_points, name_ref, read_range)
            name = " ".join(point for point in name_points if point) or (ser.findtext("c:tx/c:v", default="", namespaces=_NS) or "")
            categories, category_ref, category_format = _points(_first(ser, "c:cat", "c:xVal"))
            categories = _from_cells(categories, category_ref, read_range)
            values, value_ref, value_format = _points(_first(ser, "c:val", "c:yVal"))
            values = _from_cells(values, value_ref, read_range)
            format_code = format_code or value_format
            series.append(
                ChartSeries(
                    name=name.strip(),
                    categories=categories,
                    values=values,
                    category_ref=category_ref,
                    value_ref=value_ref,
                    category_format=category_format,
                )
            )
    title = ""
    title_node = chart.find("c:title", _NS)
    deleted = chart.find("c:autoTitleDeleted", _NS)
    if title_node is not None:
        title = _rich_text(title_node.find("c:tx/c:rich", _NS))
        if not title:
            cached, _, _ = _points(title_node.find("c:tx", _NS))
            title = " ".join(point for point in cached if point)
    elif (deleted is None or deleted.get("val") in {"0", "false"}) and len(series) == 1:
        # Excel titles a one-series chart with the series name unless the title was deleted.
        title = series[0].name
    return WorkbookChart(
        path=chart_path,
        types=tuple(types),
        title=title.strip(),
        series=tuple(series),
        format_code=format_code,
    )


def _is_date_format(format_code: str) -> bool:
    plain = re.sub(r"\[[^\]]*\]|\"[^\"]*\"", "", format_code or "")
    return bool(_DATE_FORMAT.search(plain)) and "General" not in plain


def _format_value(value: str, format_code: str) -> str:
    if not value:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if _is_date_format(format_code) and 0 < number < 2_958_466:
        moment = dt.datetime(1899, 12, 30) + dt.timedelta(days=number)
        return moment.strftime("%Y-%m-%d %H:%M" if number % 1 else "%Y-%m-%d")
    if format_code.endswith("%"):
        decimals = format_code.split(".", 1)[1].count("0") if "." in format_code else 0
        return f"{number * 100:.{decimals}f}%"
    if number.is_integer():
        return str(int(number))
    return f"{number:.6g}"


def chart_summary(chart: WorkbookChart, *, max_points: int = SUMMARY_MAX_POINTS) -> str:
    """What the chart plots, in lines the agent can read and quote."""
    kinds = ", ".join(kind.removesuffix("Chart") for kind in chart.types) or "unknown"
    lines = [f"Chart type: {kinds}" + (f"; title: {chart.title}" if chart.title else "; no title")]
    if not chart.series:
        lines.append("The chart has no series.")
    for index, series in enumerate(chart.series, start=1):
        head = f"Series {index}" + (f" \"{series.name}\"" if series.name else "")
        refs = ", ".join(
            part for part in (
                f"values {series.value_ref}" if series.value_ref else "",
                f"categories {series.category_ref}" if series.category_ref else "",
            ) if part
        )
        lines.append(head + (f" ({refs})" if refs else "") + ":")
        count = max(len(series.values), len(series.categories))
        for point in range(min(count, max_points)):
            label = (
                _format_value(series.categories[point], series.category_format)
                if point < len(series.categories) and series.categories[point]
                else str(point + 1)
            )
            value = series.values[point] if point < len(series.values) else ""
            lines.append(f"  {label}: {_format_value(value, chart.format_code)}")
        if count > max_points:
            lines.append(f"  … {count - max_points} more points; read the value cells for the rest.")
    lines.append("The values are the copy Excel saved with the chart; the cells named above hold the current data.")
    return "\n".join(lines)
