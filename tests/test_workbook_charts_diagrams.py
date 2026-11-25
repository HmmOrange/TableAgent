import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.drawing.image import Image as SheetImage
from PIL import Image

from service.runtime import render_answer_media
from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.utils.workbook_charts import chart_summary, read_chart
from TableAgent.utils.workbook_media import index_workbook_media

from tests.test_workbook_media import STRUCTURE


def _picture(path: Path, size: tuple[int, int], color: str) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _anchor(from_col: int, from_row: int, to_col: int, to_row: int, body: str) -> str:
    return (
        "<xdr:twoCellAnchor>"
        f"<xdr:from><xdr:col>{from_col}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{from_row}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>"
        f"<xdr:to><xdr:col>{to_col}</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>{to_row}</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>"
        f"{body}<xdr:clientData/></xdr:twoCellAnchor>"
    )


def _box(shape_id: int, text: str, *, text_box: bool = False, geometry: str = "flowChartProcess") -> str:
    flags = ' txBox="1"' if text_box else ""
    body = f"<xdr:txBody><a:bodyPr/><a:p><a:r><a:t>{text}</a:t></a:r></a:p></xdr:txBody>" if text else ""
    return (
        f'<xdr:sp macro="" textlink=""><xdr:nvSpPr><xdr:cNvPr id="{shape_id}" name="Shape {shape_id}"/>'
        f"<xdr:cNvSpPr{flags}/></xdr:nvSpPr><xdr:spPr><a:prstGeom prst=\"{geometry}\"><a:avLst/></a:prstGeom></xdr:spPr>"
        f"{body}</xdr:sp>"
    )


def _arrow(shape_id: int, start: int, end: int, *, arrow_at: str = "tailEnd") -> str:
    return (
        f'<xdr:cxnSp macro=""><xdr:nvCxnSpPr><xdr:cNvPr id="{shape_id}" name="Arrow {shape_id}"/>'
        f'<xdr:cNvCxnSpPr><a:stCxn id="{start}" idx="2"/><a:endCxn id="{end}" idx="0"/></xdr:cNvCxnSpPr></xdr:nvCxnSpPr>'
        '<xdr:spPr><a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom>'
        f'<a:ln><a:{arrow_at} type="triangle"/></a:ln></xdr:spPr></xdr:cxnSp>'
    )


def _rewrite(path: Path, part: str, change) -> None:
    """Replace one package part of a saved workbook with ``change(old text)``."""
    with zipfile.ZipFile(path) as original:
        parts = {item.filename: original.read(item.filename) for item in original.infolist()}
    parts[part] = change(parts[part].decode("utf-8")).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as patched:
        for name, data in parts.items():
            patched.writestr(name, data)


def _add_shapes(path: Path, drawing: str, anchors: list[str]) -> None:
    """Append shape anchors to a drawing part openpyxl wrote (it cannot write shapes itself)."""

    def change(xml: str) -> str:
        # openpyxl writes the drawing namespace as the default one; the anchors use `xdr:`.
        xml = xml.replace(
            "<wsDr ",
            '<wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" ',
            1,
        )
        return xml.replace("</wsDr>", "".join(anchors) + "</wsDr>")

    _rewrite(path, drawing, change)


def _workbook(tmp_path: Path) -> Path:
    workbook = openpyxl.Workbook()
    gallery = workbook.active
    gallery.title = "Gallery"
    gallery.append(["Name", "Photo"])
    gallery.append(["Tom", None])
    gallery.append(["Jerry", None])
    gallery.add_image(SheetImage(str(_picture(tmp_path / "tom.png", (20, 10), "blue"))), "B2")

    stats = workbook.create_sheet("Stats")
    stats.append(["Day", "Output"])
    for day in range(1, 6):
        stats.append([f"09/{day:02d}", 1000 + day * 10])
    bars = BarChart()
    bars.title = "Output by day"
    bars.add_data(Reference(stats, min_col=2, min_row=1, max_row=6), titles_from_data=True)
    bars.set_categories(Reference(stats, min_col=1, min_row=2, max_row=6))
    stats.add_chart(bars, "D2")

    flow = workbook.create_sheet("Flow")
    # Pictures give the sheet its drawing part; shapes are added to it after saving.
    flow.add_image(SheetImage(str(_picture(tmp_path / "photo.png", (80, 38), "green"))), "H2")
    flow.add_image(SheetImage(str(_picture(tmp_path / "burr.png", (80, 38), "gray"))), "L2")
    path = tmp_path / "visuals.xlsx"
    workbook.save(path)
    workbook.close()
    _add_shapes(
        path,
        "xl/drawings/drawing3.xml",
        [
            _anchor(1, 1, 3, 3, _box(2, "Start")),
            _anchor(1, 4, 3, 6, _box(3, "Check part")),
            _anchor(2, 3, 2, 4, _arrow(4, 2, 3)),
            # An empty text box draws nothing.
            _anchor(9, 19, 11, 21, _box(5, "", text_box=True)),
            # A caption under the photo at H2 touches it without covering it.
            _anchor(7, 3, 9, 4, _box(6, "Photo caption", text_box=True)),
            # A label written on the photo at L2.
            _anchor(11, 1, 12, 2, _box(7, "Burr here", text_box=True)),
        ],
    )
    return path


@pytest.fixture
def visuals_env(tmp_path: Path):
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(_workbook(tmp_path)))
    try:
        yield env
    finally:
        env.workbook.close()


def test_charts_and_diagrams_are_indexed_after_the_pictures_of_their_sheet(tmp_path: Path):
    media = index_workbook_media(_workbook(tmp_path))
    by_kind = {kind: [item for item in media if item.kind == kind] for kind in ("picture", "chart", "diagram")}

    # Pictures keep the ids they had before charts and diagrams were indexed.
    assert [item.media_id for item in by_kind["picture"]] == ["1_B2", "3_H2", "3_L2"]
    [chart] = by_kind["chart"]
    assert chart.sheet == "Stats"
    assert chart.media_path == "xl/charts/chart1.xml"
    assert chart.description == "Output by day"
    assert chart.previewable and chart.file_type == "chart"
    # openpyxl saves no value cache, so the index names the cells; the operator reads them.
    assert "values 'Stats'!$B$2:$B$6" in chart.text

    diagrams = sorted(by_kind["diagram"], key=lambda item: item.from_col)
    # The boxes joined by an arrow, and the label on the photo; not the caption beside a
    # photo, nor the empty text box.
    assert [item.cells for item in diagrams] == ["B2:C6", "L2:M3"]
    flow, label = diagrams
    assert flow.description == "Start / Check part"
    assert "Arrows and lines between boxes: Start → Check part" in flow.text
    assert "Pictures it is drawn on: 3_L2 (Flow!L2:M3)" in label.text
    assert all(item.previewable and item.media_path == "" for item in diagrams)


def test_an_arrowhead_at_the_start_reverses_the_arrow(tmp_path: Path):
    path = _workbook(tmp_path)
    _rewrite(
        path,
        "xl/drawings/drawing3.xml",
        lambda xml: xml.replace('<a:tailEnd type="triangle"/>', '<a:headEnd type="triangle"/>'),
    )

    [flow] = [item for item in index_workbook_media(path) if item.kind == "diagram" and item.cells == "B2:C6"]
    assert "Check part → Start" in flow.text


def test_chart_summary_reads_dates_and_percentages(tmp_path: Path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Day", "Share"])
    for day in range(1, 4):
        sheet.append([46112 + day, day / 10])
        sheet.cell(row=day + 1, column=1).number_format = "m/d;@"
        sheet.cell(row=day + 1, column=2).number_format = "0.0%"
    line = LineChart()
    line.add_data(Reference(sheet, min_col=2, min_row=1, max_row=4), titles_from_data=True)
    line.set_categories(Reference(sheet, min_col=1, min_row=2, max_row=4))
    sheet.add_chart(line, "D2")
    path = tmp_path / "line.xlsx"
    workbook.save(path)
    workbook.close()
    # openpyxl writes no value cache; give the chart the cache Excel would have saved
    # (openpyxl writes the chart namespace as the default one).
    def add_cache(xml: str) -> str:
        xml = re.sub(
            r"(<cat><numRef><f>[^<]*</f>)",
            r'\1<numCache><formatCode>m/d;@</formatCode><ptCount val="3"/>'
            r'<pt idx="0"><v>46113</v></pt><pt idx="1"><v>46114</v></pt>'
            r'<pt idx="2"><v>46115</v></pt></numCache>',
            xml,
        )
        return re.sub(
            r"(<val><numRef><f>[^<]*</f>)",
            r'\1<numCache><formatCode>0.0%</formatCode><ptCount val="3"/>'
            r'<pt idx="0"><v>0.1</v></pt><pt idx="1"><v>0.2</v></pt>'
            r'<pt idx="2"><v>0.3</v></pt></numCache>',
            xml,
        )

    _rewrite(path, "xl/charts/chart1.xml", add_cache)

    with zipfile.ZipFile(path) as archive:
        chart = read_chart(archive, "xl/charts/chart1.xml")
    assert chart is not None and chart.types == ("lineChart",)
    summary = chart_summary(chart)
    assert "2026-04-01: 10.0%" in summary
    assert "2026-04-03: 30.0%" in summary


def test_media_operator_lists_and_reads_charts_and_diagrams_without_vision(visuals_env):
    def no_vision(**_):
        raise AssertionError("charts and diagrams are read from the workbook")

    visuals_env.vision_client = SimpleNamespace(generate_with_image=no_vision)
    stats = visuals_env.operators.list_media(sheet="Stats")
    flow = visuals_env.operators.list_media(sheet="Flow")

    assert list(stats["kind"]) == ["chart"]
    assert stats.iloc[0]["name"] == "Output by day"
    assert "diagram" in set(flow["kind"])
    chart_id = stats.iloc[0]["id"]
    described = visuals_env.operators.describe_media(chart_id)
    assert described.startswith("Chart at Stats!")
    assert "09/05: 1050" in described
    [flow_id] = list(flow[flow["cells"] == "B2:C6"]["id"])
    assert "Start → Check part" in visuals_env.operators.describe_media(flow_id)
    # Table rows still carry only their own pictures.
    frame = visuals_env.operators.read_table_as_dataframe("gallery", has_headers=True)
    assert list(frame["media"]) == ["Photo: [[media:1_B2]]", None]


def test_chart_and_diagram_tokens_link_the_drawings_of_their_cells(tmp_path: Path):
    workbook = _workbook(tmp_path)
    media = index_workbook_media(workbook)
    chart = next(item for item in media if item.kind == "chart")
    flow = next(item for item in media if item.kind == "diagram" and item.cells == "B2:C6")

    rendered = render_answer_media(
        f"Chart: [[media:{chart.media_id}]]\nFlow: [[media:{flow.media_id}]]",
        workbook,
        [{"document_id": "doc-1"}],
    )

    assert rendered == (
        f"Chart: ![Output by day — Stats!{chart.cells}]"
        f"(/api/v1/documents/doc-1/spreadsheet-preview/drawing-area?sheet=Stats&range={chart.cells.replace(':', '%3A')})\n"
        "Flow: ![Start / Check part — Flow!B2:C6]"
        "(/api/v1/documents/doc-1/spreadsheet-preview/drawing-area?sheet=Flow&range=B2%3AC6)"
    )
    # Without a stored document there is nothing to draw: the location stays.
    assert render_answer_media(f"[[media:{chart.media_id}]]", workbook, []) == f"`Stats!{chart.cells} (chart)`"
