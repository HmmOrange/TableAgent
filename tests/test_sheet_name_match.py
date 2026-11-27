"""Sheet names written loosely in a structure still reach the workbook's sheets and pictures."""
from pathlib import Path

import openpyxl
import pytest
from openpyxl.drawing.image import Image as SheetImage
from PIL import Image

from TableAgent.environment.qa_env import QAEnvironment


def _structure(sheet: str) -> str:
    return f"""table1:
  id: gallery
  name: Gallery
  sheet: {sheet}
  headers:
  - id: name
    label: Name
    description: Item name
    orientation: column
    header_range: A1
    data_range: A2:A3
    sub_headers: []
  - id: photo
    label: Photo
    description: Item picture
    orientation: column
    header_range: B1
    data_range: B2:B3
    sub_headers: []
"""


def _workbook(tmp_path: Path, sheet: str, *, extra_sheets: tuple[str, ...] = ()) -> Path:
    workbook = openpyxl.Workbook()
    for name in extra_sheets:
        workbook.create_sheet(name)["A1"] = "other"
    gallery = workbook.active
    gallery.title = sheet
    gallery.append(["Name", "Photo"])
    gallery.append(["Tom", None])
    gallery.append(["Jerry", None])
    picture = tmp_path / "tom.png"
    Image.new("RGB", (20, 10), "blue").save(picture)
    gallery.add_image(SheetImage(str(picture)), "B2")
    # A picture outside the table, as a chart or diagram beside it would be.
    gallery.add_image(SheetImage(str(picture)), "E2")
    path = tmp_path / "book.xlsx"
    workbook.save(path)
    workbook.close()
    return path


def _env(tmp_path: Path, written: str, actual: str, **kwargs) -> QAEnvironment:
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(_structure(written), encoding="utf-8")
    return QAEnvironment(str(structure_path), str(_workbook(tmp_path, actual, **kwargs)))


@pytest.mark.parametrize(
    ("written", "actual"),
    [
        ("Sheet2", "Sheet2.."),
        ("'재공현황 (4월)'", "재공 현황 (4월) "),
        ("price list 03.14", "Price List 03.14"),
    ],
)
def test_pictures_are_found_when_the_structure_writes_the_sheet_name_loosely(tmp_path, written, actual):
    env = _env(tmp_path, written, actual)
    try:
        assert env.resolve_sheet_name(written.strip("'")) == actual
        assert env.get_sheet(written.strip("'")).title == actual
        media = env.operators.list_media()
        frame = env.operators.read_table_as_dataframe("gallery", has_headers=True)
    finally:
        env.workbook.close()

    assert sorted(media["sheet"]) == [actual, actual]
    assert "media: Pictures placed on the loaded sheets" in env.operators.operator_catalog()
    # The picture on a table row still reaches that row.
    assert list(frame["media"]) == ["Photo: [[media:1_B2]]", None]


def test_an_exact_sheet_name_wins_and_an_unclear_one_resolves_to_none(tmp_path):
    env = _env(tmp_path, "Sheet2", "Sheet2..", extra_sheets=("Sheet2", "Sheet 3", "sheet3"))
    try:
        assert env.resolve_sheet_name("Sheet2") == "Sheet2"
        assert env.resolve_sheet_name("Sheet2..") == "Sheet2.."
        # "Sheet 3" and "sheet3" both fit "SHEET-3": no guess.
        assert env.resolve_sheet_name("SHEET-3") is None
        assert env.resolve_sheet_name("") is None
        assert env.get_sheet("missing") is None
    finally:
        env.workbook.close()


def test_pictures_on_sheets_that_were_not_loaded_are_named_in_the_catalog(tmp_path):
    workbook = openpyxl.Workbook()
    data = workbook.active
    data.title = "Data"
    data.append(["Name", "Photo"])
    data.append(["Tom", None])
    data.append(["Jerry", None])
    chart_sheet = workbook.create_sheet("Sheet2..")
    picture = tmp_path / "chart.png"
    Image.new("RGB", (20, 10), "blue").save(picture)
    chart_sheet.add_image(SheetImage(str(picture)), "C3")
    path = tmp_path / "book.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(_structure("Data"), encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(path))
    try:
        catalog = env.operators.operator_catalog()
        listed = env.operators.list_media(sheet="Sheet2")
    finally:
        env.workbook.close()

    assert "other sheets of this workbook do:\n- Sheet2..: 1 picture" in catalog
    # The full picture guide stays out: the loaded sheet has none.
    assert "Pictures placed on the loaded sheets" not in catalog
    assert list(listed["token"]) == ["[[media:2_C3]]"]


def test_a_workbook_without_pictures_keeps_its_catalog(tmp_path):
    env = _env(tmp_path, "Data", "Data")
    try:
        # Drop the pictures this helper adds: no sheet holds any.
        env.operators._media._media = []
        env.operators._media._areas = None
        catalog = env.operators.operator_catalog()
    finally:
        env.workbook.close()
    assert "media:" not in catalog
