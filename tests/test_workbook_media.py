from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest
from openpyxl.drawing.image import Image as SheetImage
from PIL import Image

from service.runtime import render_answer_media
from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.utils.workbook_media import index_workbook_media


STRUCTURE = """table1:
  id: gallery
  name: Gallery
  sheet: Gallery
  headers:
  - id: name
    label: Name
    description: Character name
    orientation: column
    header_range: A1
    data_range: A2:A3
    sub_headers: []
  - id: photo
    label: Photo
    description: Character picture
    orientation: column
    header_range: B1
    data_range: B2:B3
    sub_headers: []
"""


def _picture(path: Path, color: str) -> Path:
    Image.new("RGB", (20, 10), color).save(path)
    return path


def _workbook(tmp_path: Path, *, with_pictures: bool = True) -> Path:
    workbook = openpyxl.Workbook()
    gallery = workbook.active
    gallery.title = "Gallery"
    gallery.append(["Name", "Photo"])
    gallery.append(["Tom", None])
    gallery.append(["Jerry", None])
    stats = workbook.create_sheet("Stats")
    stats["A1"] = "Featured picture"
    if with_pictures:
        tom = _picture(tmp_path / "tom.png", "blue")
        jerry = _picture(tmp_path / "jerry.png", "brown")
        gallery.add_image(SheetImage(str(tom)), "B2")
        gallery.add_image(SheetImage(str(jerry)), "B3")
        # The same picture placed again on another sheet, outside any table.
        stats.add_image(SheetImage(str(jerry)), "A2")
    path = tmp_path / "gallery.xlsx"
    workbook.save(path)
    workbook.close()
    return path


@pytest.fixture
def gallery_env(tmp_path: Path):
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(_workbook(tmp_path)))
    try:
        yield env
    finally:
        env.workbook.close()


def test_pictures_are_indexed_with_the_cells_they_cover(tmp_path: Path):
    media = index_workbook_media(_workbook(tmp_path))

    assert [(item.media_id, item.location) for item in media] == [
        ("1_B2", "Gallery!B2"),
        ("1_B3", "Gallery!B3"),
        ("2_A2", "Stats!A2"),
    ]
    assert media[1].sha256 == media[2].sha256 != media[0].sha256
    assert all(item.previewable for item in media)
    # Ids do not depend on which sheets are asked for.
    assert [item.media_id for item in index_workbook_media(_workbook(tmp_path), sheets=["Stats"])] == ["2_A2"]


def test_table_rows_carry_their_picture_tokens(gallery_env):
    frame = gallery_env.operators.read_table_as_dataframe("gallery", has_headers=True)

    assert list(frame["name"]) == ["Tom", "Jerry"]
    assert list(frame["media"]) == ["Photo: [[media:1_B2]]", "Photo: [[media:1_B3]]"]
    assert "media: Pictures placed on the loaded sheets" in gallery_env.operators.operator_catalog()


def test_tables_without_pictures_are_unchanged(tmp_path: Path):
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(_workbook(tmp_path, with_pictures=False)))
    try:
        frame = env.operators.read_table_as_dataframe("gallery", has_headers=True)
        catalog = env.operators.operator_catalog()
    finally:
        env.workbook.close()

    assert list(frame.columns) == ["name", "photo"]
    assert "Pictures placed" not in catalog


def test_identical_pictures_are_found_on_other_sheets(gallery_env):
    listed = gallery_env.operators.list_media()
    same = gallery_env.operators.find_same_media("2_A2")
    stats = gallery_env.operators.list_media(sheet="Stats")

    assert list(listed["id"]) == ["1_B2", "1_B3"]
    assert list(listed["row"]) == [2, 3]
    assert list(listed["header"]) == ["Photo", "Photo"]
    assert list(same["id"]) == ["1_B3"]
    assert stats.iloc[0]["text_above"] == "A1: Featured picture"
    assert stats.iloc[0]["same_as"] == "1_B3"


def test_describe_media_uses_the_vision_client_once_per_picture(gallery_env):
    calls = []

    def generate_with_image(*, prompt, image_path):
        calls.append(Path(image_path).read_bytes()[:8])
        return SimpleNamespace(content="A brown square.")

    gallery_env.vision_client = SimpleNamespace(generate_with_image=generate_with_image)

    assert gallery_env.operators.describe_media("1_B3") == "A brown square."
    assert gallery_env.operators.describe_media("[[media:1_B3]]") == "A brown square."
    assert calls == [b"\x89PNG\r\n\x1a\n"]
    with pytest.raises(KeyError):
        gallery_env.operators.describe_media("img9_9")


def test_answer_tokens_become_images_of_the_stored_document(tmp_path: Path):
    workbook = _workbook(tmp_path)
    answer = "Jerry: [[media:1_B3]]\nUnknown: [[media:img9_9]]"

    rendered = render_answer_media(answer, workbook, [{"document_id": "doc-1"}])

    assert rendered == (
        "Jerry: ![Gallery!B3](/api/v1/documents/doc-1/spreadsheet-preview/media?path=xl/media/image2.png)\n"
        "Unknown:"
    )
    assert render_answer_media("No pictures.", workbook, []) == "No pictures."


TWO_BLOCK_STRUCTURE = """table1:
  id: standard
  name: Standard
  sheet: Standard
  headers:
  - id: item
    label: Item
    description: Inspection item
    orientation: column
    header_range: A1
    data_range: A2:A3
    sub_headers: []
  - id: phenomenon
    label: Phenomenon
    description: Picture of the defect
    orientation: column
    header_range: B1
    data_range: null
    sub_headers: []
  - id: revision
    label: Revision
    description: Revision history
    orientation: column
    header_range: A4:B4
    data_range: A5:B5
    sub_headers: []
"""


def test_pictures_follow_rows_of_every_table_block(tmp_path: Path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Standard"
    for row in (["Item", None], ["Copper burr", None], ["Open", None], ["Revision", None], ["Rev 1", None]):
        sheet.append(row)
    sheet.add_image(SheetImage(str(_picture(tmp_path / "logo.png", "red"))), "B1")
    sheet.add_image(SheetImage(str(_picture(tmp_path / "burr.png", "gray"))), "B2")
    path = tmp_path / "standard.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(TWO_BLOCK_STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(path))
    try:
        raw = env.operators.read_table_as_dataframe("standard", has_headers=False)
        cells = env.operators.read_sheet_as_dataframe("Standard", min_row=2, max_row=3)
        ranged = env.operators.read_range_as_dataframe("A1:B3", sheet="Standard")
        listed = env.operators.list_media()
    finally:
        env.workbook.close()

    # The header logo (row 1) is not a row picture; the burr photo is on row 2.
    assert list(raw["media"]) == [None, "Phenomenon: [[media:1_B2]]", None, None, None]
    assert list(cells["media"]) == ["Phenomenon: [[media:1_B2]]", None]
    assert list(ranged["media"]) == ["Phenomenon: [[media:1_B2]]", None]
    assert list(listed["row"]) == [None, 2]
    assert listed.iloc[1]["header"] == "Phenomenon"
