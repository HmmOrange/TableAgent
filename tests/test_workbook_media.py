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
    # Rows with pictures are labelled by sheet row, the `row` that list_media reports.
    assert list(frame.index) == [2, 3]
    assert "media: Pictures placed on the loaded sheets" in gallery_env.operators.operator_catalog()
    # Pictures on table rows come with the rows; the sheet gets no count line.
    assert "outside table rows" not in gallery_env.operators.operator_catalog()


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
    assert list(frame.index) == [0, 1]
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

    described = gallery_env.operators.describe_media("1_B3")
    assert described.startswith("A brown square.")
    # The reading also names where the identical picture is placed.
    assert "Stats!A2 (id 2_A2)" in described
    # A picture on a table row comes with the values on the same row.
    assert "Values on the same table row (gallery, sheet row 3): Name: Jerry" in described
    assert gallery_env.operators.describe_media("[[media:1_B3]]") == described
    # Outside tables there is no row to report.
    assert "Values on the same table row" not in gallery_env.operators.describe_media("2_A2")
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
    # An id written without the token brackets reads as the picture's location.
    assert render_answer_media("From the chart (media:1_B3) and `media:1_B2`; media:img9_9.", workbook, []) == (
        "From the chart (Gallery!B3) and Gallery!B2; media:img9_9."
    )
    # A bare id of a picture in the workbook reads as its location too; other numbers stay.
    assert render_answer_media("See picture 1_B3 (row 2_000, code 9_X1).", workbook, []) == (
        "See picture Gallery!B3 (row 2_000, code 9_X1)."
    )


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
        catalog = env.operators.operator_catalog()
    finally:
        env.workbook.close()

    # The header logo (row 1) is not a row picture; the burr photo is on row 2.
    assert list(raw["media"]) == [None, "Phenomenon: [[media:1_B2]]", None, None, None]
    assert list(cells["media"]) == ["Phenomenon: [[media:1_B2]]", None]
    assert list(ranged["media"]) == ["Phenomenon: [[media:1_B2]]", None]
    assert list(listed["row"]) == [None, 2]
    assert "- Standard: 1" in catalog
    assert listed.iloc[1]["header"] == "Phenomenon"


def test_nearby_text_names_the_cell_it_comes_from(tmp_path: Path):
    workbook = openpyxl.Workbook()
    gallery = workbook.active
    gallery.title = "Gallery"
    gallery.append(["Name", "Photo"])
    guide = workbook.create_sheet("Guide")
    guide["A1"] = "Work steps"
    guide["B1"] = "Owner"
    guide["B5"] = "Section title"
    guide.merge_cells("B5:H5")
    guide.add_image(SheetImage(str(_picture(tmp_path / "steps.png", "blue"))), "B2")
    guide.add_image(SheetImage(str(_picture(tmp_path / "wide.png", "green"))), "F6")
    path = tmp_path / "guide.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(path))
    try:
        listed = env.operators.list_media(sheet="Guide")
    finally:
        env.workbook.close()

    # Sheet order, and each label carries the address of the cell holding the text.
    assert list(listed["id"]) == ["2_B2", "2_F6"]
    assert listed.iloc[0]["text_above"] == "A1: Work steps | B1: Owner"
    # A title merged across a wide row is not the label of every picture under it.
    assert listed.iloc[1]["text_above"] is None


def test_picked_list_columns_are_all_printed(gallery_env):
    import pandas as pd

    listed = gallery_env.operators.list_media(sheet="Stats")
    with pd.option_context("display.max_columns", 0, "display.width", 40):
        picked = str(listed[["id", "cells", "kind", "text_above", "token"]])
        everything = str(listed)

    assert "A1: Featured picture" in picked and "..." not in picked.splitlines()[0]
    # The full list keeps pandas' own printout.
    assert "..." in everything.splitlines()[0]


def test_describe_media_reads_each_picture_once(gallery_env):
    calls = []

    def generate_with_image(*, prompt, image_path):
        calls.append(prompt)
        return SimpleNamespace(content="A brown square.")

    gallery_env.vision_client = SimpleNamespace(generate_with_image=generate_with_image)

    assert gallery_env.operators.describe_media("1_B3", question="What colour?").startswith("A brown square.")
    # Another question, and the identical copy on another sheet, reuse the first reading.
    assert gallery_env.operators.describe_media("1_B3", question="What shape?").startswith("A brown square.")
    assert gallery_env.operators.describe_media("2_A2").startswith("A brown square.")
    assert len(calls) == 1


def test_stacked_header_blocks_each_keep_their_rows(tmp_path: Path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Standard"
    for row in (["Item", None], ["Copper burr", None], ["Open", None], ["Revision", None], ["Rev 1", None]):
        sheet.append(row)
    sheet.add_image(SheetImage(str(_picture(tmp_path / "burr.png", "gray"))), "B2")
    path = tmp_path / "standard.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(TWO_BLOCK_STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(path))
    try:
        frame = env.operators.read_table_as_dataframe("standard", has_headers=True)
    finally:
        env.workbook.close()

    # Both blocks are read, each named by its own headers, indexed by sheet row.
    assert list(frame.index) == [2, 3, 5]
    assert list(frame["block"]) == ["Item", "Item", "Revision"]
    assert list(frame["item"]) == ["Copper burr", "Open", None]
    assert list(frame["revision"]) == [None, None, "Rev 1"]
    assert list(frame["media"]) == ["Phenomenon: [[media:1_B2]]", None, None]
    assert list(frame.columns)[-1] == "media"


def test_a_picture_over_a_merged_value_goes_with_the_record_row(tmp_path: Path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Standard"
    for row in (["Item", "Phenomenon"], ["Open", None], [None, None], ["Short", None]):
        sheet.append(row)
    # "Open" and its phenomenon span rows 2-3; the photo sits on row 3 inside the merge.
    sheet.merge_cells("A2:A3")
    sheet.merge_cells("B2:B3")
    sheet.add_image(SheetImage(str(_picture(tmp_path / "open.png", "gray"))), "B3")
    path = tmp_path / "standard.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(
        STRUCTURE.replace("Gallery", "Standard").replace("A2:A3", "A2:A4").replace("B2:B3", "B2:B4"),
        encoding="utf-8",
    )
    env = QAEnvironment(str(structure_path), str(path))
    try:
        frame = env.operators.read_table_as_dataframe("gallery", has_headers=True)
        listed = env.operators.list_media()
    finally:
        env.workbook.close()

    assert frame.loc[2, "media"] == "Photo: [[media:1_B3]]"
    assert frame.loc[3, "media"] is None
    assert list(listed["row"]) == [2]


def test_tables_of_related_sheets_are_readable(tmp_path: Path):
    stats_structure = tmp_path / "001-stats.yaml"
    stats_structure.write_text(
        """table1:
  id: stats
  name: Stats
  sheet: Stats
  headers:
  - id: title
    label: Title
    description: Title
    orientation: column
    header_range: A1
    data_range: A2
    sub_headers: []
""",
        encoding="utf-8",
    )
    gallery_structure = tmp_path / "002-gallery.yaml"
    gallery_structure.write_text(STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(stats_structure), str(_workbook(tmp_path)), related_structure_paths=[gallery_structure])
    try:
        tables = env.operators.list_tables()
        frame = env.operators.read_table_as_dataframe("gallery", has_headers=True)
    finally:
        env.workbook.close()

    # The planner is shown the related sheet, so its table can be listed and read too.
    assert tables == ["stats", "gallery"]
    assert list(frame["name"]) == ["Tom", "Jerry"]


def test_a_caption_belongs_to_the_picture_below_it(tmp_path: Path):
    workbook = openpyxl.Workbook()
    gallery = workbook.active
    gallery.title = "Gallery"
    gallery.append(["Name", "Photo"])
    guide = workbook.create_sheet("Guide")
    guide["A5"] = "Work steps"
    tall = tmp_path / "tall.png"
    Image.new("RGB", (60, 160), "blue").save(tall)
    # A tall photo on the right shares rows with the caption of the diagram under A5.
    guide.add_image(SheetImage(str(tall)), "B1")
    guide.add_image(SheetImage(str(_picture(tmp_path / "diagram.png", "red"))), "A6")
    path = tmp_path / "guide.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(path))
    try:
        listed = env.operators.list_media(sheet="Guide")
    finally:
        env.workbook.close()

    diagram = listed[listed["cells"] == "A6"].iloc[0]
    photo = listed[listed["cells"].str.startswith("B1")].iloc[0]
    assert diagram["text_above"] == "A5: Work steps"
    assert photo["text_left"] is None


def test_row_pictures_described_but_left_out_are_restored(gallery_env):
    gallery_env.vision_client = SimpleNamespace(
        generate_with_image=lambda *, prompt, image_path: SimpleNamespace(content="A brown mouse.")
    )
    media = gallery_env.operators._media

    gallery_env.operators.describe_media("1_B3")
    assert media.pictures_left_out("Jerry is a brown mouse.") == ["[[media:1_B3]]"]
    assert media.pictures_left_out("Jerry: [[media:1_B3]]") == []
    # Pictures read on several rows were a search, not the reported row.
    gallery_env.operators.describe_media("1_B2")
    assert media.pictures_left_out("Jerry is a brown mouse.") == []


def test_find_media_ranks_pictures_by_their_caption(tmp_path: Path):
    workbook = openpyxl.Workbook()
    gallery = workbook.active
    gallery.title = "Gallery"
    gallery.append(["Name", "Photo"])
    guide = workbook.create_sheet("Guide")
    guide["A1"] = "Work steps"
    guide["D1"] = "Tool photo"
    drawing = tmp_path / "drawing.png"
    Image.new("RGB", (40, 40), "white").save(drawing)
    guide.add_image(SheetImage(str(drawing)), "A2")
    guide.add_image(SheetImage(str(_picture(tmp_path / "tool.png", "gray"))), "D2")
    path = tmp_path / "guide.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(path))
    try:
        ranked = env.operators.find_media("Show the work steps", sheet="Guide")
        listed = env.operators.list_media(sheet="Guide")
    finally:
        env.workbook.close()

    # Every picture is listed, the matching caption first.
    assert [cells.split(":")[0] for cells in ranked["cells"]] == ["A2", "D2"]
    assert list(ranked["score"])[-1] == 0
    kinds = {cells.split(":")[0]: kind for cells, kind in zip(listed["cells"], listed["kind"])}
    assert kinds == {"A2": "drawing", "D2": "photo"}


def test_pictures_join_the_value_of_their_header(tmp_path: Path):
    workbook = openpyxl.Workbook()
    gallery = workbook.active
    gallery.title = "Gallery"
    gallery.append(["Name", "Photo"])
    gallery.append(["Tom", None])
    gallery.append(["Jerry", "q"])
    gallery.add_image(SheetImage(str(_picture(tmp_path / "tom.png", "blue"))), "B2")
    gallery.add_image(SheetImage(str(_picture(tmp_path / "jerry.png", "brown"))), "B3")
    path = tmp_path / "gallery.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(STRUCTURE, encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(path))
    try:
        frame = env.operators.read_table_as_dataframe("gallery", has_headers=True)
    finally:
        env.workbook.close()

    # An empty cell shows the picture; a cell with text keeps it and shows the picture after it.
    assert list(frame["photo"]) == ["[[media:1_B2]]", "q [[media:1_B3]]"]
