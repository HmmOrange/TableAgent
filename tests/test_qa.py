"""Offline tests of the QA stage: workbook tools, checks, notebook, and the answer loop with a scripted model."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openpyxl
from openpyxl.styles import Font, PatternFill
import pytest

from TableAgent.src import orchestrator as orchestrator_module
from TableAgent.src.checker import reverse_facts, same_values
from TableAgent.src.inspector import shown_number
from TableAgent.src.orchestrator import answer_question
from TableAgent.src.ledger import Evidence, Need, Ledger
from TableAgent.src.prompts import (
    DEBATE_SYSTEM_PROMPT,
    REVERSE_PROMPT,
    INSPECTOR_SYSTEM_PROMPT,
    INSPECTOR_SYSTEM_PROMPT_WITH_LOOK,
    REASONER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    without_tools,
)
from TableAgent.src.synthesizer import part_text
from TableAgent.src.tools.formats import format_marks, legend_notes, legends
from TableAgent.src.tools.headers import read_layout
from TableAgent.src.tools.notebook import Notebook
from TableAgent.src.tools.sheet import SheetTools
from TableAgent.src.utils.llm import parse_json


@pytest.fixture
def workbook_files(tmp_path: Path) -> Path:
    """Two sections (North, South) under one header row; the label "Alpha" occurs in both."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet["A1"] = "Staff by region"
    sheet.merge_cells("A1:D1")
    for col, label in zip("BCD", ("Region", "2018", "2019")):
        sheet[f"{col}3"] = label
    sheet["A4"] = "North"
    rows = {5: ("Alpha", 10, 12), 6: ("Beta", 20, 25), 8: ("Alpha", 30, 35), 9: ("Gamma", 40, 45)}
    for row, values in rows.items():
        for col, value in zip("BCD", values):
            sheet[f"{col}{row}"] = value
    sheet["A7"] = "South"
    workbook_path = tmp_path / "staff.xlsx"
    workbook.save(workbook_path)
    return workbook_path


@pytest.fixture
def tools(workbook_files: Path) -> SheetTools:
    return SheetTools(openpyxl.load_workbook(workbook_files, data_only=True))


class ScriptedLLM:
    """Replies from per-role scripts; records every prompt (and each Checker system prompt) by role."""

    def __init__(
        self, reason: list[dict], inspect: list[dict] = (), synthesize: list[dict] = (), check: list[dict] = (),
        look: list[str] = (), start: list[dict] = (),
    ):
        # `reason` opens with the starting plan. reason_start asks for the plan again (a second plan, then the
        # reconciled one): `start` scripts those replies; without it the same plan comes back.
        self.scripts = {
            "reason": list(reason), "inspect": list(inspect), "synthesize": list(synthesize), "check": list(check),
            "look": list(look), "start": list(start),
        }
        self.prompts: dict[str, list[str]] = {role: [] for role in self.scripts}
        self.check_systems: list[str] = []
        self.inspect_systems: list[str] = []

    def generate(self, prompt: str, system_prompt: str | None = None):
        roles = {
            REASONER_SYSTEM_PROMPT: "reason", INSPECTOR_SYSTEM_PROMPT: "inspect", INSPECTOR_SYSTEM_PROMPT_WITH_LOOK: "inspect",
            SYNTHESIZER_SYSTEM_PROMPT: "synthesize",
        }
        role = "check" if system_prompt.startswith("You are a Checker") else roles.get(system_prompt, "inspect")
        if role == "check":
            self.check_systems.append(system_prompt)
        if role == "inspect":
            self.inspect_systems.append(system_prompt)
        if role == "reason" and self.prompts["reason"] and "Reason about the question before anything is read" in prompt:
            role = "start"  # the starting plan asked for again
        self.prompts[role].append(prompt)
        if role == "start" and not self.scripts["start"]:
            return SimpleNamespace(content=json.dumps(self.plan), prompt_tokens=10, completion_tokens=5)
        reply = self.scripts[role].pop(0)
        if role == "reason" and len(self.prompts["reason"]) == 1:
            self.plan = reply
        return SimpleNamespace(content=json.dumps(reply), prompt_tokens=10, completion_tokens=5)

    def generate_with_image(self, prompt: str, image_path: Path, system_prompt: str | None = None):
        assert Path(image_path).exists()
        self.prompts["look"].append(prompt)
        return SimpleNamespace(content=self.scripts["look"].pop(0), prompt_tokens=10, completion_tokens=5)


START = {
    "analysis": "One value is asked for.",
    "hypothesis": "The answer is the 2019 value of the item in its section.",
    "answer_form": "a number",
    "answer_slots": [{"slot": "the 2019 value", "asked_by": "value in 2019"}],
    "needs": [{"id": "n1", "description": "the 2019 value of Alpha in the South section"}],
}
INSPECT_N1 = {"assessment": "n1 is open.", "action": "inspect", "need_id": "n1", "strategy": "search_sheet",
              "instruction": "Alpha occurs twice; take the South one.", "reason": "need n1"}
SYNTHESIZE = {"assessment": "n1 is found.", "action": "synthesize", "reason": "done"}
ACCEPT = {"assessment": "Fits.", "verdict": "accept", "problem": "", "instruction": ""}
LOOK = {"thought": "Find the label.", "code": "print(sheet.find('Alpha'))", "done": False}
SOUTH_EVIDENCE = {
    "thought": "Take the South row.",
    "code": "evidence_n1 = {'value': 35, 'cells': ['D8'], 'where': 'South > Alpha | 2019', 'note': 'not the North row D5'}",
    "done": True,
}
PARTS = {"thought": "One slot.", "code": "answer_parts = [evidence_n1['value']]"}
REFUTED = {"verdict": "refuted", "reason": "the North row fits the need", "cells": ["D5"]}
CONFIRMED = {"verdict": "confirmed", "reason": "D8 is South's 2019 value", "cells": ["D8"]}


def _run(
    tmp_path: Path, workbook_files: Path, llm: ScriptedLLM,
    question: str = "What is Alpha's value in 2019 in the South?", **settings,
):
    return answer_question(llm, question, workbook_files, tmp_path / "runs", **settings)


def _ruled(ruling: str, action: dict) -> dict:
    return {**action, "rulings": [{"need_id": "n1", "ruling": ruling, "why": f"{ruling} it"}]}


# ---------------------------------------------------------------------------------------------- workbook tools

def test_reads_resolve_merged_cells_and_find_matches_whole_words(tools: SheetTools):
    assert tools.read("A1:B1") == [["Staff by region", "Staff by region"]]
    assert tools.read("'Data'!C5") == [[10]]
    assert [ref for ref, _ in tools.find("alpha")] == ["B5", "B8"]
    assert tools.find("alp") == []
    assert tools.frame().loc[8, "D"] == 35


def test_find_falls_back_to_plurals_split_headers_and_near_labels():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["A2"], sheet["B1"], sheet["C1"] = "Output per", "worker", "Unadjusted indexes", "Lower threshold (km)"
    tools = SheetTools(workbook)
    assert tools.find("output per worker") == [("A1", "Output per worker")]
    assert tools.find("unadjusted index") == [("B1", "Unadjusted indexes")]
    assert tools.find("lower threshold distance") == [("C1", "Lower threshold (km)")]
    assert tools.find("upper distance") == []


def test_reads_give_text_without_its_outer_spaces():
    workbook = openpyxl.Workbook()
    workbook.active["A1"], workbook.active["A2"], workbook.active["A3"] = "CA ", "  ", " New  York"
    tools = SheetTools(workbook)
    assert tools.read("A1:A3") == [["CA"], [None], ["New  York"]]
    column = tools.frame()["A"]
    assert column[1] == "CA" and column.isna()[2] and column[3] == "New  York"


def test_reading_past_the_sheet_and_walking_headers_leave_its_size_as_loaded():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Region", "2019", "2020"])
    sheet.append(["North", 1, 2])
    sheet.append(["South", 3, 4])
    tools = SheetTools(workbook)
    size, frame = tools.sheets(), tools.frame()

    assert tools.read("C3:E5") == [[4, None, None], [None, None, None], [None, None, None]]
    assert tools.headers("B3", look=False)["row_labels"] == ["South"] and "E9=" not in tools.around("D8")
    assert tools.cells("A1:Z50").shape[0] == 9 and len(tools.formats("A1:Z50").splitlines()) == 9
    assert tools.sheets() == size == ["Sheet (3 rows x 3 columns)"] and tools.frame().equals(frame)
    assert (sheet.max_row, sheet.max_column) == (3, 3)  # no cell was created past the loaded size


def test_headers_walk_to_the_row_label_column_header_and_section(tools: SheetTools):
    found = tools.headers("D8")
    assert found["row_labels"] == ["Alpha"]
    assert found["column_headers"] == ["Staff by region", "2019"]
    assert found["sections"] == [{"label": "South", "row": 7, "rows": "8-9"}]
    assert found["siblings"] == [{"label": "North", "row": 4, "rows": "5-6"}]
    assert found["uncertain"] == []
    assert tools.describe("D5").startswith("D5 = 12 | row: Alpha | column: Staff by region > 2019 | section: North (row 4)")
    assert tools.around("D8", rows=1, cols=1).splitlines() == [
        "row 7: (empty)", "row 8: C8=30 | *D8=35", "row 9: C9=40 | D9=45",
    ]


def test_headers_list_parent_sections_but_not_siblings_and_bold_is_the_higher_level():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for ref, value in {"A2": "Region A", "B3": "Urban", "B4": "x", "C4": 5, "B5": "Rural", "B6": "y", "C6": 7}.items():
        sheet[ref] = value
    assert [section["label"] for section in SheetTools(workbook).headers("C6")["sections"]] == ["Rural", "Region A"]
    same_column = openpyxl.Workbook()
    sheet = same_column.active
    for ref, value in {"A1": "All", "A2": "Group", "A3": "x", "B3": 5}.items():
        sheet[ref] = value
    sheet["A1"].font = Font(bold=True)
    assert [section["label"] for section in SheetTools(same_column).headers("B3")["sections"]] == ["Group", "All"]


def test_a_section_label_merged_across_the_row_is_still_a_section():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"], sheet["C1"], sheet["D1"] = "Item", "Code", "Unit", "Value"
    for row, label, item in ((2, "Cafes", ("Alpha", "a", "kg", 5)), (4, "Other income", ("Beta", "b", "kg", 9))):
        sheet[f"A{row}"] = label
        sheet.merge_cells(f"A{row}:C{row}")
        sheet[f"A{row + 1}"], sheet[f"B{row + 1}"], sheet[f"C{row + 1}"], sheet[f"D{row + 1}"] = item
    tools = SheetTools(workbook)
    assert "section: Other income (row 4)" in tools.describe("D5")
    assert tools.headers("D3")["sections"][0]["rows"] == "3-3"


def test_headers_carry_merged_keys_down_and_ignore_striping(tmp_path: Path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"], sheet["C1"] = "Location", "Item", "Qty"
    for row, (location, item, qty) in enumerate(
        [("North", "Alpha", 1), (None, "Beta", 2), (None, "Gamma", 3), ("South", "Delta", 4), (None, "Eps", 5)], start=2,
    ):
        sheet[f"A{row}"], sheet[f"B{row}"], sheet[f"C{row}"] = location, item, qty
        sheet[f"B{row}"].fill = PatternFill("solid", fgColor="FFDDEEFF" if row % 2 else "FFFFFFFF")
    sheet.merge_cells("A2:A4")
    sheet["B7"], sheet["C7"] = "Total", "=SUM(C2:C6)"
    path = tmp_path / "keys.xlsx"
    workbook.save(path)
    tools = SheetTools(openpyxl.load_workbook(path, data_only=True), formula_path=path)

    found = tools.headers("C3")
    assert found["block"] == {"rows": "2-6", "starts": "row 1 is the header", "ends": "row 7 has no data (label 'Total')"}
    assert {"header": "Location", "value": "North", "from": "A2", "rows": "2-4", "next": "South"}.items() <= found["keys"][0].items()
    assert "key: Location = North (from A2)" in tools.describe("C3")
    formula_sheet = openpyxl.load_workbook(path).active
    assert read_layout(formula_sheet, 1, (formula_sheet.max_row, formula_sheet.max_column), formula_sheet).totals == {7: (2, 6)}


def test_striping_in_part_of_a_sheet_does_not_split_blocks_but_a_bold_total_does():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"] = "Item", "Value"
    for row in range(2, 222):  # 20 rows striped for reading, then 200 plain rows
        sheet[f"A{row}"], sheet[f"B{row}"] = f"item {row}", row
        if row < 22 and row % 2:
            sheet[f"A{row}"].fill = PatternFill("solid", fgColor="FFDDEEFF")
    for row in (222, 223):
        sheet[f"A{row}"], sheet[f"B{row}"] = "Total" if row == 222 else "after", 1
    sheet["A222"].font = Font(bold=True)
    tools = SheetTools(workbook)

    assert tools.headers("B6")["block"]["rows"] == "2-221"
    assert tools.headers("B6")["block"]["ends"] == "row 222 is formatted differently (bold)"


def test_an_unsure_walk_asks_the_vision_model_about_the_area_a_few_times():
    workbook = openpyxl.Workbook()
    for row in range(1, 6):
        workbook.active[f"A{row}"] = row
    asked = []
    tools = SheetTools(workbook, look=lambda ref, question, sheet: asked.append(ref) or "a title row above")

    found = tools.headers("A5")
    assert found["uncertain"][0] == "no column header found above the cell"
    assert found["vision"] == "a title row above" and asked == ["A1:A5"]
    assert "vision" not in tools.headers("A4", look=False)
    for _ in range(3):
        tools.headers("A3")
    assert len(asked) == 3  # at most three images per question


def _stacked_sections_sheet() -> openpyxl.Workbook:
    """A title, a header row of years written as text, two stacked section rows merged across, then the data."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "Exposure statistics"
    sheet["A3"], sheet["C3"], sheet["D3"] = "Region", "2017", "2018"
    for row, label in ((4, "Measure: river flooding"), (5, "Unit: percent")):
        sheet[f"A{row}"] = label
        sheet.merge_cells(f"A{row}:D{row}")
    for row, (region, first, second) in enumerate([("North", 1.5, 2.5), ("South", 3.0, 4.0)], start=6):
        sheet[f"A{row}"], sheet[f"C{row}"], sheet[f"D{row}"] = region, first, second
    return workbook


def test_headers_pass_stacked_sections_to_the_years_written_as_text():
    found = SheetTools(_stacked_sections_sheet()).headers("D7", look=False)
    assert found["column_headers"][-1] == "2018"
    assert found["sections"][0] == {"label": "Measure: river flooding / Unit: percent", "row": 4, "rows": "6-7"}
    assert found["block"]["starts"] == "row 5 has no data (label 'Unit: percent')"


def _titled_table(header: list[Any], rows: list[list[Any]]) -> openpyxl.Workbook:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Report title"])
    sheet.append([])
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    return workbook


def test_headers_give_a_label_cell_of_the_body_its_column_header():
    workbook = _titled_table(["School", "State", "Loans"], [[f"School {i}", "AK" if i < 15 else "AL", i * 10] for i in range(30)])
    found = SheetTools(workbook).headers("B30", look=False)
    assert found["column_headers"] == ["State"] and found["row_labels"] == ["School 26"]
    assert found["block"]["starts"] == "row 3 is the header"


def test_headers_read_a_placeholder_in_the_data_as_a_number():
    workbook = _titled_table(["Region", "Cases", "Rate"], [[f"R{i}", "-" if i % 3 == 0 else i * 5, ".." if i % 3 == 0 else 0.5] for i in range(30)])
    found = SheetTools(workbook).headers("B31", look=False)
    assert found["column_headers"] == ["Cases"] and found["row_labels"] == ["R27"]
    assert found["block"] == {"rows": "4-33", "starts": "row 3 is the header", "ends": "end of sheet"}


def test_headers_take_repeated_text_for_data_where_no_number_is_near():
    workbook = _titled_table(["Name", "State", "Grade"], [[f"P{i}", "AK" if i < 15 else "AL", "B"] for i in range(30)])
    found = SheetTools(workbook).headers("B30", look=False)
    assert found["column_headers"] == ["State"] and found["block"]["starts"] == "row 3 is the header"
    texts = _titled_table(["Region", "2019", "2020"], [[f"R{i}", str(i * 3), str(i * 4)] for i in range(30)])
    assert SheetTools(texts).headers("C30", look=False)["column_headers"] == []  # numbers written as text are data here


def test_row_labels_keep_codes_written_as_text_and_the_label_columns():
    workbook = _titled_table(
        ["Grade", "Gender", "Number", "Reporting", "Percent"],
        [["3", "Male", 1436, "1-3", 58.6], [None, "Female", 1013, "1-3", 41.4]],
    )
    tools = SheetTools(workbook)
    assert tools.headers("C4", look=False)["row_labels"] == ["3", "Male"]
    assert tools.headers("E4", look=False)["row_labels"] == ["3", "Male", "1-3"]
    assert tools.headers("A4", look=False)["uncertain"] == []  # a label in the first column needs no row label


def test_headers_count_a_merged_header_once():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "Students by grade"
    sheet.merge_cells("A1:B1")
    sheet["A3"], sheet["B3"], sheet["B5"] = "Grade", "Total students", "Number"
    sheet.merge_cells("A3:A5")
    sheet.merge_cells("B3:B4")
    for row, (grade, count) in enumerate([("1", 10), ("2", 20), ("3", 30)], start=6):
        sheet[f"A{row}"], sheet[f"B{row}"] = grade, count
    assert SheetTools(workbook).headers("B7", look=False)["column_headers"] == ["Students by grade", "Total students", "Number"]


def test_the_vision_area_skips_empty_cells_and_never_cuts_a_merged_cell():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "A title over the table"
    sheet.merge_cells("A1:F1")
    sheet["A3"], sheet["B3"] = 5, 7
    asked = []
    tools = SheetTools(workbook, look=lambda ref, question, sheet: asked.append(ref) or "a title row above")
    assert "vision" not in tools.headers("B4")  # empty: nothing to ask about
    assert tools.headers("B3")["vision"] == "a title row above" and asked == ["A1:F3"]


def test_formats_list_the_format_and_formula_of_filled_cells(tmp_path: Path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"], sheet["A2"], sheet["B2"], sheet["A3"], sheet["B3"] = "Item", None, "x", 4, "Total", "=SUM(B2:B2)"
    sheet["A3"].font = Font(bold=True)
    sheet["A2"].font = Font(italic=True, color="FFFF0000")
    sheet["B2"].fill = PatternFill("solid", fgColor="FFFFFF00")
    path = tmp_path / "f.xlsx"
    workbook.save(path)
    tools = SheetTools(openpyxl.load_workbook(path), formula_path=path)
    assert tools.formats("A1:B3").splitlines() == [
        "A1 = Item | plain", "A2 = x | italic, font FFFF0000", "B2 = 4 | fill FFFFFF00", "A3 = Total | bold", "B3 = =SUM(B2:B2) | plain | =SUM(B2:B2)",
    ]


def test_headers_report_replicate_header_spans_and_colour_legends():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"], sheet["D1"] = "Dose", "Replicates", "Single"
    sheet.merge_cells("B1:C1")
    for row, values in enumerate([(1, 4, 5, 6), (2, 7, 8, 9)], start=2):
        for col, value in enumerate(values, start=1):
            sheet.cell(row, col, value)
    sheet["F2"] = "Treated"
    for ref in ("F2", "B2", "C3"):
        sheet[ref].fill = PatternFill("solid", fgColor="FFFF0000")
    tools = SheetTools(workbook)
    assert tools.headers("C2", look=False)["spans"] == ["Replicates: B1:C1 (2 columns)"]
    assert tools.headers("D2", look=False)["spans"] == []
    assert tools.headers("C3", look=False)["legend"] == "Treated (F2, fill FFFF0000)"
    assert "legend" not in tools.headers("D3", look=False)


def _legend_workbook() -> openpyxl.Workbook:
    """Two prices filled red, a legend label "On sale" with the same fill set apart from the table, a merged title."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Prices"
    sheet["A1"] = "Price list"
    sheet.merge_cells("A1:B1")
    for row, (item, price) in enumerate([("Item", "Price"), ("pen", 2), ("ink", 9), ("pad", 4)], start=2):
        sheet.cell(row, 1, item)
        sheet.cell(row, 2, price)
    sheet["A6"], sheet["B6"] = "Total", "=SUM(B3:B5)"
    sheet["A6"].font = Font(bold=True)
    sheet["D2"] = "On sale"
    for ref in ("D2", "B3", "B5"):
        sheet[ref].fill = PatternFill("solid", fgColor="FFFF0000")
    return workbook


def test_cells_gives_one_row_per_filled_cell_with_its_format_for_selecting_in_code(tmp_path: Path):
    path = tmp_path / "prices.xlsx"
    _legend_workbook().save(path)
    tools = SheetTools(openpyxl.load_workbook(path), formula_path=path)

    cells = tools.cells("A1:B6")
    assert list(cells.columns) == ["ref", "row", "column", "value", "range", "bold", "italic", "fill", "fill_colour", "font", "font_colour", "indent", "formula"]
    assert cells[cells.ref == "A1"]["range"].tolist() == ["A1:B1"] and "B1" not in cells.ref.tolist()  # a merged cell once
    assert cells[cells.fill_colour == "red"].ref.tolist() == ["B3", "B5"]
    assert cells[cells.bold].ref.tolist() == ["A6"] and cells[cells.ref == "B6"].formula.tolist() == ["=SUM(B3:B5)"]
    assert tools.cells("B:B").ref.tolist() == ["A1", "B2", "B3", "B4", "B5", "B6"]  # a whole column; the merged title reaches into it


def test_legends_and_format_marks_are_found_in_code_over_every_cell():
    workbook = _legend_workbook()
    found = legends(workbook.worksheets)
    assert found == [{"sheet": "Prices", "label": "On sale", "cell": "D2", "fill": "FFFF0000", "columns": {"B": 2}}]
    assert legend_notes(found) == "- sheet 'Prices': \"On sale\" (D2) has fill FFFF0000; number cells with this fill, by column: B (2)"
    marks = {(mark["kind"], mark["value"]): mark for mark in format_marks(workbook.worksheets)}
    red = marks["fill", "FFFF0000"]
    assert (red["colour"], red["numbers"], red["labels"], red["columns"], red["legend"]) == ("red", 2, 1, ["D", "B"], "On sale (D2)")
    assert marks["bold", True]["labels"] == 1

    workbook.active["C2"] = "note"  # a filled label inside a row of the table is a shaded cell, not a legend set apart
    assert legends(workbook.worksheets) == []


def test_formats_reach_the_model_through_code_not_through_the_preview(tmp_path):
    path = tmp_path / "prices.xlsx"
    _legend_workbook().save(path)
    start = {**START, "needs": [{"id": "n1", "description": "the prices filled with the On sale colour"}]}
    select = {
        "thought": "Select by fill.",
        "code": "found = sheet.cells('B:B'); on_sale = found[found.fill == legends['On sale']]\n"
                "print([mark['colour'] for mark in formats if mark['kind'] == 'fill'])\n"
                "evidence_n1 = {'value': on_sale.value.tolist(), 'cells': on_sale.ref.tolist(), 'where': 'Price | On sale', 'note': ''}",
        "done": False,
    }
    done = {"thought": "Set.", "code": "", "done": True}
    probe = {"thought": "Select too.", "code": "print(cells('B:B')[lambda found: found.fill == legends['On sale']].ref.tolist(), len(formats))", "done": False}
    llm = ScriptedLLM([start, INSPECT_N1, SYNTHESIZE, ACCEPT], [select, done], [PARTS], [probe, {**CONFIRMED, "done": True}])
    result = _run(tmp_path, path, llm, question="Which prices are on sale?", checks=("independent",), view="frame")

    assert "Fill-colour legends, found in code over every cell (the preview shows no colours)" in llm.prompts["reason"][0]
    assert '"On sale" (D2) has fill FFFF0000; number cells with this fill, by column: B (2)' in llm.prompts["reason"][0]
    assert "The notebook variable `legends` holds them" in llm.prompts["inspect"][0] and "The variable `formats` lists" in llm.prompts["inspect"][0]
    assert "['red']" in llm.prompts["inspect"][1]
    assert result.ledger.evidence[0].value == [2, 4] and result.ledger.evidence[0].cells == ["B3", "B5"]
    assert "['B3', 'B5'] 2" in llm.prompts["check"][1]


def test_without_legends_the_starting_prompt_names_none(tmp_path, workbook_files):
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS])
    _run(tmp_path, workbook_files, llm)
    assert "Fill-colour legends" not in llm.prompts["reason"][0] and "`legends`" not in llm.prompts["inspect"][0]
    assert "| A1:D1:Staff by region | E1: |" not in llm.prompts["reason"][0]
    assert "| A1:D1:Staff by region |" in llm.prompts["reason"][0]  # the preview writes a merged cell once, with its range


def test_a_row_of_plain_numbers_above_formatted_data_is_a_header_row():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"], sheet["C1"] = "Item", 1, 2
    for row, values in enumerate([("x", 10.5, 11.5), ("y", 20.5, 21.5)], start=2):
        for col, value in enumerate(values, start=1):
            sheet.cell(row, col, value)
            if col > 1:
                sheet.cell(row, col).number_format = "0.0"
    tools = SheetTools(workbook)
    assert tools.headers("C3", look=False)["column_headers"] == ["2"]

    sheet["B2"].number_format = sheet["C2"].number_format = "General"  # data typed like the numbers above: no header row
    assert SheetTools(workbook).headers("C3", look=False)["column_headers"] == []


def test_numbers_written_once_per_group_of_rows_are_keys_not_data():
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"], sheet["C1"] = "Code", "Item", "Count"
    row = 2
    for code in (101, 102, 103):
        for index, item in enumerate(("a", "b", "c")):
            sheet.cell(row, 1, code if index == 0 else None)
            sheet.cell(row, 2, item)
            sheet.cell(row, 3, row * 10)
            row += 1
    layout = read_layout(sheet, 1, (sheet.max_row, sheet.max_column))
    assert layout.data_columns == [3] and layout.key_columns == [1, 2]

def test_look_is_unavailable_without_a_vision_model(tools: SheetTools):
    with pytest.raises(ValueError, match="not available"):
        tools.look("B4:D8", "Which cells are shaded?")
    assert "vision" not in tools.headers("A2")


# ------------------------------------------------------------------------------------------------------ checks

def test_reverse_facts_give_the_meaning_of_the_cells_and_their_values(tools: SheetTools):
    ledger = Ledger("q")
    facts = reverse_facts(tools, ledger, Evidence("n1", 1, 1, "search_sheet", "", True, 35, ["D8"]))
    assert facts[1].strip().startswith("D8 = 35 | row: Alpha | column: Staff by region > 2019 | section: South (row 7)")
    assert "the cells hold the reported values" in facts

    facts = reverse_facts(tools, ledger, Evidence("n1", 1, 1, "search_sheet", "", True, [12, 99], ["D5", "D8"]))
    assert "values differ: D8 holds '35', reported '99'" in facts
    assert "2 filled cell(s) in the column within rows 5-6, 8-9 were not taken:" in facts
    assert any(line.strip().startswith("D6 = 25 | row: Beta") for line in facts)
    assert "4 value(s) reported but 2 cell(s) given" in " ".join(
        reverse_facts(tools, ledger, Evidence("n1", 1, 1, "s", "", True, [1, 2, 3, 4], ["D5", "D8"]))
    )


def test_reverse_facts_name_cells_shared_with_another_need(tools: SheetTools):
    ledger = Ledger("q")
    ledger.start({**START, "needs": [{"id": "n1", "description": "a"}, {"id": "n2", "description": "b"}]})
    ledger.record(Evidence("n1", 1, 1, "search_sheet", "", True, [12, 25], ["D5:D6"]))
    facts = reverse_facts(tools, ledger, Evidence("n2", 1, 2, "search_sheet", "", True, 25, ["D6"]))
    assert "shared with the evidence of n1: D6 (combining both counts them twice)" in facts


def test_values_are_the_same_in_any_order_and_form():
    assert same_values({"Alpha": 12, "Beta": "25"}, [25.0, 12])
    assert not same_values([12, 25], [12])
    assert not same_values("North", "South")


# ---------------------------------------------------------------------------------------------------- notebook

def test_notebook_keeps_variables_and_refuses_system_modules():
    notebook = Notebook({})
    assert notebook.run("x = 2\nprint(x * 3)", "a").output == "6"
    assert notebook.run("print(x + 1)", "b").output == "3"
    assert "not allowed" in notebook.run("import os", "c").error
    failed = notebook.run("y = 1\nz = y / 0", "d")
    assert failed.error.splitlines() == ["line 2: z = y / 0", "ZeroDivisionError: division by zero"]


def test_notebooks_running_in_threads_keep_their_own_output():
    notebooks = [Notebook({}) for _ in range(8)]
    code = "import time\nfor i in range(5):\n    print(name, end='')\n    time.sleep(0.02)"
    for index, notebook in enumerate(notebooks):
        notebook.namespace["name"] = str(index)
    with ThreadPoolExecutor(8) as pool:
        cells = list(pool.map(lambda notebook: notebook.run(code, "a"), notebooks))
    assert [cell.output for cell in cells] == [str(index) * 5 for index in range(8)]


def test_notebook_allows_strptime_but_not_its_module():
    notebook = Notebook({})
    assert notebook.run("import datetime\nprint(datetime.datetime.strptime('2020-01-31', '%Y-%m-%d').day)", "a").output == "31"
    assert "not allowed" in notebook.run("import _strptime", "b").error


# ------------------------------------------------------------------------------------------------- answer loop

def test_answers_through_reasoning_inspection_synthesis_and_end(tmp_path, workbook_files):
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [LOOK, SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm)

    assert result.success and result.answer == "35"
    assert "[('B5', 'Alpha'), ('B8', 'Alpha')]" in llm.prompts["inspect"][1]  # turn 2 sees what turn 1 printed
    assert "where: South > Alpha | 2019" in llm.prompts["reason"][2]
    assert result.ledger.evidence[0].checks == []  # no checks run by default
    trace = Path(result.artifacts["trace_md"]).read_text(encoding="utf-8")
    assert "#### Inspection n1, attempt 1 (search_sheet): found" in trace
    assert "### Synthesis → 35" in trace and "### Abstract reasoning: end (accept)" in trace
    ledger = json.loads(Path(result.artifacts["ledger_json"]).read_text(encoding="utf-8"))
    assert ledger["needs"]["n1"]["status"] == "found"
    assert sorted(path.name for path in Path(result.artifacts["cells_dir"]).iterdir())[0] == "01_n1_attempt1_turn1.py"
    events = [json.loads(line)["event_type"] for line in Path(result.artifacts["events_jsonl"]).read_text().splitlines()]
    assert events[0] == "question" and events[-1] == "result"
    readable = Path(result.artifacts["events_log"]).read_text(encoding="utf-8")
    assert "llm_call #1 ok" in readable and "prompt: (" in readable
    assert "inspection_turn\n  need_id: n1\n  attempt: 1\n  number: 1\n  thought: Find the label." in readable


def test_synthesis_is_refused_while_a_need_is_open(tmp_path, workbook_files):
    llm = ScriptedLLM([START, SYNTHESIZE, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm)

    assert result.success
    assert "Cannot synthesize while needs are open: n1" in llm.prompts["reason"][2]
    assert "refused the step before" not in llm.prompts["reason"][2]
    assert result.ledger.decisions[0].accepted is False


def test_an_action_refused_twice_for_the_same_reason_is_called_out(tmp_path, workbook_files):
    llm = ScriptedLLM([START, SYNTHESIZE, SYNTHESIZE, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm)

    assert result.success
    assert "The same action was refused the step before for the same reason" in llm.prompts["reason"][3]


def test_the_starting_plan_is_written_twice_and_reconciled(tmp_path, workbook_files):
    other = {**START, "hypothesis": "The answer is the North value.", "needs": [{"id": "n1", "description": "the 2019 value of Alpha in the North section"}]}
    llm = ScriptedLLM([other, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS], start=[START, START])
    result = _run(tmp_path, workbook_files, llm)

    first, second, reconciling = llm.prompts["reason"][0], *llm.prompts["start"]
    assert first == second  # the two plans are written independently, from the same prompt
    assert "Plan A:\nhypothesis: The answer is the North value." in reconciling
    assert "Plan B:\nhypothesis: The answer is the 2019 value of the item in its section." in reconciling
    assert "  - n1: the 2019 value of Alpha in the South section" in reconciling
    assert result.ledger.needs["n1"].description == "the 2019 value of Alpha in the South section"  # the reconciled plan stands
    readable = Path(result.artifacts["events_log"]).read_text(encoding="utf-8")
    named = [line.strip() for line in readable.splitlines() if line.startswith("  plan: ")]
    assert named == ["plan: Plan A", "plan: Plan B", "plan: reconciled from Plan A and Plan B"]  # the log names each plan
    assert result.answer == "35"


def test_a_starting_plan_with_many_needs_is_sent_back_once_to_merge_them(tmp_path, workbook_files):
    split = {**START, "needs": [{"id": f"n{i}", "description": f"the {year} value of Alpha in the South"} for i, year in enumerate(range(2016, 2020), 1)]}
    llm = ScriptedLLM([split, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS], start=[split, split, START])
    result = _run(tmp_path, workbook_files, llm)

    assert "Your previous reply listed 4 needs" in llm.prompts["start"][2]
    assert "  plan: needs merged (the plan before listed 4)" in Path(result.artifacts["events_log"]).read_text(encoding="utf-8")
    assert list(result.ledger.needs) == ["n1"] and result.answer == "35"


def test_an_error_turn_is_an_observation_and_done_needs_the_evidence(tmp_path, workbook_files):
    broken = {"thought": "Try.", "code": "print(undefined_name)", "done": False}
    early = {"thought": "Done?", "code": "print('ready')", "done": True}
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [broken, early, SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm)

    assert "NameError: name 'undefined_name' is not defined" in llm.prompts["inspect"][1]
    assert "done was refused: `evidence_n1` is not set to a dict." in llm.prompts["inspect"][2]
    assert result.answer == "35"


def test_the_last_inspection_turn_asks_for_the_evidence(tmp_path, workbook_files):
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [LOOK] * 11 + [SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm)

    assert "This is the last turn" in llm.prompts["inspect"][11] and "last turn" not in llm.prompts["inspect"][10]
    assert "Most needs take 3 to 6 turns" in llm.prompts["inspect"][0]
    assert result.answer == "35"


def test_turns_used_up_get_one_report_turn_and_a_failed_attempt_hands_on_its_thoughts(tmp_path, workbook_files):
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [LOOK] * 12 + [SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm)
    assert "Report turn: look nothing up" in llm.prompts["inspect"][12]  # the 13th turn only reports
    assert result.ledger.evidence[0].found and result.answer == "35"

    llm = ScriptedLLM([START, INSPECT_N1, INSPECT_N1, SYNTHESIZE, ACCEPT], [LOOK] * 13 + [SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm)
    assert not result.ledger.evidence[0].found and result.ledger.evidence[1].found
    assert "it ended thinking: Find the label." in llm.prompts["inspect"][13]  # the second attempt's first turn


def test_a_number_read_from_one_cell_is_reported_as_its_format_shows_it(tmp_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["B1"], sheet["A2"], sheet["B2"] = "Item", "Share", "x", 35.44
    sheet["B2"].number_format = "0.0"
    path = tmp_path / "shown.xlsx"
    workbook.save(path)
    start = {**START, "needs": [{"id": "n1", "description": "the share of x"}]}
    evidence = {"thought": "Read it.", "code": "evidence_n1 = {'value': sheet.read('B2')[0][0], 'cells': ['B2'], 'where': 'x | Share', 'note': ''}", "done": True}
    llm = ScriptedLLM([start, INSPECT_N1, SYNTHESIZE, ACCEPT], [evidence], [PARTS])
    result = _run(tmp_path, path, llm, question="What is the share of x?")

    assert result.ledger.evidence[0].value == 35.4
    assert "Shown as 35.4 by the cell's number format 0.0 (stored 35.44)" in result.ledger.evidence[0].note
    assert shown_number(0.1234, "0.0%") == 0.123 and shown_number(1234.5, "#,##0") == 1234 and shown_number(3.14159, "General") is None


def test_synthesis_retries_until_there_is_one_part_per_slot(tmp_path, workbook_files):
    two = {"thought": "Two.", "code": "answer_parts = [35, 'extra']"}
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [two, PARTS])
    result = _run(tmp_path, workbook_files, llm)

    assert "has 2 entries; it needs exactly 1" in llm.prompts["synthesize"][1]
    assert result.answer == "35"


def test_answer_parts_are_written_plainly():
    assert part_text(True) == "Yes" and part_text(16.0) == "16" and part_text(2.5) == "2.5"
    assert part_text(["A", 3.0]) == "A, 3"
    assert part_text(10.15 + 10.15) == "20.3" and part_text(0.30000000000000004) == "0.3"
    assert part_text(82.15370000000075) == "82.1537" and part_text(2633.66201433206) == "2633.662"


def test_a_returned_review_reopens_exploration_once(tmp_path, workbook_files):
    returned = {"assessment": "Wrong section.", "verdict": "return", "problem": "North value", "instruction": "use South"}
    revise = {"assessment": "Revise.", "action": "revise_need", "need_id": "n1",
              "description": "the 2019 value of Alpha under the South section", "reason": "review"}
    llm = ScriptedLLM(
        [START, INSPECT_N1, SYNTHESIZE, returned, revise, INSPECT_N1, SYNTHESIZE],
        [SOUTH_EVIDENCE, SOUTH_EVIDENCE], [PARTS, PARTS],
    )
    result = _run(tmp_path, workbook_files, llm)

    assert result.success and result.ledger.returns == 1
    assert "The review returned the answer '35'" in llm.prompts["reason"][4]
    assert result.ledger.needs["n1"].inspections_left == 2  # revising restarts the inspection count
    assert len(result.ledger.answers) == 2  # the second answer is final without another review


def test_a_review_that_finds_a_missing_slot_adds_it(tmp_path, workbook_files):
    slots = [{"slot": "the 2019 value", "asked_by": "value in 2019"}, {"slot": "the section", "asked_by": "which section"}]
    returned = {"assessment": "The section is not reported.", "verdict": "return", "problem": "no section",
                "instruction": "report the section", "answer_slots": slots}
    two_parts = {"thought": "Two slots.", "code": "answer_parts = [evidence_n1['value'], 'South']"}
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, returned, SYNTHESIZE], [SOUTH_EVIDENCE], [PARTS, two_parts])
    result = _run(tmp_path, workbook_files, llm)

    assert result.success and result.ledger.answer_slots == slots
    assert "The answer slots were revised" in llm.prompts["reason"][4]
    assert result.ledger.answers[-1]["answer"] == "35, South"


def test_running_out_of_steps_with_an_open_need_fails(tmp_path, workbook_files, monkeypatch):
    monkeypatch.setattr(orchestrator_module, "MAX_STEPS", 2)
    llm = ScriptedLLM([START, SYNTHESIZE, SYNTHESIZE])
    result = _run(tmp_path, workbook_files, llm)

    assert not result.success and result.error == "Out of budget with open needs: n1."
    assert Path(result.artifacts["trace_md"]).is_file()


def test_look_renders_the_range_for_the_vision_model(tmp_path, workbook_files):
    class Renderer:
        def render_workbook_range(self, workbook_path, sheet, area, image_path):
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"png")

    look = {"thought": "Look at the marks.", "code": "print(sheet.look('B4:D8', 'Which cells are shaded?'))", "done": False}
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [look, SOUTH_EVIDENCE], [PARTS], look=["D8 is shaded."])
    result = _run(tmp_path, workbook_files, llm, vision_client=llm, renderer=Renderer())

    assert result.success
    assert 'range B4:D8 of sheet "Data"' in llm.prompts["look"][0]
    assert "D8 is shaded." in llm.prompts["inspect"][1]


# ------------------------------------------------------------------------------------------------------ ledger

def test_ledger_rules():
    ledger = Ledger("q")
    ledger.start({**START, "needs": [{"id": "n1", "description": "a"}, {"id": "n2", "description": "b"}]})
    assert ledger.refusal("inspect", {"need_id": "n1", "strategy": "guess"}).startswith("Unknown strategy")
    assert ledger.refusal("drop_need", {"need_id": "n1"}) == ""
    ledger.apply("drop_need", {"need_id": "n1"}, 1, "not needed")
    assert ledger.refusal("drop_need", {"need_id": "n2"}).startswith("Cannot drop n2, the last need")
    for attempt in (1, 2, 3):
        ledger.record(Evidence("n2", attempt, attempt, "search_sheet", ""))
    assert "inspected 3 times" in ledger.refusal("inspect", {"need_id": "n2", "strategy": "window"})
    ledger.apply("revise_need", {"need_id": "n2", "description": "b, in rows 2-9"}, 4, "narrow it")
    assert "inspect it before revising it again" in ledger.refusal("revise_need", {"need_id": "n2", "description": "c"})
    assert ledger.refusal("synthesize", {}).startswith("Cannot synthesize while needs are open: n2")
    assert ledger.refusal("revise_hypothesis", {"hypothesis": "h", "answer_slots": [{"slot": "x"}]}).startswith(
        "each answer slot"
    )
    assert ledger.refusal("jump", {}).startswith("Unknown action 'jump'")
    assert Need("n9", "x").variable == "evidence_n9"


def test_parse_json_takes_code_from_python_block_and_raw_line_breaks():
    assert parse_json('{"thought": "two\nlines", "done": false}\n```python\nprint(sheet.find("Alpha"))\n```') == {
        "thought": "two\nlines", "done": False, "code": 'print(sheet.find("Alpha"))',
    }


def test_parse_json_repairs_unescaped_quotes_inside_strings():
    reply = '{"thought": "the "Alpha" row, then sheet.read("B3:D9")", "done": false}'
    assert parse_json(reply) == {"thought": 'the "Alpha" row, then sheet.read("B3:D9")', "done": False}


def test_parse_json_repairs_quoted_lists_and_bad_escapes_inside_strings():
    reply = '{"note": "it read ["B3", "B4"] as \\\'Alpha\\\' \\T", "cells": ["B3", "B4"], "done": true}'
    assert parse_json(reply) == {"note": "it read [\"B3\", \"B4\"] as 'Alpha' \\T", "cells": ["B3", "B4"], "done": True}


def test_parse_json_takes_code_from_a_python_block_written_with_escaped_line_breaks():
    reply = '{"thought": "read", "done": false}\\n```python\\nrows = sheet.read(\\"A1:B2\\")\\nprint(rows)\\n```'
    assert parse_json(reply) == {"thought": "read", "done": False, "code": 'rows = sheet.read("A1:B2")\nprint(rows)'}


# ----------------------------------------------------------------------------------------------------- checkers

def test_a_refuting_check_must_be_ruled_on_before_synthesis_then_shows_in_one_line(tmp_path, workbook_files):
    llm = ScriptedLLM(
        [START, INSPECT_N1, SYNTHESIZE, _ruled("keep", SYNTHESIZE), ACCEPT], [SOUTH_EVIDENCE], [PARTS], [REFUTED],
    )
    result = _run(tmp_path, workbook_files, llm, checks=("reverse",))

    assert "D8 = 35 | row: Alpha | column: Staff by region > 2019 | section: South" in llm.prompts["check"][0]
    step = llm.prompts["reason"][2]
    assert "check reverse: REFUTED - the North row fits the need (cells: D5)" in step and "RULE ON IT" in step
    assert "Checks to rule on: n1." in step
    assert "Cannot synthesize before ruling on the checks of: n1." in llm.prompts["reason"][3]
    assert "check reverse: refuted - the North row fits the need; ruled keep: keep it" in llm.prompts["reason"][4]
    assert result.success and result.answer == "35"
    trace = Path(result.artifacts["trace_md"]).read_text(encoding="utf-8")
    assert "##### Check reverse: refuted" in trace and "**Ruling on n1:** keep" in trace


def test_rejecting_the_evidence_reopens_the_need(tmp_path, workbook_files):
    reinspect = _ruled("reject", INSPECT_N1)
    llm = ScriptedLLM(
        [START, INSPECT_N1, reinspect, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE, SOUTH_EVIDENCE], [PARTS], [REFUTED, CONFIRMED],
    )
    result = _run(tmp_path, workbook_files, llm, checks=("reverse",))

    assert result.ledger.evidence[0].ruling["ruling"] == "reject" and len(result.ledger.evidence) == 2
    assert "rejected: reject it" in llm.prompts["reason"][3]
    assert "check reverse: confirmed - D8 is South's 2019 value" in llm.prompts["reason"][3]
    assert result.answer == "35"


def test_a_later_check_runs_only_when_the_earlier_did_not_confirm(tmp_path, workbook_files):
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS], [CONFIRMED])
    result = _run(tmp_path, workbook_files, llm, checks=("reverse", "debate"))

    assert [check["by"] for check in result.ledger.evidence[0].checks] == ["reverse"]


def test_debate_objections_must_name_cells(tmp_path, workbook_files):
    probe = {"thought": "Other Alpha rows.", "code": "print(sheet.describe('D5'))", "done": False}
    objections = {"thought": "Two doubts.", "done": True, "objections": [
        {"objection": "the North Alpha row also fits", "cells": ["D5"]}, {"objection": "a feeling", "cells": []},
    ]}
    llm = ScriptedLLM(
        [START, INSPECT_N1, _ruled("keep", SYNTHESIZE), ACCEPT], [SOUTH_EVIDENCE], [PARTS], [probe, objections],
    )
    result = _run(tmp_path, workbook_files, llm, checks=("debate",))

    step = llm.prompts["reason"][2]
    assert "check debate: REFUTED - 1 objection(s) (cells: D5)" in step
    assert "- the North Alpha row also fits (D5 = 12)" in step and "a feeling" not in step
    assert "D5 = 12 | row: Alpha" in llm.prompts["check"][1]  # the debate has its own notebook with `claim`
    assert "n1_attempt1_debate_turn1.py" in Path(result.artifacts["trace_md"]).read_text(encoding="utf-8")


def test_a_blind_independent_check_reports_its_own_value_and_the_runner_compares(tmp_path, workbook_files):
    own = {"thought": "North row.", "done": True, "code": "found = {'value': 12, 'cells': ['D5']}", "reason": "row 5"}
    llm = ScriptedLLM([START, INSPECT_N1, _ruled("keep", SYNTHESIZE), ACCEPT], [SOUTH_EVIDENCE], [PARTS], [own])
    result = _run(tmp_path, workbook_files, llm, checks=("independent",), sees_value=False)

    check = result.ledger.evidence[0].checks[0]
    assert check["verdict"] == "refuted" and check["reason"] == "the Checker read 12 from D5 (row 5)"
    assert "`claim`" not in llm.check_systems[0] and "reads raw cells only" in llm.check_systems[0]
    assert "Inspector's evidence" not in llm.prompts["check"][0]
    assert "Workbook preview" in llm.prompts["check"][0] and "Alpha" in llm.prompts["check"][0]


def test_an_independent_check_sees_only_its_view_and_its_last_turn_gives_the_verdict(tmp_path, workbook_files):
    probe = {"thought": "Which tools?", "code": "print(hasattr(sheet, 'headers'), claim['value'], sheet.formats('A7:D8'))", "done": False}
    verdict_with_code = {**CONFIRMED, "thought": "It holds.", "done": False, "code": "print('not run')"}
    llm = ScriptedLLM(
        [START, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS], [probe] * 3 + [verdict_with_code],
    )
    result = _run(tmp_path, workbook_files, llm, checks=("independent",))

    assert "False 35 A7 = South | plain\nB8 = Alpha | plain" in llm.prompts["check"][1]
    assert "Workbook preview" in llm.prompts["check"][0] and "Workbook preview" not in llm.prompts["check"][1]  # first turn only
    assert "This is the last turn" in llm.prompts["check"][3]
    assert result.ledger.evidence[0].checks[0]["verdict"] == "confirmed"
    assert "check independent: confirmed - D8 is South's 2019 value" in llm.prompts["reason"][2]


def test_the_method_check_reaches_the_final_review(tmp_path, workbook_files):
    flawed = {"verdict": "flawed", "problem": "the question asks for a list, the code sums"}
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS], [flawed])
    result = _run(tmp_path, workbook_files, llm, synthesis_check=True)

    assert "answer_parts = [evidence_n1['value']]" in llm.prompts["check"][0]
    assert "found it flawed: the question asks for a list, the code sums" in llm.prompts["reason"][-1]
    assert "**Method check:** flawed" in Path(result.artifacts["trace_md"]).read_text(encoding="utf-8")


def test_prompts_name_no_structure():
    assert "structure" not in INSPECTOR_SYSTEM_PROMPT
    assert "structure" not in INSPECTOR_SYSTEM_PROMPT_WITH_LOOK
    assert "structure" not in DEBATE_SYSTEM_PROMPT
    assert "structure.yaml" not in REVERSE_PROMPT


def test_ledger_rejects_the_structure_strategy():
    ledger = Ledger("q")
    ledger.needs["n1"] = Need("n1", "a need")
    assert "Unknown strategy" in ledger.refusal("inspect", {"need_id": "n1", "strategy": "structure"})
    assert ledger.refusal("inspect", {"need_id": "n1", "strategy": "window"}) == ""


def test_evidence_with_date_keys_is_written_as_json():
    from datetime import date

    from TableAgent.src.checker import _json

    assert json.loads(_json({date(2024, 1, 31): [1, {date(2024, 2, 29): 2}]})) == {"2024-01-31": [1, {"2024-02-29": 2}]}


def test_a_reply_cut_off_at_the_length_limit_does_not_use_up_the_retry():
    from TableAgent.src.utils.llm import CountingLLM, ReplyError, ask_json, require_dict
    from TableAgent.src.utils.trace import RunLog

    class Replies:
        def __init__(self, *replies: tuple[str, bool]):
            self.replies, self.prompts = list(replies), []

        def generate(self, prompt: str, system_prompt: str | None = None):
            self.prompts.append(prompt)
            content, capped = self.replies.pop(0)
            return SimpleNamespace(content=content, prompt_tokens=1, completion_tokens=1, token_capped=capped)

    def ask(*replies: tuple[str, bool]):
        client = Replies(*replies)
        try:
            return ask_json(CountingLLM(client, RunLog()), "kind", "system", "PROMPT", require_dict), client.prompts
        except ReplyError as exc:
            return str(exc), client.prompts

    cut, bad, good = ('{"thought": "and so', True), ("not json", False), ('{"a": 1}', False)
    # A cut-off reply, then an ordinary bad one, still leaves the retry; each retry quotes the latest problem only.
    reply, prompts = ask(cut, bad, good)
    assert reply == {"a": 1}
    assert [prompt.count("Your previous reply") for prompt in prompts] == [0, 1, 1]
    assert "cut off at the length limit" in prompts[1] and "was rejected" in prompts[2]
    assert ask(bad, cut, good)[0] == {"a": 1}
    # Only one cut-off reply is free, and without one the ask still ends after two replies.
    assert ask(cut, cut, good)[0] == {"a": 1}
    assert ask(cut, cut, cut, good)[0] == "kind reply was invalid 3 times"
    assert ask(bad, bad, good)[0] == "kind reply was invalid twice"


# ---------------------------------------------------------------------------------------------- ablation settings
def test_settings_default_to_the_current_limits_and_check_the_ablation_values():
    from TableAgent import QASettings
    from TableAgent.src.inspector import MAX_TURNS
    from TableAgent.src.ledger import MAX_RETURNS

    defaults = QASettings.from_config({})
    assert (defaults.inspector_max_turns, defaults.max_returns) == (MAX_TURNS, MAX_RETURNS)
    assert (defaults.disabled_tools, defaults.save_artifacts, defaults.start_view) == ((), True, "text")
    assert QASettings.from_config({"start_view": "none"}).start_view == "none"
    chosen = QASettings.from_config({"disabled_tools": "find, headers, describe", "max_returns": 0, "inspector_max_turns": 4})
    assert (chosen.disabled_tools, chosen.max_returns, chosen.inspector_max_turns) == (("find", "headers", "describe"), 0, 4)
    for bad in ({"disabled_tools": ["headers"]}, {"disabled_tools": ["read"]}, {"max_returns": -1}, {"inspector_max_turns": 0}, {"start_view": "image"}):
        with pytest.raises(ValueError):
            QASettings.from_config(bad)


def test_the_settings_reach_the_exploration(monkeypatch):
    from TableAgent import QASettings
    from TableAgent.src import workbook_qa

    seen = {}

    def answer(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(workbook_qa, "answer_question", answer)
    settings = QASettings(inspector_max_turns=8, max_returns=2, disabled_tools=("find",), save_artifacts=False, start_view="none")
    with pytest.raises(RuntimeError):
        workbook_qa.WorkbookQA(SimpleNamespace(), settings).run(
            question="q", workbook_path=Path("w.xlsx"), qa_artifact_dir=Path("runs"), fallback_prompt="",
        )
    keys = ("inspector_max_turns", "max_returns", "disabled_tools", "save_artifacts", "start_view")
    assert {key: seen[key] for key in keys} == {
        "inspector_max_turns": 8, "max_returns": 2, "disabled_tools": ("find",), "save_artifacts": False, "start_view": "none",
    }


def test_tools_switched_off_leave_the_inspector_prompts_and_its_notebook(tmp_path, workbook_files):
    assert without_tools(INSPECTOR_SYSTEM_PROMPT, ()) == INSPECTOR_SYSTEM_PROMPT
    tried = {"thought": "Read it.", "code": "print(sheet.read('B5:B8'))\nsheet.find('Alpha')", "done": False}
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [tried, SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm, disabled_tools=("find", "headers", "describe"))

    assert result.answer == "35"
    for name in ("find", "headers", "describe"):
        assert f"sheet.{name}" not in llm.inspect_systems[0] and f"sheet.{name}" not in llm.prompts["inspect"][0]
    assert "sheet.read" in llm.inspect_systems[0]
    assert "Alpha" in llm.prompts["inspect"][1] and "sheet.find is not available" in llm.prompts["inspect"][1]


def test_inspector_turns_review_returns_and_artifacts_follow_the_settings(tmp_path, workbook_files):
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [LOOK] * 3 + [SOUTH_EVIDENCE], [PARTS])
    result = _run(tmp_path, workbook_files, llm, inspector_max_turns=4)
    assert "Turn 4 of 4. This is the last turn" in llm.prompts["inspect"][3] and result.answer == "35"
    assert "Turn 1 of 4. As soon as you have seen where the value is" in llm.prompts["inspect"][0]
    assert "Most needs take" not in llm.prompts["inspect"][0]

    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE], [SOUTH_EVIDENCE], [PARTS])  # no review reply is scripted
    result = _run(tmp_path / "no_review", workbook_files, llm, max_returns=0, save_artifacts=False)
    assert result.answer == "35" and not any(step.kind == "end" for step in result.ledger.steps)
    assert result.artifacts == {} and not (tmp_path / "no_review").exists()


def test_the_starting_plans_see_the_preview_or_only_the_sheet_sizes(tmp_path, workbook_files):
    llm = ScriptedLLM([START, INSPECT_N1, SYNTHESIZE, ACCEPT], [SOUTH_EVIDENCE], [PARTS], [CONFIRMED])
    result = _run(tmp_path, workbook_files, llm, start_view="none", checks=("independent",), view="frame")
    first = llm.prompts["reason"][0]
    assert 'Workbook: "Data" (9 rows x 4 columns)\nNo preview of its cells is given.' in first
    assert "Staff by region" not in first and "Workbook preview" not in first
    assert all("Staff by region" not in prompt for prompt in llm.prompts["start"])  # Plan B and the reconciled plan
    assert "| A1:D1:Staff by region |" in llm.prompts["check"][0]  # the Checker keeps the preview
    assert result.answer == "35"
