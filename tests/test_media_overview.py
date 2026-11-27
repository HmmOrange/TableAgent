"""Overview answers name a workbook's charts and pictures in words; the media catalog says what kind they are."""
from pathlib import Path

import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.drawing.image import Image as SheetImage
from PIL import Image

from TableAgent.QA.actions.common_info import CommonInfoSubtaskAction
from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.prompts.planner import PLANNER_SYSTEM_PROMPT
from TableAgent.QA.actions.write_plan import PLAN_CATEGORY_REVIEW_SYSTEM_PROMPT
from TableAgent.schema.subtask import SubTask

STRUCTURE = """table1:
  id: defects
  name: Defects
  description: Defect counts
  sheet: Defects
  headers:
  - id: code
    label: Code
    description: Defect code
    orientation: column
    header_range: A1
    data_range: A2:A4
    sub_headers: []
  - id: count
    label: Count
    description: Number of defects
    orientation: column
    header_range: B1
    data_range: B2:B4
    sub_headers: []
"""


def _env(tmp_path: Path) -> QAEnvironment:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Defects"
    sheet.append(["Code", "Count"])
    for code, count in (("D01", 412), ("D02", 268), ("D03", 173)):
        sheet.append([code, count])
    chart = BarChart()
    chart.title = "Pareto of defects"
    chart.add_data(Reference(sheet, min_col=2, min_row=1, max_row=4), titles_from_data=True)
    sheet.add_chart(chart, "E2")
    photos = workbook.create_sheet("Photos")
    for index in range(12):
        picture = tmp_path / f"p{index}.png"
        Image.new("RGB", (20, 10), (index * 20, 0, 0)).save(picture)
        photos.add_image(SheetImage(str(picture)), f"A{index * 3 + 1}")
    path = tmp_path / "book.xlsx"
    workbook.save(path)
    workbook.close()
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(STRUCTURE, encoding="utf-8")
    return QAEnvironment(str(structure_path), str(path))


def test_the_catalog_says_what_kind_each_sheet_holds(tmp_path):
    env = _env(tmp_path)
    try:
        catalog = env.operators.operator_catalog()
    finally:
        env.workbook.close()
    assert "- Defects: 1 chart" in catalog
    assert "- Photos: 12 pictures" in catalog
    assert "do not rebuild it from the table's data" in catalog


def test_an_overview_names_charts_and_counts_pictures_without_showing_them(tmp_path):
    env = _env(tmp_path)
    try:
        action = CommonInfoSubtaskAction(env)
        subtask = SubTask(
            id="inspect_workbook",
            description="Describe the workbook",
            layer="inspect",
            category="common_info",
            metadata={"common_info_scope": "workbook"},
        )
        overview = action._render(action._records("What is this file about?", subtask))
    finally:
        env.workbook.close()
    assert "- Charts, diagrams and pictures: 1 chart (Pareto of defects)" in overview
    assert "- Charts, diagrams and pictures: 12 pictures" in overview
    # Named in words only: an overview never shows the pictures themselves.
    assert "[[media:" not in overview


def test_requests_to_see_pictures_are_routed_to_inspection():
    for prompt in (PLANNER_SYSTEM_PROMPT, PLAN_CATEGORY_REVIEW_SYSTEM_PROMPT):
        assert "see, show or read pictures, charts or diagrams" in prompt


def test_charts_are_listed_with_their_titles_and_other_sheets_are_named(tmp_path):
    env = _env(tmp_path)
    try:
        listed = env.operators.list_media()
        catalog = env.operators.operator_catalog()
    finally:
        env.workbook.close()
    chart = listed[listed["kind"] == "chart"].iloc[0]
    assert chart["title"] == "Pareto of defects"
    assert list(listed.columns[:4]) == ["id", "cells", "kind", "title"]
    # Without a sheet, list_media covers the loaded sheets; the catalog says what it leaves out.
    assert "Photos" not in set(listed["sheet"])
    assert "`list_media()` without a sheet leaves them out" in catalog


def test_the_sheets_an_answer_reads_are_recorded(tmp_path):
    env = _env(tmp_path)
    try:
        env.operators.read_table_as_dataframe("defects", has_headers=True)
        env.operators.read_range("A1:B2", sheet="photos")
        used = list(env.used_sheets)
    finally:
        env.workbook.close()
    assert used == ["Defects", "Photos"]


def test_the_final_review_sees_what_each_picture_token_shows(tmp_path):
    from types import SimpleNamespace

    from TableAgent.QA.actions.review_final_answer import ReviewFinalAnswerAction

    env = _env(tmp_path)
    prompts = []

    class Client:
        def generate(self, prompt, system_prompt=None):
            prompts.append(prompt)
            return SimpleNamespace(content='{"accepted": true, "score": 1.0, "feedback": "ok"}')

    try:
        chart_id = env.operators.list_media()["id"].iloc[0]
        legend = env.operators.media_legend(f"Pareto: [[media:{chart_id}]] and [[media:9_Z9]]")
        review = ReviewFinalAnswerAction(env, llm_client=Client())
        review.run(question="Show the Pareto chart", plan=[], outputs=[], final_answer=f"Here: [[media:{chart_id}]]")
        review.run(question="How many defects?", plan=[], outputs=[], final_answer="853")
    finally:
        env.workbook.close()

    assert f'[[media:{chart_id}]] = chart "Pareto of defects" at Defects!' in legend
    assert "[[media:9_Z9]] = no such picture in the workbook" in legend
    assert "Pictures shown by the answer" in prompts[0]
    # Answers without pictures keep the review prompt they always had.
    assert "Pictures shown by the answer" not in prompts[1]
