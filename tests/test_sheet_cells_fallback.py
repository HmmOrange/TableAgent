from pathlib import Path

import openpyxl
import yaml

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.sheet_cells import (
    sheet_cells_fallback_prompt,
    sheet_cells_text,
    structure_sheet_names,
)


def _form_workbook(path: Path) -> Path:
    """A form-like sheet: title block, a table, and a revision table below."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "HP layer (VN) "  # trailing space, as in the real file
    ws["A1"] = "Tiêu chuẩn kiểm tra công đoạn H/P Layer"
    ws["A3"] = "Số văn bản"
    ws["D3"] = "VSIIS-V0104I-002"
    ws["A6"] = "Phân loại"
    ws["B6"] = "Hạng mục kiểm tra"
    ws["O6"] = "Tiêu chuẩn kiểm tra"
    ws["B8"] = "Nhăn 구김"
    ws["O8"] = 'Nhăn tạo thành hình chữ "V" là NG'
    ws["B24"] = "Lệch lớp xếp적층치우침"
    ws["O24"] = "Lớp trong:\nCoupon 3 chấm lệch quá 1/3 hole là NG"
    ws.merge_cells("B24:D31")
    ws["A40"] = "Rev"
    ws["A41"] = 1
    wb.create_sheet("Other")["A1"] = "not selected"
    wb.save(path)
    return path


def _structure(path: Path, sheet: str) -> Path:
    path.write_text(
        yaml.safe_dump({"table1": {"id": "t", "name": "n", "sheet": sheet, "headers": []}}, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def test_cells_of_the_structure_sheet_are_read_as_rows(tmp_path: Path) -> None:
    workbook = _form_workbook(tmp_path / "book.xlsx")

    text = sheet_cells_text(workbook, ["HP layer (VN)"], "lệch lớp")

    assert text.startswith("## Sheet: HP layer (VN) ")
    assert "R24: B24=Lệch lớp xếp적층치우침 | O24=Lớp trong: / Coupon 3 chấm lệch quá 1/3 hole là NG" in text
    # A merged cell appears once, in its top-left cell; other sheets are left out.
    assert "B25=" not in text and "not selected" not in text


def test_a_cut_keeps_the_leading_rows_and_the_rows_of_the_question(tmp_path: Path) -> None:
    workbook = _form_workbook(tmp_path / "book.xlsx")
    wb = openpyxl.load_workbook(workbook)
    ws = wb["HP layer (VN) "]
    for row in range(50, 400):
        ws[f"A{row}"] = f"filler row {row} " + "x" * 40
    wb.save(workbook)

    text = sheet_cells_text(
        workbook, ["HP layer (VN)"], "Tiêu chuẩn lệch lớp xếp?\n\n[Answer language] Vietnamese", max_chars=1500
    )

    assert len(text) <= 1600
    assert "R1: A1=Tiêu chuẩn kiểm tra công đoạn H/P Layer" in text
    assert "R24: B24=Lệch lớp xếp적층치우침" in text
    assert "..." in text


def test_prompt_needs_a_sheet_the_workbook_has(tmp_path: Path) -> None:
    workbook = _form_workbook(tmp_path / "book.xlsx")

    assert sheet_cells_fallback_prompt("q", workbook, [_structure(tmp_path / "s.yaml", "Missing")]) is None
    prompt = sheet_cells_fallback_prompt("Lệch lớp?", workbook, [_structure(tmp_path / "s.yaml", "HP layer (VN)")])
    assert prompt is not None and "Question:\nLệch lớp?" in prompt and "B24=Lệch lớp xếp적층치우침" in prompt
    assert structure_sheet_names([tmp_path / "s.yaml", tmp_path / "missing.yaml"]) == ["HP layer (VN)"]


def test_failed_qa_without_verified_observations_answers_from_the_cells(tmp_path: Path, monkeypatch) -> None:
    import TableAgent.pipeline.table_agent_pipeline as pipeline_module
    from TableAgent.pipeline import TableAgentPipeline
    from TableAgent.schema.qa import QAResult

    from tests.test_table_agent_pipeline import FakeLayoutVLM, FakeLLM

    workbook = _form_workbook(tmp_path / "book.xlsx")
    structure = _structure(tmp_path / "structure.yaml", "HP layer (VN)")

    class FailingRunner:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def run(self, question):
            from TableAgent.schema.qa import AgentOutput
            from TableAgent.schema.subtask import SubTask

            # One inspection "succeeded" on misread cells, as in the HP layer runs.
            return QAResult(
                question=question,
                plan=[SubTask(id="inspect", description="Read the row.", layer="inspect")],
                subtask_outputs=[AgentOutput("inspect", "", "", True, "standard=Không được phép", namespace_updates={})],
                success=False,
                error="Table shape: (0, 0)",
            )

    monkeypatch.setattr(pipeline_module, "TableQARunner", FailingRunner)
    from TableAgent.configs import TableAgentConfig, load_config

    real_from_config = TableAgentConfig.from_config
    monkeypatch.setattr(
        TableAgentConfig,
        "from_config",
        staticmethod(
            lambda config=None: real_from_config(
                {
                    **load_config("config.example.yaml")["table_agent"],
                    **(config or {}),
                    "structure_cache_dir": str(tmp_path / "structure-cache"),
                }
            )
        ),
    )
    pipeline = TableAgentPipeline(
        llm_client=FakeLLM(), layout_vlm_client=FakeLayoutVLM(), config={"artifact_dir": str(tmp_path)}
    )
    prompts = []

    def answer(prompt, image_path=None, fallback_prompt=None):
        prompts.append(prompt)
        return LLMResponse(content="Coupon 3 chấm lệch quá 1/3 hole là NG")

    monkeypatch.setattr(pipeline.qa_agent, "run", answer)

    response, qa_info = pipeline._run_verified_qa(
        question="Tiêu chuẩn lệch lớp xếp?",
        structure_path=structure,
        workbook_path=workbook,
        qa_artifact_dir=tmp_path / "qa",
        fallback_prompt="cards only",
    )

    assert qa_info["fallback_source"] == "sheet_cells"
    assert response.content == "Coupon 3 chấm lệch quá 1/3 hole là NG"
    assert "B24=Lệch lớp xếp적층치우침" in prompts[0] and "cards only" not in prompts[0]


def test_observations_of_the_failed_run_join_only_a_cut_sheet(tmp_path: Path) -> None:
    workbook = _form_workbook(tmp_path / "book.xlsx")
    structure = [_structure(tmp_path / "s.yaml", "HP layer (VN)")]
    observations = "## Verified inspection: read row\nstandard=Không được phép"

    whole = sheet_cells_fallback_prompt("Lệch lớp?", workbook, structure, observations=observations)
    cut = sheet_cells_fallback_prompt("Lệch lớp?", workbook, structure, max_chars=300, observations=observations)

    # The whole sheet is enough; misread observations would only leak into the answer.
    assert "Không được phép" not in whole
    assert cut.index("standard=Không được phép") < cut.index("## Sheet: HP layer (VN) ")
    assert "trust the cells below when they disagree" in cut
