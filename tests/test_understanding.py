import json
from pathlib import Path

import openpyxl
import pytest

from TableAgent.llm import LLMResponse
from TableAgent.stages.understanding import UnderstandingInput, UnderstandingStage
from TableAgent.stages.understanding.parsing import parse_understanding
from TableAgent.stages.understanding.stage import _top_left_viewport
from service.runtime import TableAgentService


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def source_viewport_to_image(self, source_path, sheet_name, cell_range, image_path):
        self.calls.append((Path(source_path), sheet_name, cell_range, Path(image_path)))
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")
        image_path.with_suffix(".metadata.json").write_text("{}", encoding="utf-8")


class FakeVLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_with_image(self, prompt, image_path, system_prompt=None):
        self.calls.append((prompt, Path(image_path), system_prompt))
        return LLMResponse(content=self.responses.pop(0))


def _workbook(path: Path, *, rows: int = 1, columns: int = 1) -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Summary"
    sheet.cell(rows, columns, "value")
    workbook.save(path)
    workbook.close()
    return path


def test_top_left_viewport_is_clipped_to_50_by_50(tmp_path: Path):
    assert _top_left_viewport(_workbook(tmp_path / "large.xlsx", rows=80, columns=60), "Summary") == "A1:AX50"
    assert _top_left_viewport(_workbook(tmp_path / "small.xlsx", rows=7, columns=4), "Summary") == "A1:D7"


def test_parser_requires_exactly_four_string_lists_and_deduplicates():
    result = parse_understanding(
        json.dumps(
            {
                "row_headers": ["North", "North", "South"],
                "column_headers": ["January"],
                "row_group_headers": [],
                "column_group_headers": ["Q1"],
            }
        )
    )

    assert result.row_headers == ("North", "South")
    assert result.column_headers == ("January",)
    assert result.row_group_headers == ()
    assert result.column_group_headers == ("Q1",)


@pytest.mark.parametrize(
    "payload",
    [
        {"row_headers": []},
        {
            "row_headers": [],
            "column_headers": [],
            "row_group_headers": [],
            "column_group_headers": [],
            "extra": [],
        },
        {
            "row_headers": [1],
            "column_headers": [],
            "row_group_headers": [],
            "column_group_headers": [],
        },
    ],
)
def test_parser_rejects_invalid_contracts(payload):
    with pytest.raises(ValueError):
        parse_understanding(json.dumps(payload))


def test_stage_renders_writes_artifacts_and_repairs_once(tmp_path: Path):
    workbook_path = _workbook(tmp_path / "book.xlsx", rows=8, columns=5)
    renderer = FakeRenderer()
    valid = json.dumps(
        {
            "row_headers": ["Region"],
            "column_headers": ["Revenue"],
            "row_group_headers": [],
            "column_group_headers": ["2026"],
        }
    )
    vlm = FakeVLM(["not json", valid])
    output = UnderstandingStage(renderer, vlm).run(
        UnderstandingInput(workbook_path, "Summary", tmp_path / "artifacts")
    )

    assert renderer.calls[0][2] == "A1:E8"
    assert len(vlm.calls) == 2
    assert output.understanding.row_headers == ("Region",)
    assert output.result_path.is_file()
    assert (tmp_path / "artifacts" / "viewport.png").is_file()
    assert (tmp_path / "artifacts" / "repair_response.txt").is_file()
    assert json.loads(output.result_path.read_text(encoding="utf-8"))["column_headers"] == ["Revenue"]


def test_service_runs_understanding_without_structure_or_query(tmp_path: Path, monkeypatch):
    source = _workbook(tmp_path / "book.xlsx", rows=3, columns=2)
    response = json.dumps(
        {
            "row_headers": ["Product"],
            "column_headers": ["Amount"],
            "row_group_headers": [],
            "column_group_headers": [],
        }
    )
    vlm = FakeVLM([response])
    renderer_settings = []

    class ServiceRenderer(FakeRenderer):
        def __init__(self, settings, logger):
            super().__init__()
            renderer_settings.append(settings)

    class ForbiddenPipeline:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Understanding must not construct the existing pipeline")

    monkeypatch.setattr("service.runtime.WorkbookRenderer", ServiceRenderer)
    service = TableAgentService(
        {
            "service": {"root_dir": str(tmp_path / "output")},
            "table_agent": {
                "artifact_dir": str(tmp_path / "unused"),
                "max_refinement_rounds": 1,
                "max_context_chars": 1000,
                "render_timeout_seconds": 10,
                "image_tile_overlap": 0,
            },
        },
        layout_vlm_client=vlm,
        pipeline_factory=ForbiddenPipeline,
    )

    result = service.run(
        stage="understanding",
        workbooks=[source],
        job_id="understanding-run",
    )

    assert result["stage"] == "understanding"
    assert renderer_settings[0].workbook_show_coordinates is False
    assert result["understandings"][0]["row_headers"] == ["Product"]
    assert "viewport" not in result["understandings"][0]
    assert result["understandings"][0]["artifact"].endswith("headers.json")
    assert (tmp_path / "output" / "understanding-run" / "run.json").is_file()
