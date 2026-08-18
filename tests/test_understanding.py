import json
from pathlib import Path

import openpyxl
import pytest
import yaml

from TableAgent.llm import LLMResponse
from TableAgent.stages.understanding import UnderstandingInput, UnderstandingStage
from TableAgent.stages.understanding.parsing import parse_understanding
from TableAgent.stages.understanding.stage import _used_range
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


def test_understanding_uses_complete_worksheet_range(tmp_path: Path):
    assert _used_range(_workbook(tmp_path / "large.xlsx", rows=80, columns=60), "Summary") == "A1:BH80"
    assert _used_range(_workbook(tmp_path / "small.xlsx", rows=7, columns=4), "Summary") == "A1:D7"


def test_parser_returns_flat_lists_and_deduplicates():
    result = parse_understanding(
        yaml.safe_dump(
            {
                "headers": ["Indexes", "June", "June"],
                "groups": ["North", "North"],
            }
        )
    )

    assert result.headers == ("Indexes", "June")
    assert result.groups == ("North",)


@pytest.mark.parametrize(
    "payload",
    [
        {"headers": []},
        {
            "headers": [],
            "groups": [],
            "extra": [],
        },
        {
            "headers": [1],
            "groups": [],
        },
        {"headers": [{"label": "Amount"}], "groups": []},
    ],
)
def test_parser_rejects_invalid_contracts(payload):
    with pytest.raises(ValueError):
        parse_understanding(json.dumps(payload))


def test_stage_renders_writes_artifacts_and_repairs_once(tmp_path: Path):
    workbook_path = _workbook(tmp_path / "book.xlsx", rows=8, columns=5)
    renderer = FakeRenderer()
    valid = yaml.safe_dump(
        {
            "headers": ["2026", "Revenue"],
            "groups": ["Region"],
        }
    )
    vlm = FakeVLM(["not json", valid])
    output = UnderstandingStage(renderer, vlm).run(
        UnderstandingInput(workbook_path, "Summary", tmp_path / "artifacts")
    )

    assert renderer.calls[0][2] == "A1:E8"
    assert len(vlm.calls) == 2
    assert output.understanding.groups[0] == "Region"
    assert output.result_path.is_file()
    assert (tmp_path / "artifacts" / "worksheet.png").is_file()
    assert (tmp_path / "artifacts" / "repair_response.txt").is_file()
    result = yaml.safe_load(output.result_path.read_text(encoding="utf-8"))
    assert result["headers"] == ["2026", "Revenue"]


def test_service_runs_understanding_without_structure_or_query(tmp_path: Path, monkeypatch):
    source = _workbook(tmp_path / "book.xlsx", rows=3, columns=2)
    response = yaml.safe_dump(
        {
            "headers": ["Amount"],
            "groups": ["Product"],
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
    assert result["understandings"][0]["groups"] == ["Product"]
    assert "viewport" not in result["understandings"][0]
    assert result["understandings"][0]["artifact"].endswith("understanding.yaml")
    assert (tmp_path / "output" / "understanding-run" / "run.json").is_file()
