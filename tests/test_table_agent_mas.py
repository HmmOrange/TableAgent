from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import yaml

from TableAgent.agents import LayoutAgent, VerificationAgent, _VERIFIER_CODE, _execute_verifier
from TableAgent.config import TableAgentConfig
from TableAgent.perception.metadata import ExStructMetadataExtractor, SheetMetadata
from TableAgent.pipeline.layout_workflow import TableLayoutWorkflow
from TableAgent.pipeline.traversal import Direction, DirectionQueue, TraversalTask, Viewport
from TableAgent.rendering.workbook import WorkbookRenderer
from table2img.core import RenderResult
from utils.llm.base import LLMResponse


class StaticLayoutVLM:
    model_name = "layout"
    temperature = 0.0

    def __init__(self, header_range: str = "A1:A1"):
        self.header_range = header_range

    def generate_with_image(self, prompt, image_path, system_prompt=None):
        structure = {
            "table1": {
                "name": "Sales",
                "description": "Sales table",
                "headers": [{
                    "label": "Region",
                    "description": "Sales region",
                    "orientation": "column",
                    "header_range": self.header_range,
                    "data_range": "A2:A10",
                    "sub_headers": [],
                }],
            }
        }
        return LLMResponse(content=yaml.safe_dump({
            "structure": structure,
            "changelog": "Added the Region header.",
            "remaining_directions": [],
        }, sort_keys=False))


class GoodVerificationLLM:
    model_name = "verifier"
    temperature = 0.0

    def generate(self, prompt, system_prompt=None):
        return LLMResponse(content="status: good\nfeedback: Verified.\nnull_fields: []\n")


class RecordingRenderer:
    def __init__(self):
        self.ranges = []

    def __call__(self, document, image_path, **kwargs):
        image_path = Path(image_path)
        self.ranges.append(document)
        image_path.write_bytes(b"viewport")
        html_path = image_path.with_suffix(".html")
        html_path.write_text(document.html, encoding="utf-8")
        return RenderResult(image_path, html_path, 100, 80, Path("fake"))


def _settings(tmp_path: Path, **override) -> TableAgentConfig:
    return TableAgentConfig.from_config({
        "artifact_dir": str(tmp_path),
        "viewport_rows": 20,
        "viewport_columns": 20,
        "shift_cells": 15,
        "max_retry": 5,
        **override,
    })


def _workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet["A1"] = "Region"
    worksheet["A2"] = "North"
    for row in range(3, 11):
        worksheet[f"A{row}"] = f"Region {row}"
    worksheet["B2"] = 12.5
    worksheet["B2"].number_format = "0.00"
    workbook.save(path)


def _hierarchical_workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet["A1"] = "Month"
    worksheet["A2"] = "In"
    worksheet["B2"] = "Out"
    worksheet["A3"] = 1
    worksheet["B3"] = 2
    workbook.save(path)


def _run_verifier(tmp_path: Path, workbook_path: Path, structure: dict) -> dict:
    verifier_path = tmp_path / "verification.py"
    structure_path = tmp_path / "structure_after.yaml"
    verifier_path.write_text(_VERIFIER_CODE, encoding="utf-8")
    structure_path.write_text(yaml.safe_dump(structure, sort_keys=False), encoding="utf-8")
    return _execute_verifier(verifier_path, workbook_path, "Sheet1", structure_path)


def test_direction_queue_uses_required_priority():
    queue = DirectionQueue()
    viewport = Viewport(1, 1, 20, 20)
    for direction in [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT, Direction.STAY]:
        queue.push(TraversalTask(direction, viewport.shifted(direction, 15)))

    assert [queue.pop().direction for _ in range(5)] == [
        Direction.STAY,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT,
        Direction.UP,
    ]


def test_exstruct_payload_becomes_metadata_yaml(tmp_path: Path):
    workbook_path = tmp_path / "book.xlsx"
    _workbook(workbook_path)
    payload = {
        "sheets": {
            "Sheet1": {
                "rows": [
                    {"r": 1, "c": {"A": "Region"}},
                    {"r": 2, "c": {"A": "North", "B": 12.5}},
                ],
                "merged_ranges": ["A1:B1"],
            }
        }
    }

    metadata = ExStructMetadataExtractor("light").sheet_metadata(workbook_path, payload, "Sheet1")

    assert metadata.used_range == "A1:B2"
    assert metadata.merged_ranges == ["A1:B1"]
    assert list(yaml.safe_load(metadata.to_yaml())) == ["sheet_name", "used_range", "merged_ranges"]


def test_verifier_accepts_consistent_header_and_sub_header_ranges(tmp_path: Path):
    workbook_path = tmp_path / "book.xlsx"
    _hierarchical_workbook(workbook_path)
    structure = {
        "table1": {
            "name": "Movement",
            "description": "Movement by month",
            "headers": [{
                "label": "Month",
                "description": "Month group",
                "orientation": "column",
                "header_range": "A1:B1",
                "data_range": "A3:B3",
                "sub_headers": [
                    {
                        "label": "In",
                        "description": "Inbound",
                        "orientation": "column",
                        "header_range": "A2:A2",
                        "data_range": "A3:A3",
                    },
                    {
                        "label": "Out",
                        "description": "Outbound",
                        "orientation": "column",
                        "header_range": "B2:B2",
                        "data_range": "B3:B3",
                    },
                ],
            }],
        }
    }

    report = _run_verifier(tmp_path, workbook_path, structure)

    assert report == {"status": "good", "errors": []}


def test_verifier_rejects_header_text_and_data_range_mismatches(tmp_path: Path):
    workbook_path = tmp_path / "book.xlsx"
    _hierarchical_workbook(workbook_path)
    structure = {
        "table1": {
            "name": "Movement",
            "description": "Movement by month",
            "headers": [{
                "label": "Month",
                "description": "Month group",
                "orientation": "column",
                "header_range": "A1:A2",
                "data_range": "A2:B3",
                "sub_headers": [{
                    "label": "In",
                    "description": "Inbound",
                    "orientation": "column",
                    "header_range": "A2:A2",
                    "data_range": "A3:A3",
                }],
            }],
        }
    }

    report = _run_verifier(tmp_path, workbook_path, structure)

    assert report["status"] == "not_good"
    assert any("header_range contains unrelated text" in error for error in report["errors"])
    assert any("data_range overlaps" in error for error in report["errors"])


def test_workflow_continues_direction_once_after_first_no_change(tmp_path: Path):
    workbook_path = tmp_path / "book.xlsx"
    _workbook(workbook_path)
    settings = _settings(tmp_path)
    recording_renderer = RecordingRenderer()
    renderer = WorkbookRenderer(settings, recording_renderer, logger=None)
    workflow = TableLayoutWorkflow(
        settings,
        renderer,
        LayoutAgent(StaticLayoutVLM()),
        VerificationAgent(GoodVerificationLLM()),
    )
    metadata = SheetMetadata("Sheet1", "A1:AN10", [])

    result = workflow.run(
        workbook_path=workbook_path,
        sheet_name="Sheet1",
        metadata=metadata,
        output_dir=tmp_path / "artifacts",
    )

    events = [json.loads(line) for line in (tmp_path / "artifacts" / "events.jsonl").read_text().splitlines()]
    assert [(event["direction"], event["viewport"]) for event in events] == [
        ("stay", "A1:T20"),
        ("right", "P1:AI20"),
        ("right", "AE1:AX20"),
    ]
    assert result.iterations == 3
    assert (tmp_path / "artifacts" / "metadata.yaml").is_file()
    assert (tmp_path / "artifacts" / "changelog.md").is_file()
    for iteration_dir in (tmp_path / "artifacts" / "iterations").iterdir():
        assert (iteration_dir / "viewport.png").is_file()
        assert (iteration_dir / "layout_prompt.txt").is_file()
        assert (iteration_dir / "verification.py").is_file()
        assert (iteration_dir / "verification_output.json").is_file()


def test_workflow_nulls_ranges_after_max_retry(tmp_path: Path):
    workbook_path = tmp_path / "book.xlsx"
    _workbook(workbook_path)
    settings = _settings(tmp_path, max_retry=2)
    renderer = WorkbookRenderer(settings, RecordingRenderer(), logger=None)
    workflow = TableLayoutWorkflow(
        settings,
        renderer,
        LayoutAgent(StaticLayoutVLM(header_range="NOT_A_RANGE")),
        VerificationAgent(GoodVerificationLLM()),
    )

    result = workflow.run(
        workbook_path=workbook_path,
        sheet_name="Sheet1",
        metadata=SheetMetadata("Sheet1", "A1:A2", []),
        output_dir=tmp_path / "retry-artifacts",
    )

    structure = yaml.safe_load(result.structure_text)
    header = structure["table1"]["headers"][0]
    assert result.iterations == 2
    assert header["header_range"] is None
    assert header["data_range"] is None
    assert "retries exhausted" in (tmp_path / "retry-artifacts" / "changelog.md").read_text().lower()


def test_workflow_ignores_suggested_direction_outside_used_range(tmp_path: Path):
    class RightSuggestingLayoutVLM(StaticLayoutVLM):
        def generate_with_image(self, prompt, image_path, system_prompt=None):
            response = yaml.safe_load(super().generate_with_image(prompt, image_path, system_prompt).content)
            response["remaining_directions"] = ["right"]
            return LLMResponse(content=yaml.safe_dump(response, sort_keys=False))

    workbook_path = tmp_path / "book.xlsx"
    _workbook(workbook_path)
    settings = _settings(tmp_path)
    renderer = WorkbookRenderer(settings, RecordingRenderer(), logger=None)
    workflow = TableLayoutWorkflow(
        settings,
        renderer,
        LayoutAgent(RightSuggestingLayoutVLM()),
        VerificationAgent(GoodVerificationLLM()),
    )

    workflow.run(
        workbook_path=workbook_path,
        sheet_name="Sheet1",
        metadata=SheetMetadata("Sheet1", "A1:Q51", []),
        output_dir=tmp_path / "bounded-artifacts",
    )

    events = [json.loads(line) for line in (tmp_path / "bounded-artifacts" / "events.jsonl").read_text().splitlines()]
    assert [event["viewport"] for event in events] == ["A1:T20", "A16:T35", "A31:T50"]
    assert all(event["direction"] != "right" for event in events)
