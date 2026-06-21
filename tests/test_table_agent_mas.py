from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import yaml

from TableAgent.agents import LayoutAgent, VerificationAgent
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
    worksheet["B2"] = 12.5
    worksheet["B2"].number_format = "0.00"
    workbook.save(path)


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
                "table_candidates": ["A1:B2"],
            }
        }
    }

    metadata = ExStructMetadataExtractor("light").sheet_metadata(workbook_path, payload, "Sheet1")

    assert metadata.used_range == "A1:B2"
    assert metadata.merged_ranges == ["A1:B1"]
    assert metadata.table_candidates == ["A1:B2"]
    assert metadata.number_formats == {"0.00": ["B2"]}
    assert list(yaml.safe_load(metadata.to_yaml())) == [
        "sheet_name", "used_range", "merged_ranges", "number_formats", "table_candidates"
    ]


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
    metadata = SheetMetadata("Sheet1", "A1:AN10", [], {}, ["A1:AN10"])

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
        metadata=SheetMetadata("Sheet1", "A1:A2", [], {}, ["A1:A2"]),
        output_dir=tmp_path / "retry-artifacts",
    )

    structure = yaml.safe_load(result.structure_text)
    header = structure["table1"]["headers"][0]
    assert result.iterations == 2
    assert header["header_range"] is None
    assert header["data_range"] is None
    assert "retries exhausted" in (tmp_path / "retry-artifacts" / "changelog.md").read_text().lower()
