from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import openpyxl
import yaml

from TableAgent.pipeline.base import PipelineOutput
from TableAgent.llm import LLMResponse
from service.runtime import TableAgentService


class FakePipeline:
    instances = []

    def __init__(self, llm_client, layout_vlm_client, config):
        self.llm_client = llm_client
        self.layout_vlm_client = layout_vlm_client
        self.config = config
        self.prepared = []
        self.runs = []
        type(self).instances.append(self)

    def verify_samples(self, samples, force=False):
        workbook_path = Path(samples[0].table_path.split(";")[0])
        workbook = openpyxl.load_workbook(workbook_path, read_only=True)
        try:
            sheet_names = list(workbook.sheetnames)
        finally:
            workbook.close()
        selected = samples[0].raw.get("selected_sheets") or sheet_names
        records = []
        for sheet_name in selected:
            structure_path = (
                Path(self.config["source_artifact_dir"])
                / "fake"
                / sheet_name
                / "structure.yaml"
            )
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_text(
                f"table1:\n  name: {sheet_name}\n  headers: []\n",
                encoding="utf-8",
            )
            records.append(
                SimpleNamespace(
                    workbook_path=workbook_path,
                    sheet_name=sheet_name,
                    structure_path=structure_path,
                    status="good",
                    cache_hit=False,
                )
            )
        return records

    def prepare_samples(self, samples):
        self.prepared.extend(samples)

    def run(self, sample):
        self.runs.append(sample)
        workbook_path = sample.table_path.split(";")[0]
        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table="table1:\n  headers: []\n",
            predicted_answer=f"answer: {sample.question}",
            latency=0.25,
            token_usage={"prompt": 4, "completion": 2},
            metadata={
                "workbook_path": workbook_path,
                "workbook_sheets": ["Sheet"],
                "verification": {"status": "good"},
                "qa": {"success": True, "artifacts": {"private": "path"}},
            },
        )


class FakeSummaryClient:
    def generate(self, prompt, system_prompt=None):
        description = "Workbook summary" if "workbook as a whole" in prompt else "Sheet summary"
        return LLMResponse(content=f'{{"description": "{description}"}}')


def _workbook(path: Path) -> Path:
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = "value"
    workbook.save(path)
    workbook.close()
    return path


def _multi_sheet_workbook(path: Path) -> Path:
    workbook = openpyxl.Workbook()
    workbook.active.title = "Summary"
    workbook.create_sheet("Detail")
    workbook.create_sheet("Archive")
    workbook.save(path)
    workbook.close()
    return path


def test_service_runs_structure_once_and_answers_all_queries(tmp_path: Path):
    FakePipeline.instances = []
    answer_client = FakeSummaryClient()
    layout_client = object()
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=answer_client,
        layout_vlm_client=layout_client,
        pipeline_factory=FakePipeline,
    )

    result = service.run(
        stage="all",
        workbooks=[source],
        queries=["first question", "second question"],
        job_id="job-one",
    )

    assert len(FakePipeline.instances) == 2
    assert FakePipeline.instances[0].layout_vlm_client is layout_client
    assert FakePipeline.instances[1].llm_client is answer_client
    assert len(FakePipeline.instances[1].runs) == 2
    assert result["workbooks"] == ["book.xlsx"]
    assert result["structures"][0]["artifact"].endswith(".yaml")
    assert result["schema_artifacts"] == [
        {"workbook": "book.xlsx", "artifact": "workbooks/book.xlsx/schema.yaml"}
    ]
    assert result["metadata_artifacts"] == [
        {"workbook": "book.xlsx", "artifact": "workbooks/book.xlsx/metadata.json"}
    ]
    assert [item["answer"] for item in result["answers"]] == [
        "answer: first question",
        "answer: second question",
    ]
    assert "artifacts" not in result["answers"][0]["qa"]
    assert (service.jobs_dir / "job-one" / "run.json").is_file()


def test_metadata_only_structure_stage_skips_pipeline_and_vlm(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        pipeline_factory=FakePipeline,
    )

    result = service.run(
        stage="structure",
        workbooks=[source],
        metadata=True,
        job_id="metadata-only",
    )

    assert FakePipeline.instances == []
    assert result["structures"] == []
    assert result["schema_artifacts"] == []
    assert result["metadata_artifacts"][0]["artifact"] == "workbooks/book.xlsx/metadata.json"


def test_service_normalizes_repeated_comma_separated_sheet_filters(tmp_path: Path):
    FakePipeline.instances = []
    source = _multi_sheet_workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        pipeline_factory=FakePipeline,
    )

    result = service.run(
        stage="structure",
        workbooks=[source],
        schema=True,
        sheets=["Summary, Detail", "Summary"],
        job_id="selected-sheets",
    )

    assert [item["sheet"] for item in result["structures"]] == ["Summary", "Detail"]
    schema_path = service.jobs_dir / "selected-sheets" / result["schema_artifacts"][0]["artifact"]
    assert list(yaml.safe_load(schema_path.read_text(encoding="utf-8"))) == ["Summary", "Detail"]


def test_service_rejects_missing_sheet_before_pipeline_work(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        pipeline_factory=FakePipeline,
    )

    try:
        service.run(
            stage="structure",
            workbooks=[source],
            metadata=True,
            sheets=["Missing"],
        )
    except ValueError as exc:
        assert "book.xlsx: Missing" in str(exc)
    else:
        raise AssertionError("Expected a missing-sheet validation error")

    assert FakePipeline.instances == []


def test_service_rejects_queries_missing_from_qa_stage(tmp_path: Path):
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=object(),
        pipeline_factory=FakePipeline,
    )
    source = _workbook(tmp_path / "book.xlsx")

    try:
        service.run(stage="qa", workbooks=[source], queries=[])
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("Expected an empty-query validation error")


def test_local_paths_are_disabled_by_default(tmp_path: Path):
    service = TableAgentService({"service": {"root_dir": str(tmp_path / "service")}})
    source = _workbook(tmp_path / "book.xlsx")

    try:
        service.validate_local_workbook(source)
    except PermissionError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("Expected local paths to be disabled")


def test_service_uses_explicit_model_profiles(monkeypatch, tmp_path: Path):
    calls = []

    def fake_create_model_client(config, *, kind, profile):
        calls.append((kind, profile))
        return object()

    monkeypatch.setattr("service.runtime.create_model_client", fake_create_model_client)
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_profile="alternate_answer",
        vlm_profile="alternate_layout",
    )

    service._answer_client()
    service._layout_client()

    assert calls == [
        ("llm", "alternate_answer"),
        ("vlm", "alternate_layout"),
    ]
