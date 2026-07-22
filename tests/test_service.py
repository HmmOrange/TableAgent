from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import openpyxl

from pipelines.base import PipelineOutput
from TableAgent.service import TableAgentService


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
        structure_path = Path(self.config["source_artifact_dir"]) / "fake" / "structure.yaml"
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        structure_path.write_text("table1:\n  headers: []\n", encoding="utf-8")
        return [
            SimpleNamespace(
                workbook_path=workbook_path,
                sheet_name="Sheet",
                structure_path=structure_path,
                status="good",
                cache_hit=False,
            )
        ]

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


def _workbook(path: Path) -> Path:
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = "value"
    workbook.save(path)
    workbook.close()
    return path


def test_service_runs_structure_once_and_answers_all_queries(tmp_path: Path):
    FakePipeline.instances = []
    answer_client = object()
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
    assert [item["answer"] for item in result["answers"]] == [
        "answer: first question",
        "answer: second question",
    ]
    assert "artifacts" not in result["answers"][0]["qa"]
    assert (service.jobs_dir / "job-one" / "run.json").is_file()


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
