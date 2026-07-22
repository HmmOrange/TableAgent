from __future__ import annotations

from pathlib import Path

import pytest

from TableAgent.integrations.qa import TableQAEngine, TableQARequest
from TableAgent.schema.qa import QAResult


class FakeRunner:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).calls.append(kwargs)

    def run(self, question: str) -> QAResult:
        return QAResult(
            question=question,
            plan=[],
            final_answer="42",
            success=True,
            execution_time=0.25,
            artifacts={"notebook": "qa/run.ipynb"},
            token_usage={"prompt": 4, "completion": 2},
        )


def _request(tmp_path: Path) -> TableQARequest:
    workbook_path = tmp_path / "book.xlsx"
    workbook_path.write_bytes(b"fixture")
    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text("table1:\n  headers: []\n", encoding="utf-8")
    return TableQARequest(
        question="What is the answer?",
        workbook_path=workbook_path,
        structure_path=structure_path,
        artifact_dir=tmp_path / "artifacts",
    )


def test_engine_accepts_external_structure_without_model_calls(tmp_path: Path) -> None:
    FakeRunner.calls = []
    engine = TableQAEngine(runner_factory=FakeRunner)

    response = engine.answer(_request(tmp_path))

    assert response.success is True
    assert response.answer == "42"
    assert response.artifacts == {"notebook": "qa/run.ipynb"}
    assert response.token_usage == {"prompt": 4, "completion": 2}
    assert FakeRunner.calls[0]["llm_client"] is None
    assert FakeRunner.calls[0]["config"] == {
        "qa_artifact_dir": str((tmp_path / "artifacts").resolve())
    }


def test_engine_rejects_missing_structure_before_constructing_runner(
    tmp_path: Path,
) -> None:
    FakeRunner.calls = []
    request = _request(tmp_path)
    request.structure_path.unlink()
    engine = TableQAEngine(runner_factory=FakeRunner)

    with pytest.raises(FileNotFoundError, match="Structure file not found"):
        engine.answer(request)

    assert FakeRunner.calls == []


def test_engine_rejects_sheet_specific_structure_filename(tmp_path: Path) -> None:
    FakeRunner.calls = []
    request = _request(tmp_path)
    renamed_path = request.structure_path.with_name("sales.yaml")
    request.structure_path.rename(renamed_path)
    request = TableQARequest(
        question=request.question,
        workbook_path=request.workbook_path,
        structure_path=renamed_path,
        artifact_dir=request.artifact_dir,
    )
    engine = TableQAEngine(runner_factory=FakeRunner)

    with pytest.raises(ValueError, match="must be named structure.yaml"):
        engine.answer(request)

    assert FakeRunner.calls == []
