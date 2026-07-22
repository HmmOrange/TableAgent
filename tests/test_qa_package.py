from __future__ import annotations

import json
from pathlib import Path

from TableAgent.integrations.qa import TableQAEngine, TableQAResponse
from TableAgent.integrations.qa_cli import main
from TableAgent.integrations.qa_package import (
    TableQAPackage,
    load_qa_request,
    qa_response_payload,
)
from TableAgent.schema.qa import QAResult


class FakeRunner:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def run(self, question: str) -> QAResult:
        return QAResult(
            question=question,
            plan=[],
            final_answer="20",
            success=True,
            execution_time=0.25,
            artifacts={"notebook": "qa/notebook.ipynb"},
            token_usage={"prompt": 5, "completion": 2},
        )


def _request_file(tmp_path: Path) -> Path:
    (tmp_path / "input.xlsx").write_bytes(b"fixture")
    (tmp_path / "structure.yaml").write_text(
        "people:\n  headers: []\n", encoding="utf-8"
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "question": "How many people are listed?",
                "workbook_path": "input.xlsx",
                "structure_path": "structure.yaml",
                "artifact_dir": "qa-output",
                "table_id": "people",
            }
        ),
        encoding="utf-8",
    )
    return request_path


def test_package_maps_json_request_to_json_response(tmp_path: Path) -> None:
    request = load_qa_request(_request_file(tmp_path))
    package = TableQAPackage(TableQAEngine(runner_factory=FakeRunner))

    response = package.run(request)
    payload = qa_response_payload(response)

    assert request.workbook_path == tmp_path / "input.xlsx"
    assert request.structure_path == tmp_path / "structure.yaml"
    assert request.artifact_dir == tmp_path / "qa-output"
    assert payload == {
        "schema_version": 1,
        "success": True,
        "answer": "20",
        "error": None,
        "execution_time": 0.25,
        "artifacts": {"notebook": "qa/notebook.ipynb"},
        "token_usage": {"prompt": 5, "completion": 2},
        "limitations": [],
    }


def test_cli_runs_package_request_to_output(
    tmp_path: Path,
    capsys,
) -> None:
    request_path = _request_file(tmp_path)
    output_path = tmp_path / "response.json"

    class FakePackage:
        closed = False

        def run(self, request) -> TableQAResponse:
            assert request.question == "How many people are listed?"
            return TableQAResponse(
                success=True,
                answer="20",
                error=None,
                execution_time=0.1,
                artifacts={},
                token_usage={},
            )

        def close(self) -> None:
            self.closed = True

    package = FakePackage()

    def factory(config_path, *, llm_profile):
        assert config_path == Path("config.yaml")
        assert llm_profile == "answer"
        return package

    exit_code = main(
        [
            "--request",
            str(request_path),
            "--output",
            str(output_path),
            "--llm-profile",
            "answer",
        ],
        package_factory=factory,
    )

    assert exit_code == 0
    assert package.closed is True
    assert json.loads(output_path.read_text(encoding="utf-8"))["answer"] == "20"
    assert json.loads(capsys.readouterr().out)["success"] is True
