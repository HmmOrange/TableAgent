"""The QA service: jobs over one workbook, through the runtime, the HTTP API and the command line."""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pytest
from fastapi.testclient import TestClient

from TableAgent.llm import LLMResponse
from service import cli
from service.api import create_app
from service.runtime import TableAgentService


class FailingLLM:
    """Every reply is unusable, so each question ends in the fallback answer: one model call per question."""

    model_name = "fake"
    temperature = 0.0
    max_tokens = None

    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(content="not json" if "Workbook preview" in prompt or "before anything" in prompt else "42")


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    path = tmp_path / "book.xlsx"
    book = openpyxl.Workbook()
    book.active.append(["Year", "Value"])
    book.active.append([2021, 42])
    book.save(path)
    return path


def _service(tmp_path: Path, **service) -> TableAgentService:
    config = {"service": {"root_dir": str(tmp_path / "jobs"), **service}}
    return TableAgentService(config, llm_client=FailingLLM())


def test_a_job_answers_each_question_in_its_own_directory(tmp_path: Path, workbook: Path):
    result = _service(tmp_path, max_workers=2).run(workbook, ["What is the 2021 value?", " ", "And the year?"])

    assert [answer["question"] for answer in result["answers"]] == ["What is the 2021 value?", "And the year?"]
    assert all(answer["answer"] == "42" for answer in result["answers"])
    assert all(answer["qa"]["fallback_source"] == "source_context" for answer in result["answers"])
    job_dir = Path(result["job_dir"])
    assert (job_dir / "book.xlsx").is_file() and "book.xlsx" in result["artifacts"]
    assert {path.parent.name for path in (job_dir / "qa").glob("q*/*")} == {"q1", "q2"}


def test_a_job_needs_a_question_and_an_xlsx_workbook(tmp_path: Path, workbook: Path):
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="At least one non-empty question"):
        service.run(workbook, ["  "])
    legacy = tmp_path / "book.xls"
    legacy.write_bytes(b"old")
    with pytest.raises(ValueError, match="Unsupported workbook type"):
        service.run(legacy, ["What?"])


def test_local_paths_are_off_by_default_and_limited_to_the_allowed_roots(tmp_path: Path, workbook: Path):
    with pytest.raises(PermissionError, match="disabled"):
        _service(tmp_path).validate_local_workbook(workbook)
    allowed = _service(tmp_path, allow_local_paths=True, allowed_input_roots=[str(tmp_path / "elsewhere")])
    with pytest.raises(PermissionError, match="outside the allowed input roots"):
        allowed.validate_local_workbook(workbook)


def test_the_api_answers_an_upload_and_checks_the_api_key(tmp_path: Path, workbook: Path):
    client = TestClient(create_app(_service(tmp_path, api_key="secret")))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/status").status_code == 401

    payload = {"payload": json.dumps({"questions": ["What is the 2021 value?"]})}
    files = {"file": ("book.xlsx", workbook.read_bytes())}
    response = client.post("/v1/jobs/upload", data=payload, files=files, headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    assert response.json()["answers"][0]["answer"] == "42"

    bad = client.post("/v1/jobs", json={"workbook": str(workbook), "questions": ["What?"]}, headers={"X-API-Key": "secret"})
    assert bad.status_code == 403


def test_the_command_line_prints_the_job(tmp_path: Path, workbook: Path, monkeypatch, capsys):
    monkeypatch.setattr(cli.TableAgentService, "from_config", classmethod(lambda cls, path, **kwargs: _service(tmp_path)))
    assert cli.main(["--workbook", str(workbook), "--question", "What is the 2021 value?"]) == 0
    assert json.loads(capsys.readouterr().out)["answers"][0]["answer"] == "42"
