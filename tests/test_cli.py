from __future__ import annotations

import json

import pytest

from service import cli


def test_cli_parser_accepts_repeatable_workbooks_queries_and_profiles():
    args = cli.build_parser().parse_args(
        [
            "--config",
            "private.yaml",
            "--stage",
            "all",
            "--workbook",
            "sales.xlsx",
            "--workbook",
            "costs.xlsx",
            "--query",
            "Total revenue?",
            "--query",
            "Largest cost?",
            "--llm",
            "alternate_answer",
            "--vlm",
            "alternate_layout",
        ]
    )

    assert args.config == "private.yaml"
    assert args.stage == "all"
    assert args.workbook == ["sales.xlsx", "costs.xlsx"]
    assert args.query == ["Total revenue?", "Largest cost?"]
    assert args.schema is False
    assert args.metadata is False
    assert args.force is False
    assert args.sheet == []
    assert args.llm == "alternate_answer"
    assert args.vlm == "alternate_layout"


def test_cli_parser_accepts_artifact_and_sheet_flags():
    args = cli.build_parser().parse_args(
        [
            "--stage",
            "structure",
            "--workbook",
            "book.xlsx",
            "--schema",
            "--metadata",
            "--sheet",
            "Summary,Detail",
            "--sheet",
            "Archive",
        ]
    )

    assert args.schema is True
    assert args.metadata is True
    assert args.sheet == ["Summary,Detail", "Archive"]


@pytest.mark.parametrize("stage", ["qa", "all"])
def test_cli_requires_query_for_answering_stages(stage):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--stage", stage, "--workbook", "book.xlsx"])

    assert exc_info.value.code == 2


def test_cli_rejects_force_for_qa_stage():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--stage",
                "qa",
                "--workbook",
                "book.xlsx",
                "--query",
                "Question?",
                "--force",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_runs_structure_stage_and_prints_json(monkeypatch, capsys):
    captured = {}

    class FakeTableAgentService:
        @staticmethod
        def from_config(path, **kwargs):
            captured["config"] = path
            captured.update(kwargs)
            return FakeTableAgentService()

        def run(self, **kwargs):
            captured.update(kwargs)
            return {"job_id": "job-one", "stage": kwargs["stage"]}

    monkeypatch.setattr(cli, "TableAgentService", FakeTableAgentService)

    result = cli.main(
        [
            "--config",
            "private.yaml",
            "--stage",
            "structure",
            "--workbook",
            "sales.xlsx",
            "--workbook",
            "costs.xlsx",
            "--llm",
            "alternate_answer",
            "--vlm",
            "alternate_layout",
            "--force",
        ]
    )

    assert result == 0
    assert captured == {
        "config": "private.yaml",
        "llm_profile": "alternate_answer",
        "vlm_profile": "alternate_layout",
        "stage": "structure",
        "workbooks": ["sales.xlsx", "costs.xlsx"],
        "queries": [],
        "schema": False,
        "metadata": False,
        "sheets": [],
        "force": True,
    }
    assert json.loads(capsys.readouterr().out) == {"job_id": "job-one", "stage": "structure"}


def test_cli_reports_expected_runtime_errors(monkeypatch, capsys):
    class FakeTableAgentService:
        @staticmethod
        def from_config(path, **kwargs):
            raise FileNotFoundError("Config file not found: missing.yaml")

    monkeypatch.setattr(cli, "TableAgentService", FakeTableAgentService)

    result = cli.main(
        ["--config", "missing.yaml", "--stage", "structure", "--workbook", "book.xlsx"]
    )

    assert result == 1
    assert "Config file not found: missing.yaml" in capsys.readouterr().err


def test_cli_reconfigures_stdout_for_unicode_json(monkeypatch):
    class EncodedStdout:
        def __init__(self):
            self.encoding = "cp1252"
            self.parts = []

        def reconfigure(self, *, encoding):
            self.encoding = encoding

        def write(self, value):
            value.encode(self.encoding)
            self.parts.append(value)

        def flush(self):
            pass

    class FakeTableAgentService:
        @staticmethod
        def from_config(path, **kwargs):
            return FakeTableAgentService()

        def run(self, **kwargs):
            return {"answer": "Nguyen Thi H\u1ef1u"}

    stdout = EncodedStdout()
    monkeypatch.setattr(cli, "TableAgentService", FakeTableAgentService)
    monkeypatch.setattr(cli.sys, "stdout", stdout)

    result = cli.main(["--stage", "structure", "--workbook", "book.xlsx"])

    assert result == 0
    assert stdout.encoding == "utf-8"
    assert json.loads("".join(stdout.parts))["answer"] == "Nguyen Thi H\u1ef1u"
