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
    assert args.llm == "alternate_answer"
    assert args.vlm == "alternate_layout"


@pytest.mark.parametrize("stage", ["qa", "all"])
def test_cli_requires_query_for_answering_stages(stage):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--stage", stage, "--workbook", "book.xlsx"])

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
