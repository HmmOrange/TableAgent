from __future__ import annotations

import json
import pickle

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
            "-em",
            "alternate_embedding",
            "--workers",
            "6",
        ]
    )

    assert args.config == "private.yaml"
    assert args.stage == "all"
    assert args.workbook == ["sales.xlsx", "costs.xlsx"]
    assert args.query == ["Total revenue?", "Largest cost?"]
    assert args.embed is False
    assert args.artifacts == []
    assert args.sheet == []
    assert args.llm == "alternate_answer"
    assert args.vlm == "alternate_layout"
    assert args.embedding == "alternate_embedding"
    assert args.max_workers == 6
    assert args.delete_job == []
    assert args.delete_all_jobs is False


def test_cli_parser_accepts_embed_and_sheet_flags():
    args = cli.build_parser().parse_args(
        [
            "--stage",
            "structure",
            "--workbook",
            "book.xlsx",
            "--embed",
            "--sheet",
            "Summary,Detail",
            "--sheet",
            "Archive",
        ]
    )

    assert args.embed is True
    assert args.sheet == ["Summary,Detail", "Archive"]


def test_cli_parser_accepts_repeatable_artifact_files():
    args = cli.build_parser().parse_args(
        [
            "--stage",
            "qa",
            "--workbook",
            "book.xlsx",
            "--query",
            "What is the answer?",
            "--artifacts",
            "run.json",
            "--artifacts",
            "other.pkl",
        ]
    )

    assert args.artifacts == ["run.json", "other.pkl"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--stage", "structure", "--artifacts", "run.json"],
        ["--stage", "qa", "--query", "one", "--query", "two", "--artifacts", "run.json"],
        ["--stage", "qa", "--query", "one", "--embed", "--artifacts", "run.json"],
    ],
)
def test_cli_rejects_invalid_artifact_flag_combinations(arguments):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--workbook", "book.xlsx", *arguments])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("flag", ["--schema", "--metadata"])
def test_cli_rejects_removed_output_flags(flag):
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["--stage", "structure", "--workbook", "book.xlsx", flag])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("stage", ["qa", "all"])
def test_cli_requires_query_for_answering_stages(stage):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--stage", stage, "--workbook", "book.xlsx"])

    assert exc_info.value.code == 2


def test_cli_rejects_removed_force_flag():
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["--stage", "structure", "--workbook", "book.xlsx", "--force"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_cli_rejects_invalid_worker_counts(value):
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(
            ["--stage", "structure", "--workbook", "book.xlsx", "--workers", value]
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
            "-em",
            "alternate_embedding",
        ]
    )

    assert result == 0
    assert captured == {
        "config": "private.yaml",
        "llm_profile": "alternate_answer",
        "vlm_profile": "alternate_layout",
        "embedding_profile": "alternate_embedding",
        "stage": "structure",
        "workbooks": ["sales.xlsx", "costs.xlsx"],
        "queries": [],
        "embed": False,
        "sheets": [],
    }
    assert json.loads(capsys.readouterr().out) == {"job_id": "job-one", "stage": "structure"}


def test_cli_passes_worker_override_to_service(monkeypatch, capsys):
    captured = {}

    class FakeTableAgentService:
        @staticmethod
        def from_config(path, **kwargs):
            return FakeTableAgentService()

        def run(self, **kwargs):
            captured.update(kwargs)
            return {"stage": kwargs["stage"]}

    monkeypatch.setattr(cli, "TableAgentService", FakeTableAgentService)

    result = cli.main(
        [
            "--stage",
            "structure",
            "--workbook",
            "book.xlsx",
            "--max-workers",
            "12",
        ]
    )

    assert result == 0
    assert captured["max_workers"] == 12
    assert json.loads(capsys.readouterr().out) == {"stage": "structure"}


def test_cli_routes_artifacts_to_indexed_qa_and_loads_run_json(tmp_path, monkeypatch, capsys):
    artifact_path = tmp_path / "run.json"
    artifact_path.write_text(
        json.dumps(
            {
                "retrieval_records": [
                    {
                        "id": "book:Summary:table1",
                        "upload_name": "book.xlsx",
                        "sheet": "Summary",
                        "retrieval_card": "Revenue summary",
                        "structure_yaml": "table1: {}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeTableAgentService:
        @staticmethod
        def from_config(path, **kwargs):
            return FakeTableAgentService()

        def run_indexed_qa(self, **kwargs):
            captured.update(kwargs)
            return {"stage": "qa", "answer": "indexed"}

    monkeypatch.setattr(cli, "TableAgentService", FakeTableAgentService)

    result = cli.main(
        [
            "--stage",
            "qa",
            "--workbook",
            "book.xlsx",
            "--query",
            "What is the answer?",
            "--artifacts",
            str(artifact_path),
        ]
    )

    assert result == 0
    assert captured["query"] == "What is the answer?"
    assert captured["workbooks"] == ["book.xlsx"]
    assert captured["artifacts"][0]["id"] == "book:Summary:table1"
    assert json.loads(capsys.readouterr().out) == {"stage": "qa", "answer": "indexed"}


def test_load_artifacts_accepts_jsonl_and_pickle(tmp_path):
    record = {
        "id": "book:Summary:table1",
        "upload_name": "book.xlsx",
        "sheet": "Summary",
        "retrieval_card": "Revenue summary",
    }
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    pickle_path = tmp_path / "records.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump([record], handle)

    loaded = cli.load_artifacts([str(jsonl_path), str(pickle_path)])

    assert loaded == [record]


def test_cli_deletes_selected_jobs_without_requiring_workbooks(monkeypatch, capsys):
    captured = {}

    class FakeTableAgentService:
        @staticmethod
        def from_config(path, **kwargs):
            captured["config"] = path
            return FakeTableAgentService()

        def delete_runs(self, run_ids, *, all_runs):
            captured["run_ids"] = run_ids
            captured["all_runs"] = all_runs
            return {"deleted": list(run_ids), "missing": []}

    monkeypatch.setattr(cli, "TableAgentService", FakeTableAgentService)

    result = cli.main(["--config", "private.yaml", "--delete-job", "run-one", "--delete-job", "run-two"])

    assert result == 0
    assert captured == {
        "config": "private.yaml",
        "run_ids": ["run-one", "run-two"],
        "all_runs": False,
    }
    assert json.loads(capsys.readouterr().out)["deleted"] == ["run-one", "run-two"]


def test_cli_deletes_all_jobs(monkeypatch, capsys):
    class FakeTableAgentService:
        @staticmethod
        def from_config(path, **kwargs):
            return FakeTableAgentService()

        def delete_runs(self, run_ids, *, all_runs):
            assert run_ids == []
            assert all_runs is True
            return {"deleted": ["run-one"], "missing": []}

    monkeypatch.setattr(cli, "TableAgentService", FakeTableAgentService)

    assert cli.main(["--delete-all-jobs"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] == ["run-one"]


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
