from pathlib import Path

from TableAgent.configs import TableAgentConfig, load_config
from TableAgent.pipeline.retrieval import SourceRetriever
from TableAgent.pipeline.retrieval.scoring import bm25_scores


def _settings(tmp_path: Path, *, mode: str = "lexical") -> TableAgentConfig:
    config = dict(load_config("config.example.yaml")["table_agent"])
    config["artifact_dir"] = str(tmp_path / "artifacts")
    config["source_artifact_dir"] = str(tmp_path / "sources")
    config["routing"] = {
        "retrieval": {
            "mode": mode,
            "rerank_with_llm": False,
            "bm25_weight": 0.3,
            "embedding_weight": 0.7,
            "explicit_workbook_guard": True,
            "explicit_sheet_guard": True,
        },
        "qa": {"mode": "auto"},
    }
    return TableAgentConfig.from_config(config)


def _artifact(identifier: str, sheet: str, card: str, *, score: float | None = None):
    artifact = {
        "id": identifier,
        "workbook": "report.xlsx",
        "sheet": sheet,
        "structure_yaml": "table1:\n  name: Report\n  headers: []",
        "retrieval_card": card,
        "retrieval_type": "data",
        "retrieval_level": "table",
        "table_id": "table1",
    }
    if score is not None:
        artifact["score"] = score
    return artifact


def test_bm25_prefers_rare_exact_terms():
    scores = bm25_scores(
        "compressor ZX-410 failure",
        [
            "general compressor maintenance records",
            "failure record for compressor ZX-410",
        ],
    )

    assert scores[1] > scores[0]


def test_indexed_retrieval_restricts_an_explicit_sheet(tmp_path: Path):
    retriever = SourceRetriever(_settings(tmp_path), None, None, None)

    candidate = retriever.select_indexed(
        question="What is the total on sheet Summary?",
        artifacts=[
            _artifact("detail", "Detail", "total amount detail"),
            _artifact("summary", "Summary", "total amount summary"),
        ],
        workbook_paths={"report.xlsx": tmp_path / "report.xlsx"},
        responses=[],
        fit_context=lambda value: value,
    )

    assert candidate is not None
    assert candidate.sheet_name == "Summary"
    trace = candidate.retrieval_trace[-1]["explicit_sheet_guard"]
    assert trace["applied"] is True
    assert trace["candidate_count_after"] == 1


def test_indexed_mode_reuses_upstream_vector_scores(tmp_path: Path):
    retriever = SourceRetriever(_settings(tmp_path, mode="indexed"), None, None, None)

    candidate = retriever.select_indexed(
        question="Which sheet contains the maintenance schedule?",
        artifacts=[
            _artifact("low", "Data", "maintenance schedule", score=0.2),
            _artifact("high", "Plan", "maintenance schedule", score=0.9),
        ],
        workbook_paths={"report.xlsx": tmp_path / "report.xlsx"},
        responses=[],
        fit_context=lambda value: value,
    )

    assert candidate is not None
    assert candidate.artifact_id == "high"
    assert candidate.embedding_used is True
    assert candidate.embedding_score == 0.9
