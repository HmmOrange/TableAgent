from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from TableAgent.schema import EvalSample
from TableAgent.pipeline.common import SourceCandidate
from TableAgent.pipeline.siflex_formatter import SiflexAnswerFormatterAgent
from TableAgent.pipeline.table_agent_pipeline import TableAgentPipeline
from TableAgent.configs import TableAgentConfig
from TableAgent.llm import LLMResponse


class FormatterLLM:
    model_name = "formatter-test"
    temperature = 0.0

    def __init__(self, content: str):
        self.content = content
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.calls.append((prompt, system_prompt))
        return LLMResponse(content=self.content, prompt_tokens=7, completion_tokens=3)


@pytest.fixture(autouse=True)
def resolved_table_agent_config(monkeypatch, tmp_path):
    from TableAgent.configs import load_config
    real_from_config = TableAgentConfig.from_config

    def resolve(config=None):
        merged = dict(load_config("config.example.yaml")["table_agent"])
        explicit = config or {}
        if "table_agent" in explicit:
            explicit = explicit["table_agent"]
        merged.update(explicit)
        merged["structure_cache_dir"] = str(tmp_path / "structure-cache")
        return real_from_config(merged)

    monkeypatch.setattr(TableAgentConfig, "from_config", staticmethod(resolve))


def _sample(*, sample_path: str, answer_type: str) -> EvalSample:
    return EvalSample(
        index=0,
        sample_id="case-1",
        table_id="table-1",
        table_content="",
        question="Return the matching records.",
        answer=["expected"],
        sample_path=sample_path,
        raw={"answer_type": answer_type},
    )


def test_formatter_agent_reformats_without_code_fence():
    llm = FormatterLLM("```markdown\n| Name | Value |\n| --- | --- |\n| A | 1 |\n```")
    agent = SiflexAnswerFormatterAgent(llm)

    result = agent.run(
        question="Return the value.",
        answer_type="table",
        draft_answer="Name: A, Value: 1",
    )

    assert result.answer == "| Name | Value |\n| --- | --- |\n| A | 1 |"
    assert result.fallback_used is False
    assert "Required SIFLEX answer type: table" in llm.calls[0][0]
    assert "Do not calculate, infer, correct" in (llm.calls[0][1] or "")


def test_formatter_agent_falls_back_on_model_error():
    llm = FormatterLLM("ERROR: formatter unavailable")
    result = SiflexAnswerFormatterAgent(llm).run(
        question="Return the value.",
        answer_type="list",
        draft_answer="A; B",
    )

    assert result.answer == "A; B"
    assert result.fallback_used is True


def test_formatter_agent_falls_back_when_model_raises():
    class RaisingLLM(FormatterLLM):
        def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
            raise RuntimeError("offline")

    result = SiflexAnswerFormatterAgent(RaisingLLM("unused")).run(
        question="Return the value.",
        answer_type="form",
        draft_answer="Field: value",
    )

    assert result.answer == "Field: value"
    assert result.fallback_used is True


def test_pipeline_formatter_hook_is_siflex_only(tmp_path):
    llm = FormatterLLM("- A\n- B")
    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=llm,
        config={"artifact_dir": str(tmp_path)},
    )

    responses: list[LLMResponse] = []
    formatted = pipeline._format_siflex_answer(
        _sample(sample_path="data/SiFlex/case.json", answer_type="list"),
        "A; B",
        responses,
    )

    assert formatted == "- A\n- B"
    assert len(llm.calls) == 1
    assert responses == [LLMResponse(content="- A\n- B", prompt_tokens=7, completion_tokens=3)]

    unchanged = pipeline._format_siflex_answer(
        _sample(sample_path="data/hitab/case.json", answer_type="list"),
        "A; B",
        responses,
    )

    assert unchanged == "A; B"
    assert len(llm.calls) == 1


def test_pipeline_formatter_ignores_unknown_siflex_type(tmp_path):
    llm = FormatterLLM("should not be used")
    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=llm,
        config={"artifact_dir": str(tmp_path)},
    )

    answer = pipeline._format_siflex_answer(
        _sample(sample_path="data/SiFlex/case.json", answer_type="scalar"),
        "42",
        [],
    )

    assert answer == "42"
    assert llm.calls == []


def test_prepared_siflex_pipeline_returns_formatter_agent_answer(tmp_path, monkeypatch):
    llm = FormatterLLM("- A\n- B")
    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=llm,
        config={"artifact_dir": str(tmp_path)},
    )
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    workbook_path = tmp_path / "book.xlsx"
    workbook_path.write_bytes(b"fixture")
    image_path = source_dir / "table.png"
    image_path.write_bytes(b"fixture")
    (source_dir / "structure.yaml").write_text("table1: {}\n", encoding="utf-8")
    candidate = SourceCandidate(
        directory=source_dir,
        workbook_path=workbook_path,
        sheet_name="Sheet1",
        image_path=image_path,
        html_path=None,
        structure_text="table1: {}\n",
        sheet_text="A B",
        score=1.0,
    )

    monkeypatch.setattr(
        pipeline,
        "_run_verified_qa",
        lambda **_kwargs: (
            LLMResponse(content="A; B", prompt_tokens=5, completion_tokens=2),
            {
                "success": True,
                "error": None,
                "token_usage": {"prompt": 5, "completion": 2},
                "artifacts": {},
                "fallback_used": False,
            },
        ),
    )

    output = pipeline._run_prepared_source(
        _sample(sample_path="data/SiFlex/case.json", answer_type="list"),
        candidate,
        [],
        time.perf_counter(),
    )

    assert output.predicted_answer == "- A\n- B"
    assert output.token_usage == {"prompt": 12, "completion": 5}


@pytest.mark.skipif(
    os.environ.get("RUN_SIFLEX_LIVE_FORMATTER_TEST") != "1",
    reason="Set RUN_SIFLEX_LIVE_FORMATTER_TEST=1 to call the configured live LLM.",
)
def test_live_structure_sheet_q1_qa_and_formatter():
    """Run the real SIFLEX Q1 workbook through QA and the formatter agent."""
    from TableAgent.configs import load_config
    from service import create_model_client
    from TableAgent.QA import TableQARunner

    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/SiFlex/golden_tests/compiled/golden_cases.json"
    structure_path = root / "TableAgent/structure_sheet/structure.yaml"
    workbook_path = (
        root
        / "data/SiFlex/golden_tests/data/설비/"
        / "LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26.xlsx"
    )
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))["cases"]
    case = next(
        item
        for item in cases
        if item.get("sheet_name") == "Q1 – TABLE"
        and any("LV01_설비_REPORT 2026년" in path for path in item.get("source_paths", []))
    )
    assert case["answer_type"] == "table"
    assert structure_path.is_file()
    assert workbook_path.is_file()

    config = load_config(root / "config.yaml")
    config["qa_console_progress"] = True
    llm = create_model_client(config, kind="llm", profile="table_agent")
    runner = TableQARunner(
        structure_path=str(structure_path),
        workbook_path=str(workbook_path),
        llm_client=llm,
        config=config,
    )
    qa_result = runner.run(case["question"])
    namespace_answer = runner.env.execution_namespace.get("final_answer")
    draft_answer = qa_result.final_answer
    if draft_answer is None and namespace_answer is not None:
        draft_answer = str(namespace_answer)
    if draft_answer is None:
        draft_answer = f"QA failed before producing an answer: {qa_result.error}"

    print(f"\n[SIFLEX Q1 QA success] {qa_result.success}")
    print(f"[SIFLEX Q1 QA error] {qa_result.error}")
    print("[SIFLEX Q1 QA draft]\n" + draft_answer)

    formatted = SiflexAnswerFormatterAgent(llm).run(
        question=case["question"],
        answer_type=case["answer_type"],
        draft_answer=draft_answer,
    )
    print(f"\n[SIFLEX formatter called] True")
    print(f"[SIFLEX formatter fallback used] {formatted.fallback_used}")
    print("[SIFLEX Q1 formatted answer]\n" + formatted.answer)

    # This opt-in test is intentionally diagnostic: live QA is nondeterministic.
    # Its contract is to exercise and display the formatter even when QA fails.
    assert formatted.response is not None
    assert isinstance(formatted.answer, str)
    assert formatted.answer
