from __future__ import annotations

from TableAgent.benchmarks.siflex import (
    SIFLEX_PERFECT_EXCLUDED_SAMPLE_IDS,
    SIFLEX_PERFECT_SOURCES,
    SiflexJudge,
    _select_samples,
)
from TableAgent.llm import BaseLLM, LLMResponse
from TableAgent.pipeline.base import PipelineOutput
from TableAgent.schema import EvalSample


class FakeJudgeLLM(BaseLLM):
    def __init__(self, content: str):
        super().__init__(model_name="fake")
        self.content = content

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        return LLMResponse(content=self.content)


def test_siflex_perfect_source_manifest_has_eighteen_cases():
    assert len(SIFLEX_PERFECT_SOURCES) == 18
    assert len(SIFLEX_PERFECT_EXCLUDED_SAMPLE_IDS) == 5
    assert not SIFLEX_PERFECT_EXCLUDED_SAMPLE_IDS.intersection(SIFLEX_PERFECT_SOURCES)


def test_siflex_judge_uses_ise_table_scoring_weights():
    sample = EvalSample(
        index=0,
        sample_id="sample",
        table_id="table.xlsx",
        table_content="",
        question="What is the value?",
        answer=["42"],
        raw={"answer_type": "table"},
    )
    output = PipelineOutput(sample_id="sample", predicted_answer="42")
    judge = SiflexJudge(FakeJudgeLLM(
        '{"factual_correctness": 1, "coverage": 1, "structure_fidelity": 1, '
        '"grounding": 1, "rationale": "correct"}'
    ))

    metrics = judge.evaluate(sample, output)

    assert metrics["pass"] is True
    assert metrics["overall_score"] == 1.0


def test_siflex_judge_rejects_empty_answers():
    sample = EvalSample(
        index=0,
        sample_id="sample",
        table_id="table.xlsx",
        table_content="",
        question="What is the value?",
        answer=["42"],
    )

    metrics = SiflexJudge(FakeJudgeLLM("{}"), pass_threshold=0.8).evaluate(
        sample,
        PipelineOutput(sample_id="sample", predicted_answer=""),
    )

    assert metrics["pass"] is False
    assert metrics["overall_score"] == 0.0


def test_siflex_sample_selector_accepts_unique_substring():
    samples = [
        EvalSample(index=0, sample_id="workbook:Q3 – LIST", table_id="a", table_content="", question="q", answer=["a"]),
        EvalSample(index=1, sample_id="workbook:Q4 – FORM", table_id="a", table_content="", question="q", answer=["a"]),
    ]

    selected = _select_samples(samples, ["Q3 – LIST"])

    assert [sample.sample_id for sample in selected] == ["workbook:Q3 – LIST"]
