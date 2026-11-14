"""The QA agent (src/workbook_qa.py): its metadata, its generation cap, its two fallbacks, and its settings."""

import json
from pathlib import Path

import openpyxl
import pytest

import TableAgent.src.workbook_qa as workbook_qa
from TableAgent import QASettings, WorkbookQA
from TableAgent.llm import LLMResponse
from TableAgent.src.ledger import Evidence, Ledger
from TableAgent.src.orchestrator import AnswerResult
from TableAgent.src.prompts import (
    DIRECT_ANSWER_SYSTEM_PROMPT,
    INSPECTOR_SYSTEM_PROMPT,
    INSPECTOR_SYSTEM_PROMPT_WITH_LOOK,
    REASONER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)


class FakeLLM:
    model_name = "fake"
    temperature = 0.0

    def __init__(self):
        self.calls = []
        self.max_tokens = None

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.calls.append((prompt, system_prompt))
        return LLMResponse(content="100", prompt_tokens=6, completion_tokens=1)


class SuccessfulQALLM(FakeLLM):
    """Answers the QA loop's roles: frame, one inspection, synthesis, review."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.calls.append((prompt, system_prompt))
        if system_prompt == REASONER_SYSTEM_PROMPT:
            if "before anything is read" in prompt:
                reply = {
                    "analysis": "One value.", "hypothesis": "Read it.", "answer_form": "a number",
                    "answer_slots": [{"slot": "the revenue", "asked_by": "revenue"}],
                    "needs": [{"id": "n1", "description": "the 2024 revenue"}],
                }
            elif "Proposed answer:" in prompt:
                reply = {"assessment": "Fits.", "verdict": "accept"}
            elif "Open needs:" in prompt:
                reply = {"assessment": "Open.", "action": "inspect", "need_id": "n1", "strategy": "window"}
            else:
                reply = {"assessment": "Found.", "action": "synthesize"}
            return LLMResponse(content=json.dumps(reply), prompt_tokens=5, completion_tokens=1)
        if system_prompt in (INSPECTOR_SYSTEM_PROMPT, INSPECTOR_SYSTEM_PROMPT_WITH_LOOK):
            reply = {"thought": "Read it.", "code": "evidence_n1 = {'value': 100, 'cells': []}", "done": True}
            return LLMResponse(content=json.dumps(reply), prompt_tokens=7, completion_tokens=3)
        if system_prompt == SYNTHESIZER_SYSTEM_PROMPT:
            reply = {"thought": "One slot.", "code": "answer_parts = [evidence_n1['value']]"}
            return LLMResponse(content=json.dumps(reply), prompt_tokens=6, completion_tokens=2)
        raise AssertionError(f"unexpected system prompt: {system_prompt[:60]}")


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    path = tmp_path / "book.xlsx"
    book = openpyxl.Workbook()
    book.active.append(["Year", "Revenue"])
    book.active.append([2024, 100])
    book.save(path)
    return path


def test_a_successful_exploration_reports_its_tokens_and_calls(tmp_path: Path, workbook: Path):
    agent = WorkbookQA(SuccessfulQALLM(), QASettings())
    response, qa = agent.run(
        question="What is the 2024 revenue?", workbook_path=workbook, qa_artifact_dir=tmp_path / "qa",
        fallback_prompt=lambda: pytest.fail("no fallback on success"),
    )

    assert response.content == "100"
    assert qa["success"] is True and qa["fallback_used"] is False and qa["answer_route"] == "workbook"
    # reason_start makes three calls: the plan twice, then the reconciled one.
    assert qa["token_usage"] == {"prompt": 43, "completion": 11} and qa["llm_call_count"] == 8
    assert Path(qa["artifacts"]["trace_md"]).is_file()


@pytest.mark.parametrize("plan_count, plans", [
    (1, ["Plan A"]),
    (2, ["Plan A", "Plan B", "reconciled from Plan A and Plan B"]),
    (3, ["Plan A", "Plan B", "Plan C", "reconciled from Plan A, Plan B and Plan C"]),
])
def test_the_starting_plan_is_written_plan_count_times_and_reconciled_above_one(tmp_path: Path, workbook: Path, plan_count, plans):
    llm = SuccessfulQALLM()
    _, qa = WorkbookQA(llm, QASettings(plan_count=plan_count)).run(
        question="What is the 2024 revenue?", workbook_path=workbook, qa_artifact_dir=tmp_path / "qa",
        fallback_prompt="unused",
    )
    events = Path(qa["artifacts"]["trace_md"]).with_name("events.jsonl").read_text().splitlines()
    assert [json.loads(line)["plan"] for line in events if '"starting_plan"' in line] == plans
    assert qa["llm_call_count"] == 5 + len(plans) and qa["plan_count"] == plan_count
    reconcile = [prompt for prompt, _ in llm.calls if "plans were written independently" in prompt]
    assert len(reconcile) == (plan_count > 1)
    if plan_count == 3:
        assert "Three plans were written" in reconcile[0] and "Plan C:\n" in reconcile[0]


def test_the_generation_cap_is_set_on_both_models_only_when_configured():
    llm, vision = FakeLLM(), FakeLLM()
    llm.max_tokens, vision.max_tokens = 100, 200
    WorkbookQA(llm, QASettings(generation_max_tokens=1024), vision)
    assert (llm.max_tokens, vision.max_tokens) == (1024, 1024)

    llm.max_tokens, vision.max_tokens = 500, 600
    WorkbookQA(llm, QASettings(), vision)
    assert (llm.max_tokens, vision.max_tokens) == (500, 600)


def _failed_exploration(tmp_path: Path, monkeypatch, evidence=None):
    """Run the agent with an exploration that fails, optionally after finding one checked need."""
    ledger = Ledger("What is the value?")
    ledger.start({
        "analysis": "", "hypothesis": "", "answer_form": "", "answer_slots": [],
        "needs": [{"id": "n1", "description": "the 2021 value"}],
    })
    if evidence is not None:
        ledger.record(Evidence("n1", 1, 1, "search_sheet", "", **evidence))
    failed = AnswerResult("What is the value?", None, False, "Synthesis failed: no parts", 1.0,
                          {"prompt": 3, "completion": 1}, [], ledger)
    monkeypatch.setattr(workbook_qa, "answer_question", lambda *args, **kwargs: failed)
    llm = FakeLLM()
    built = []

    def context():
        built.append(1)
        return "Workbook context goes here."

    response, qa = WorkbookQA(llm, QASettings()).run(
        question="What is the value?", workbook_path=tmp_path / "book.xlsx", qa_artifact_dir=tmp_path / "qa",
        fallback_prompt=context,
    )
    assert llm.calls[-1][1] == DIRECT_ANSWER_SYSTEM_PROMPT
    return llm.calls[-1][0], qa, built, response


def test_a_failed_exploration_answers_from_the_checked_needs_it_found(tmp_path: Path, monkeypatch):
    prompt, qa, built, response = _failed_exploration(
        tmp_path, monkeypatch, {"found": True, "value": 42, "cells": ["B2"], "checks": [
            {"by": "reverse", "verdict": "confirmed", "reason": "B2 is the 2021 value", "cells": ["B2"]}]},
    )
    assert qa["fallback_source"] == "found_evidence" and built == []
    assert "- the 2021 value: 42 (cells B2" in prompt
    assert response.prompt_tokens == 6 + 3  # the fallback call plus the exploration


def test_a_failed_exploration_with_only_refuted_needs_falls_back_to_the_given_context(tmp_path: Path, monkeypatch):
    prompt, qa, built, _ = _failed_exploration(
        tmp_path, monkeypatch, {"found": True, "value": 7, "cells": ["C2"], "checks": [
            {"by": "reverse", "verdict": "refuted", "reason": "C2 holds 8", "cells": ["C2"]}]},
    )
    assert qa["fallback_source"] == "source_context" and built == [1]
    assert prompt.startswith("An exploration of the workbook failed") and "Synthesis failed: no parts" in prompt
    assert "Workbook context goes here." in prompt


def test_settings_read_the_config_keys_and_reject_unknown_ones():
    settings = QASettings.from_config({
        "checker": "independent", "checker_view": "frame", "generation_max_tokens": 8192, "image_resolution": 600,
        "llm_provider": "other_model", "plan_count": 3,
    })
    assert QASettings.from_config({}).plan_count == 2 and settings.plan_count == 3
    with pytest.raises(ValueError, match="plan_count must be a positive integer"):
        QASettings.from_config({"plan_count": 0})
    assert settings.checker == ("independent",) and settings.checker_view == "frame"
    assert settings.generation_max_tokens == 8192 and settings.image_resolution == 600
    with pytest.raises(ValueError, match="Unknown table_agent settings: qa_checker"):
        QASettings.from_config({"qa_checker": ["independent"]})
    with pytest.raises(ValueError, match="checker must list distinct checks"):
        QASettings.from_config({"checker": ["independent", "independent"]})
