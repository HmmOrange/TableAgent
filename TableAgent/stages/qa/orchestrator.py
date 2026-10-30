"""The Orchestrator: answers one question over a workbook by reasoning over a ledger of evidence.

Its three abstract reasoning steps are made by `Reasoner`:
1. reason_start: before anything is read, what the question asks (the answer slots), which needs must be read
   from the workbook, and a hypothesis of how they combine.
2. reason_step, after each inspection: is there enough information? It rules on every check that refuted a
   need (keep the evidence or reject it), then chooses one action, usually an inspection of one need by an
   Inspector (inspector.py). When every need is found, the Synthesizer (synthesizer.py) computes one part
   per answer slot.
3. reason_end: does the answer respond to the question? It may send the loop back to step 2 once.

Settings: `checks` are the Checkers that test each found need (checker.py, none by default), and
`synthesis_check` has a Checker review how the answer was computed before reason_end. A `vision_client`
gives the Inspector `sheet.look` and lets `sheet.headers` ask about an image when its walk is unsure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openpyxl
import pandas as pd

from TableAgent.stages.qa.checker import Checker, check_method
from TableAgent.stages.qa.inspector import Inspector
from TableAgent.stages.qa.ledger import (
    NEED_ID,
    MAX_LLM_CALLS,
    MAX_RETURNS,
    MAX_STEPS,
    STRATEGIES,
    Ledger,
    Step,
    slots_problem,
)
from TableAgent.stages.qa.prompts import (
    ANSWER_FORM_RULES,
    INSPECTOR_SYSTEM_PROMPT,
    INSPECTOR_SYSTEM_PROMPT_WITH_LOOK,
    METHOD_CHECK_TEXT,
    REASON_END_PROMPT,
    REASON_START_PROMPT,
    REASON_STEP_PROMPT,
    REASONER_SYSTEM_PROMPT,
    RULINGS_TEXT,
)
from TableAgent.stages.qa.synthesizer import SynthesisError, Synthesizer
from TableAgent.stages.qa.tools.look import Vision
from TableAgent.stages.qa.tools.notebook import Notebook
from TableAgent.stages.qa.tools.sheet import SheetTools, load_structure
from TableAgent.stages.qa.utils.llm import CountingLLM, ReplyError, ask_json, require_dict
from TableAgent.stages.qa.utils.trace import RunLog, run_dir_name, write_artifacts
from TableAgent.stages.qa.utils.workbook_view import structure_map, workbook_preview

DECISION_KEYS = ("assessment", "rulings", "action", "reason")


@dataclass
class AnswerResult:
    question: str
    answer: str | None
    success: bool
    error: str | None
    seconds: float
    token_usage: dict[str, int]
    llm_calls: list[dict[str, Any]]
    ledger: Ledger
    artifacts: dict[str, str] = field(default_factory=dict)


def answer_question(
    llm_client: Any,
    question: str,
    workbook_path: Path,
    structure_paths: list[Path],
    artifact_dir: Path,
    excluded_sheet_names: set[str] | None = None,
    progress: Callable[[str], None] | None = None,
    checks: tuple[str, ...] = (),
    sees_value: bool = True,
    view: str = "grid",
    synthesis_check: bool = False,
    vision_client: Any = None,
    renderer: Any = None,
) -> AnswerResult:
    """Answer `question`; the artifacts go to a new directory under `artifact_dir`.

    `checks`, `sees_value` and `view` configure the Checkers (checker.py). `vision_client`, with the
    workbook `renderer` that draws cell ranges, turns on looking at images.
    """
    started = time.perf_counter()
    run_dir = artifact_dir / run_dir_name(question)
    log = RunLog(progress)
    llm = CountingLLM(llm_client, log, vision_client)
    log.event("question", question=question, workbook=str(workbook_path))
    vision = Vision(llm, renderer, Path(workbook_path), run_dir / "images") if vision_client is not None else None
    workbook = openpyxl.load_workbook(workbook_path, data_only=True)
    try:
        run = Orchestrator(
            question, llm, workbook, Path(workbook_path), structure_paths, excluded_sheet_names,
            vision, checks, sees_value, view, synthesis_check,
        )
        error = run.answer()
    finally:
        workbook.close()
    answers = [item for item in run.ledger.answers if "answer" in item]
    result = AnswerResult(
        question=question,
        answer=answers[-1]["answer"] if error is None else None,
        success=error is None,
        error=error,
        seconds=time.perf_counter() - started,
        token_usage=llm.token_usage(),
        llm_calls=llm.calls,
        ledger=run.ledger,
    )
    log.event("result", answer=result.answer, success=result.success, error=error)
    cells = run.notebook.cells + (run.checker.cells if run.checker else [])
    result.artifacts = write_artifacts(run_dir, result, run.ledger, log, cells)
    log.say(f"done | success={result.success} | {result.artifacts['trace_md']}")
    return result


class Orchestrator:
    """Holds the ledger of one question and runs the loop; its reasoning calls are made by `Reasoner`."""

    def __init__(
        self,
        question: str,
        llm: CountingLLM,
        workbook: Any,
        workbook_path: Path,
        structure_paths: list[Path],
        excluded_sheet_names: set[str] | None,
        vision: Vision | None,
        checks: tuple[str, ...],
        sees_value: bool,
        view: str,
        synthesis_check: bool,
    ):
        self.question = question
        self.llm = llm
        self.log = llm.log
        self.ledger = Ledger(question)
        structure = load_structure(structure_paths)
        tools = SheetTools(workbook, structure, look=vision.look if vision else None, formula_path=workbook_path)
        self.notebook = Notebook({"sheet": tools, "pd": pd, "np": np})
        self.structure = structure_map(structure)
        self.preview = workbook_preview(workbook, workbook_path.name, excluded_sheet_names, question=question)
        self.reasoner = Reasoner(llm)
        self.inspector = Inspector(
            llm, self.notebook, tools, self.structure, INSPECTOR_SYSTEM_PROMPT_WITH_LOOK if vision else INSPECTOR_SYSTEM_PROMPT,
        )
        self.checker = Checker(llm, tools, checks, sees_value, view, self.preview) if checks else None
        self.synthesis_check = synthesis_check
        self.synthesizer = Synthesizer(llm, self.notebook)

    def answer(self) -> str | None:
        """Run the loop; returns an error message, or None once an answer is set."""
        ledger = self.ledger
        try:
            ledger.start(self.reasoner.reason_start(self.question, self.preview, self.structure))
        except ReplyError as exc:
            return f"Framing the question failed: {exc}"
        self.log.say(f"framed | {len(ledger.needs)} need(s), {len(ledger.answer_slots)} answer slot(s)")
        feedback = ""
        while True:
            chose_synthesis = self._explore(feedback)
            if ledger.open_needs() and not chose_synthesis:
                return f"Out of budget with open needs: {', '.join(need.id for need in ledger.open_needs())}."
            if not ledger.found_needs():
                return "No need was found."
            step = len(ledger.decisions)
            try:
                parts, answer, code = self.synthesizer.compose(ledger, feedback)
            except SynthesisError as exc:
                ledger.answers.append({"step": step, "error": str(exc)})
                return f"Synthesis failed: {exc}"
            composed = {"step": step, "parts": parts, "answer": answer, "code": code}
            ledger.answers.append(composed)
            self.log.say(f"synthesized | {answer}")
            if not chose_synthesis or ledger.returns >= MAX_RETURNS or self.llm.used >= MAX_LLM_CALLS:
                return None
            if self.synthesis_check:
                composed["method_check"] = check_method(self.llm, ledger, composed)
                self.log.say(f"method check | {composed['method_check']['verdict']}")
            review = self._reason_end(parts, answer, composed.get("method_check"))
            if review["verdict"] == "accept":
                return None
            feedback = (
                f"The review returned the answer {answer!r}.\nProblem: {review.get('problem', '')}\n"
                f"Instruction: {review.get('instruction', '')}"
            )

    def _explore(self, feedback: str) -> bool:
        """Steps until one chooses synthesis (True) or the budget runs out (False)."""
        ledger = self.ledger
        while len(ledger.decisions) < MAX_STEPS and self.llm.used < MAX_LLM_CALLS:
            index = len(ledger.decisions) + 1
            try:
                decision = self.reasoner.reason_step(ledger, index, feedback)
                refusal = ""
            except ReplyError as exc:
                decision, refusal = {}, f"The step reply was unusable: {exc}"
            action = decision.get("action")
            args = {key: value for key, value in decision.items() if key not in DECISION_KEYS}
            rulings = ledger.rule(decision.get("rulings"), index)
            refusal = refusal or ledger.refusal(action, args)
            step = Step(
                index, "step", str(decision.get("assessment") or ""), str(action), args,
                accepted=not refusal, refusal=refusal, reason=str(decision.get("reason") or ""), rulings=rulings,
            )
            ledger.steps.append(step)
            self.log.event("step", index=index, action=step.action, args=args, refusal=refusal)
            self.log.say(f"step {index}/{MAX_STEPS} | {step.action}" + (f" refused: {refusal}" if refusal else ""))
            if refusal:
                feedback = f"Your previous action ({step.action}) was refused: {refusal}"
                continue
            feedback = ""
            if action == "synthesize":
                return True
            if action == "inspect":
                need = ledger.needs[str(args["need_id"])]
                evidence = self.inspector.inspect(
                    self.question, ledger, need, args["strategy"], str(args.get("instruction") or ""), index,
                )
                self.log.say(f"{need.id} attempt {evidence.attempt} | " + ("found" if evidence.found else "not found"))
                if self.checker and evidence.found:
                    evidence.checks = self.checker.check(self.question, ledger, need, evidence)
                    verdicts = ", ".join(f"{check['by']} {check['verdict']}" for check in evidence.checks)
                    self.log.say(f"{need.id} attempt {evidence.attempt} | checks: {verdicts or 'none'}")
                ledger.record(evidence)
            else:
                ledger.apply(action, args, index, step.reason)
        self.log.event("budget_used_up", steps=len(ledger.decisions), llm_calls=self.llm.used)
        return False

    def _reason_end(self, parts: list[str], answer: str, method_check: dict[str, str] | None) -> dict[str, Any]:
        try:
            review = self.reasoner.reason_end(self.ledger, answer, parts, method_check)
        except ReplyError as exc:
            review = {"verdict": "accept", "assessment": f"The review was unavailable: {exc}"}
        args = {key: review.get(key, "") for key in ("verdict", "problem", "instruction")}
        self.ledger.steps.append(Step(len(self.ledger.decisions), "end", str(review.get("assessment") or ""), "review", args))
        self.log.say(f"review | {review['verdict']}")
        return review


class Reasoner:
    """The Orchestrator's reasoning calls; each sees the ledger, never an Inspector's transcript."""

    def __init__(self, llm: CountingLLM):
        self.llm = llm

    def reason_start(self, question: str, preview: str, structure: str) -> dict[str, Any]:
        """Before anything is read: what the question asks, the answer slots, the needs to read, a hypothesis."""
        prompt = REASON_START_PROMPT.format(
            question=question, preview=preview, structure=structure, answer_form_rules=ANSWER_FORM_RULES,
        )
        return ask_json(self.llm, "reason_start", REASONER_SYSTEM_PROMPT, prompt, _check_start)

    def reason_step(self, ledger: Ledger, step: int, feedback: str) -> dict[str, Any]:
        """After each inspection: judge the evidence and choose the next action."""
        open_needs = ledger.open_needs()
        prompt = REASON_STEP_PROMPT.format(
            open_needs=(
                f"Open needs: {', '.join(need.id for need in open_needs)}. "
                "synthesize is not allowed until they are found or dropped.\n\n"
                if open_needs else ""
            ),
            question=ledger.question,
            hypothesis=ledger.hypothesis,
            answer_form=ledger.answer_form,
            slots=ledger.slots_text(),
            needs=ledger.needs_text(),
            history=ledger.history_text(),
            step=step,
            max_steps=MAX_STEPS,
            calls=self.llm.used,
            max_calls=MAX_LLM_CALLS,
            feedback=f"\n{feedback}\n" if feedback else "",
            rulings=RULINGS_TEXT.format(needs=", ".join(disputed)) if (disputed := ledger.disputed_needs()) else "",
            strategies=" / ".join(STRATEGIES),
        )
        return ask_json(self.llm, "reason_step", REASONER_SYSTEM_PROMPT, prompt, require_dict)

    def reason_end(
        self, ledger: Ledger, answer: str, parts: list[str], method_check: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """After synthesis: does the answer respond to the question, slot by slot, from the evidence?"""
        prompt = REASON_END_PROMPT.format(
            question=ledger.question,
            hypothesis=ledger.hypothesis,
            answer_form=ledger.answer_form,
            slots=ledger.slots_text(parts),
            needs=ledger.needs_text(),
            answer=answer,
            method_check=METHOD_CHECK_TEXT.format(**method_check) if method_check else "",
        )
        return ask_json(self.llm, "reason_end", REASONER_SYSTEM_PROMPT, prompt, _check_end)


def _check_start(data: Any) -> dict[str, Any]:
    data = require_dict(data)
    for key in ("analysis", "hypothesis", "answer_form"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f'"{key}" must be a non-empty string')
    problem = slots_problem(data.get("answer_slots"))
    if problem:
        raise ValueError(problem)
    needs = data.get("needs")
    if not isinstance(needs, list) or not needs:
        raise ValueError('"needs" must be a non-empty list')
    ids = [str(need.get("id")) if isinstance(need, dict) else "" for need in needs]
    if any(not NEED_ID.fullmatch(need_id) for need_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("need ids must be unique, made of letters, digits and underscores")
    if any(not isinstance(need.get("description"), str) or not need["description"].strip() for need in needs):
        raise ValueError("each need needs a non-empty description")
    data["needs"] = [{"id": need_id, "description": need["description"].strip()} for need_id, need in zip(ids, needs)]
    return data


def _check_end(data: Any) -> dict[str, Any]:
    data = require_dict(data)
    if data.get("verdict") not in ("accept", "return"):
        raise ValueError('"verdict" must be "accept" or "return"')
    return data
