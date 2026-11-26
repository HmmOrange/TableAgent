"""Synthesis: compute one answer part per slot, then join them into the answer.

The parts are joined by code, so the answer holds exactly the slots the question asks for. By default the parts are
computed in code from the found evidence (`Synthesizer`). Two ablations change only how the parts are made:
`DirectSynthesizer` (QASettings.synthesis: direct) has the model write them from the evidence without code, and
`WorkbookSynthesizer` (QASettings.exploration: none) has the code read the workbook itself, as nothing was explored.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from TableAgent.src.ledger import MAX_LLM_CALLS, Ledger
from TableAgent.src.utils.llm import CountingLLM, ReplyError, ask_json, require_dict
from TableAgent.src.tools.notebook import Notebook
from TableAgent.src.prompts import (
    ANSWER_FORM_RULES,
    CODE_SUPPORT,
    CODE_SUPPORT_CHECK,
    EVIDENCE_SUPPORT,
    EVIDENCE_SUPPORT_CHECK,
    SYNTHESIZER_DIRECT_SYSTEM_PROMPT,
    SYNTHESIZER_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_WORKBOOK_PROMPT,
    SYNTHESIZER_WORKBOOK_SYSTEM_PROMPT,
)

MAX_ATTEMPTS = 3
EVIDENCE_VALUE_CHARS = 2000
OUTPUT_CHARS = 1500


class SynthesisError(Exception):
    """No attempt produced one answer part per slot."""


class Synthesizer:
    """Computes the parts in code, in a notebook holding the evidence variables."""

    system_prompt = SYNTHESIZER_SYSTEM_PROMPT
    value_chars = EVIDENCE_VALUE_CHARS  # of each evidence value in the prompt; the code reads the value whole

    def __init__(self, llm: CountingLLM, notebook: Notebook):
        self.llm = llm
        self.notebook = notebook

    def compose(self, ledger: Ledger, returned: str = "") -> tuple[list[str], str, str]:
        """The answer parts, the answer joined from them, and the code that computed them (empty without code).

        `returned` says why the review sent back the last answer.
        """
        number = len(ledger.answers) + 1
        for need in ledger.needs.values():
            if need.status != "found":
                # Evidence of a need that is not found must not reach the answer.
                self.notebook.namespace.pop(need.variable, None)
        tried: list[str] = []
        problem = "the model call budget is used up"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self.llm.used >= MAX_LLM_CALLS:
                break
            try:
                reply = ask_json(self.llm, "synthesize", self.system_prompt, self._prompt(ledger, tried, returned), self._check_reply)
            except ReplyError as exc:
                problem = str(exc)
                continue
            parts, output, problem = self._parts(reply, f"synthesis{number}_attempt{attempt}", len(ledger.answer_slots))
            code = reply.get("code", "")
            self.llm.log.event("synthesis_attempt", attempt=attempt, code=code, output=output, problem=problem)
            if not problem:
                parts = [part_text(part) for part in parts]
                return parts, ", ".join(parts), code
            tried.append(f"Attempt {attempt}:\n{self._attempt_text(reply, output)}\nProblem: {problem}")
        raise SynthesisError(problem)

    def support(self, ledger: Ledger, composed: dict[str, Any]) -> dict[str, str]:
        """What the final review is shown of the answer's support, and asked of it (REASON_END_PROMPT)."""
        return {"support": EVIDENCE_SUPPORT.format(needs=ledger.needs_text()), "support_check": EVIDENCE_SUPPORT_CHECK}

    def _prompt(self, ledger: Ledger, tried: list[str], returned: str) -> str:
        evidence = "\n".join(
            f"- `{need.variable}` ({need.description}): value {_json(evidence.value, self.value_chars)}"
            + (f"; note: {evidence.note}" if evidence.note else "")
            for need in ledger.found_needs()
            for evidence in [ledger.latest(need.id)]
        )
        return SYNTHESIZER_PROMPT.format(
            question=ledger.question,
            answer_form=ledger.answer_form,
            slots=ledger.slots_text(),
            evidence=evidence,
            answer_form_rules=ANSWER_FORM_RULES,
            transcript=_transcript(tried, returned),
        )

    def _parts(self, reply: dict[str, Any], label: str, slots: int) -> tuple[Any, str, str]:
        """The parts an attempt gives, what it printed, and its problem (empty when none)."""
        self.notebook.namespace.pop("answer_parts", None)
        cell = self.notebook.run(reply["code"], label)
        parts = self.notebook.namespace.get("answer_parts")
        return parts, cell.output, cell.error or _parts_problem(parts, slots)

    @staticmethod
    def _attempt_text(reply: dict[str, Any], output: str) -> str:
        """An attempt as the next attempt is shown it."""
        return f"Code:\n{reply['code']}\nPrinted: {output[:OUTPUT_CHARS] or '(nothing)'}"

    @staticmethod
    def _check_reply(data: Any) -> dict[str, Any]:
        data = require_dict(data)
        if not isinstance(data.get("code"), str) or not data["code"].strip():
            raise ValueError('"code" must be non-empty Python')
        return data


class DirectSynthesizer(Synthesizer):
    """synthesis: direct: the model writes the parts from the evidence itself, without code, and sees each value whole
    up to `value_chars` (QASettings.max_context_chars)."""

    system_prompt = SYNTHESIZER_DIRECT_SYSTEM_PROMPT

    def __init__(self, llm: CountingLLM, notebook: Notebook, value_chars: int):
        super().__init__(llm, notebook)
        self.value_chars = value_chars

    def _parts(self, reply: dict[str, Any], label: str, slots: int) -> tuple[Any, str, str]:
        parts = reply.get("answer_parts")
        return parts, _json(parts, OUTPUT_CHARS), _parts_problem(parts, slots)

    @staticmethod
    def _attempt_text(reply: dict[str, Any], output: str) -> str:
        return f"answer_parts: {output}"

    @staticmethod
    def _check_reply(data: Any) -> dict[str, Any]:
        return require_dict(data)


class WorkbookSynthesizer(Synthesizer):
    """exploration: none: nothing is read before synthesis; the code reads the workbook (openpyxl) from the plan, and
    the final review is shown that code instead of evidence.

    `workbook` and `legends` are what the starting plans were shown of the workbook.
    """

    system_prompt = SYNTHESIZER_WORKBOOK_SYSTEM_PROMPT

    def __init__(self, llm: CountingLLM, notebook: Notebook, workbook: str, legends: str):
        super().__init__(llm, notebook)
        self.workbook = workbook
        self.legends = legends

    def support(self, ledger: Ledger, composed: dict[str, Any]) -> dict[str, str]:
        printed = self.notebook.cells[-1].output  # the cell that set the parts: the review follows the synthesis
        return {
            "support": CODE_SUPPORT.format(
                needs=_needs_list(ledger), code=composed["code"], printed=printed[:OUTPUT_CHARS] or "(nothing)",
            ),
            "support_check": CODE_SUPPORT_CHECK,
        }

    def _prompt(self, ledger: Ledger, tried: list[str], returned: str) -> str:
        if returned:  # the answer the review returned is the last one, and its code is what to correct
            returned += f"\nIts code:\n```python\n{ledger.answers[-1]['code']}\n```"
        return SYNTHESIZER_WORKBOOK_PROMPT.format(
            question=ledger.question,
            hypothesis=ledger.hypothesis,
            answer_form=ledger.answer_form,
            slots=ledger.slots_text(),
            needs=_needs_list(ledger),
            workbook=self.workbook,
            legends=self.legends,
            answer_form_rules=ANSWER_FORM_RULES,
            transcript=_transcript(tried, returned),
        )


def part_text(part: Any) -> str:
    """How one answer part is written: lists comma-joined, booleans as Yes/No, floats to four decimals without .0."""
    if isinstance(part, np.generic):
        part = part.item()
    if isinstance(part, np.ndarray):
        part = part.tolist()
    if isinstance(part, bool):
        return "Yes" if part else "No"
    if isinstance(part, (list, tuple)):
        return ", ".join(part_text(item) for item in part)
    if isinstance(part, float):
        part = round(part, 4)
        if part.is_integer():
            return str(int(part))
    return "" if part is None else str(part)


def _parts_problem(parts: Any, slots: int) -> str:
    if not isinstance(parts, (list, tuple)):
        return "`answer_parts` is not set to a list."
    if len(parts) != slots:
        return f"`answer_parts` has {len(parts)} entries; it needs exactly {slots}, one per answer slot."
    return ""


def _transcript(tried: list[str], returned: str) -> str:
    """Why the review returned the last answer, and the earlier attempts of this synthesis."""
    return (f"\n{returned}\n" if returned else "") + (("\nEarlier attempts:\n" + "\n\n".join(tried)) if tried else "")


def _needs_list(ledger: Ledger) -> str:
    return "\n".join(f"- {need.id}: {need.description}" for need in ledger.needs.values())


def _json(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"
