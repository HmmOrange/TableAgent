"""Synthesis: compute one answer part per slot from the found evidence, then join them into the answer.

The parts are joined by code, so the answer holds exactly the slots the question asks for.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from TableAgent.stages.qa.ledger import MAX_LLM_CALLS, Ledger
from TableAgent.stages.qa.utils.llm import CountingLLM, ReplyError, ask_json, require_dict
from TableAgent.stages.qa.tools.notebook import Notebook
from TableAgent.stages.qa.prompts import ANSWER_FORM_RULES, SYNTHESIZER_PROMPT, SYNTHESIZER_SYSTEM_PROMPT

MAX_ATTEMPTS = 3
EVIDENCE_VALUE_CHARS = 2000
OUTPUT_CHARS = 1500


class SynthesisError(Exception):
    """No attempt produced one answer part per slot."""


class Synthesizer:
    def __init__(self, llm: CountingLLM, notebook: Notebook):
        self.llm = llm
        self.notebook = notebook

    def compose(self, ledger: Ledger, returned: str = "") -> tuple[list[str], str, str]:
        """The answer parts, the answer joined from them, and the code that computed them.

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
                reply = ask_json(self.llm, "synthesize", SYNTHESIZER_SYSTEM_PROMPT, self._prompt(ledger, tried, returned), _check_reply)
            except ReplyError as exc:
                problem = str(exc)
                continue
            self.notebook.namespace.pop("answer_parts", None)
            cell = self.notebook.run(reply["code"], f"synthesis{number}_attempt{attempt}")
            problem = cell.error or _parts_problem(self.notebook.namespace.get("answer_parts"), len(ledger.answer_slots))
            self.llm.log.event("synthesis_attempt", attempt=attempt, code=reply["code"], output=cell.output, problem=problem)
            if not problem:
                parts = [part_text(part) for part in self.notebook.namespace["answer_parts"]]
                return parts, ", ".join(parts), reply["code"]
            tried.append(
                f"Attempt {attempt}:\nCode:\n{reply['code']}\nPrinted: {cell.output[:OUTPUT_CHARS] or '(nothing)'}\n"
                f"Problem: {problem}"
            )
        raise SynthesisError(problem)

    @staticmethod
    def _prompt(ledger: Ledger, tried: list[str], returned: str) -> str:
        evidence = "\n".join(
            f"- `{need.variable}` ({need.description}): value {_json(evidence.value)}"
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
            transcript=(f"\n{returned}\n" if returned else "") + (("\nEarlier attempts:\n" + "\n\n".join(tried)) if tried else ""),
        )


def part_text(part: Any) -> str:
    """How one answer part is written: lists comma-joined, booleans as Yes/No, floats without binary noise or .0."""
    if isinstance(part, np.generic):
        part = part.item()
    if isinstance(part, np.ndarray):
        part = part.tolist()
    if isinstance(part, bool):
        return "Yes" if part else "No"
    if isinstance(part, (list, tuple)):
        return ", ".join(part_text(item) for item in part)
    if isinstance(part, float):
        part = round(part, 10)
        if part.is_integer():
            return str(int(part))
    return "" if part is None else str(part)


def _parts_problem(parts: Any, slots: int) -> str:
    if not isinstance(parts, (list, tuple)):
        return "`answer_parts` is not set to a list."
    if len(parts) != slots:
        return f"`answer_parts` has {len(parts)} entries; it needs exactly {slots}, one per answer slot."
    return ""


def _check_reply(data: Any) -> dict[str, Any]:
    data = require_dict(data)
    if not isinstance(data.get("code"), str) or not data["code"].strip():
        raise ValueError('"code" must be non-empty Python')
    return data


def _json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= EVIDENCE_VALUE_CHARS else text[:EVIDENCE_VALUE_CHARS] + "…"
