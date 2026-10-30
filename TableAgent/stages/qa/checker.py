"""Checkers (setting `qa_checker`): they test the evidence of a found need; the reasoning step judges them.

Checks run in the configured order, and a later one runs only when the earlier ones did not confirm:
- reverse: from the cells back to their meaning. The runner walks from each cell to its headers, labels and
  sections, reads the cells, lists rows of the same section left out and cells shared with other needs;
  one model call compares this with the need.
- independent: re-derives the need in its own notebook with other tools than the Inspector's (view `grid`:
  raw cells and their formats and formulas, without structure.yaml; `image`: images of the sheet; `frame`:
  DataFrames only); `grid` and `frame` start from the workbook preview the first reasoning step saw. Seeing the
  claimed value (`sees_value`), it gives the verdict itself; blind, it reports its own value and the
  runner compares the two.
- debate: argues against the evidence in its own notebook; each objection names the cells that show it.

A refuting check is shown to the next reasoning step, which rules on it: keep the evidence or reject it.
`check_method` (setting `qa_synthesis_check`) instead reviews how the answer was computed from the evidence.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from TableAgent.stages.qa.inspector import OUTPUT_CHARS, Turn, clip, transcript
from TableAgent.stages.qa.ledger import MAX_LLM_CALLS, Evidence, Ledger, Need
from TableAgent.stages.qa.prompts import (
    CHECK_WORK_RULES,
    DEBATE_PROMPT,
    DEBATE_SYSTEM_PROMPT,
    INDEPENDENT_BLIND_FINISH,
    INDEPENDENT_CLAIM,
    INDEPENDENT_PROMPT,
    INDEPENDENT_SEES_FINISH,
    INDEPENDENT_SYSTEM_PROMPT,
    METHOD_PROMPT,
    METHOD_SYSTEM_PROMPT,
    NOTEBOOK_IMPORTS,
    REVERSE_PROMPT,
    REVERSE_SYSTEM_PROMPT,
    VIEW_TOOLS,
)
from TableAgent.stages.qa.tools.headers import headers_line, text_of
from TableAgent.stages.qa.tools.notebook import CellRun, Notebook
from TableAgent.stages.qa.tools.sheet import SheetTools
from TableAgent.stages.qa.utils.llm import CountingLLM, ReplyError, ask_json, require_dict
from TableAgent.utils.excel_utils import cell_to_a1, parse_a1_range

VIEWS = {"grid": ("sheets", "find", "read", "around", "frame", "formats"), "image": ("sheets", "find", "look"), "frame": ()}
PREVIEW_VIEWS = ("grid", "frame")  # the image view reads no values, so it gets no preview of them
MAX_TURNS = 4  # of a check in a notebook; the last turn gives the verdict
MAX_CHECKED_CELLS = 5000  # cells read back; a set of rows can be long
MAX_DESCRIBED_CELLS = 12
MAX_DESCRIPTIONS = 6  # structure.yaml descriptions of the entries holding the cells
MAX_LISTED = 10
MAX_OBJECTIONS = 3
CLAIM_KEYS = ("value", "cells", "where", "note")
CLAIM_VALUE_CHARS = 2000
LAST_VERDICT = ' This is the last turn: reply with the JSON object only, with "done": true, "verdict", "reason" and "cells"; write no code.'
LAST_FOUND = ' This is the last turn: set `found` from what you have seen (value None if nothing fits) and set "done" to true.'
LAST_OBJECTIONS = ' This is the last turn: reply with the JSON object only, with "done": true and "objections"; write no code.'


class Checker:
    def __init__(
        self, llm: CountingLLM, tools: SheetTools, checks: tuple[str, ...], sees_value: bool, view: str, preview: str,
    ):
        self.llm = llm
        self.tools = tools
        self.checks = checks
        self.sees_value = sees_value
        self.view = view
        self.preview = preview  # the A1-annotated workbook preview, for the independent check
        self.cells: list[CellRun] = []  # every check's notebook cells, for the artifacts

    def check(self, question: str, ledger: Ledger, need: Need, evidence: Evidence) -> list[dict[str, Any]]:
        """The results of the checks, in order, stopping at the first that confirms."""
        results = []
        for kind in self.checks:
            if self.llm.used >= MAX_LLM_CALLS:
                break
            if kind == "reverse":
                result = self._reverse(question, ledger, need, evidence)
            elif kind == "independent":
                result = self._independent(question, need, evidence)
            else:
                result = self._debate(question, need, evidence)
            self.llm.log.event("check", need_id=need.id, attempt=evidence.attempt, **result)
            results.append(result)
            if result["verdict"] == "confirmed":
                break
        return results

    # ------------------------------------------------------------------------------------------- reverse

    def _reverse(self, question: str, ledger: Ledger, need: Need, evidence: Evidence) -> dict[str, Any]:
        facts = reverse_facts(self.tools, ledger, evidence)
        prompt = REVERSE_PROMPT.format(
            question=question,
            need_id=need.id,
            need=need.description,
            value=_clip(_json(evidence.value), CLAIM_VALUE_CHARS),
            cells=", ".join(evidence.cells),
            where=evidence.where or "(not given)",
            note=evidence.note or "(none)",
            facts="\n".join(facts),
        )
        try:
            reply = ask_json(self.llm, "check_reverse", REVERSE_SYSTEM_PROMPT, prompt, _check_verdict)
        except ReplyError as exc:
            return {"by": "reverse", "verdict": "unsure", "reason": str(exc), "cells": [], "facts": facts}
        return {"by": "reverse", **reply, "facts": facts}

    # --------------------------------------------------------------------------------------- independent

    def _independent(self, question: str, need: Need, evidence: Evidence) -> dict[str, Any]:
        claim = {key: getattr(evidence, key) for key in CLAIM_KEYS}
        namespace: dict[str, Any] = {"pd": pd, "np": np}
        if self.view == "frame":
            namespace["tables"] = {ws.title: self.tools.frame(ws.title) for ws in self.tools.workbook.worksheets}
        else:
            namespace["sheet"] = Restricted(self.tools, VIEWS[self.view])
        if self.sees_value:
            namespace["claim"] = claim
        system_prompt = INDEPENDENT_SYSTEM_PROMPT.format(
            tools=VIEW_TOOLS[self.view],
            claim=INDEPENDENT_CLAIM if self.sees_value else "",
            imports=NOTEBOOK_IMPORTS,
            own_cells="; do not reuse the Inspector's cells as given" if self.sees_value else "",
            finish=INDEPENDENT_SEES_FINISH if self.sees_value else INDEPENDENT_BLIND_FINISH,
            work_rules=CHECK_WORK_RULES,
        )
        claim_text = f"\nThe Inspector's evidence (also in `claim`):\n{_json(claim)}\n" if self.sees_value else ""
        preview = f"\nWorkbook preview (A1-annotated rows; it may be partial):\n{self.preview}\n" if self.view in PREVIEW_VIEWS else ""

        def prompt(turns: list[Turn], number: int, last: str) -> str:
            return INDEPENDENT_PROMPT.format(
                question=question, need_id=need.id, need=need.description, preview=preview, claim=claim_text,
                transcript=transcript(turns, "This check so far"), turn=number, max_turns=MAX_TURNS, last=last,
            )

        if self.sees_value:
            reply, turns, _ = self._notebook_check(
                "independent", system_prompt, prompt, namespace, need, evidence, _check_reply, _check_verdict, LAST_VERDICT,
            )
            if reply is None:
                return _unsure("independent", turns)
            return {"by": "independent", **{key: reply[key] for key in ("verdict", "reason", "cells")}, "turns": turns}
        reply, turns, notebook = self._notebook_check(
            "independent", system_prompt, prompt, namespace, need, evidence, _check_turn, _check_done, LAST_FOUND,
        )
        found = notebook.namespace.get("found")
        if reply is None or not isinstance(found, dict) or found.get("value") in (None, "", [], {}):
            return _unsure("independent", turns, "The Checker did not find the need itself.")
        own, cells = found["value"], [str(cell) for cell in found.get("cells") or []]
        same = same_values(evidence.value, own)
        reason = f"the Checker read {_clip(_json(own), 300)} from {', '.join(cells) or '(no cells)'}"
        if reply.get("reason"):
            reason += f" ({_clip(reply['reason'], 300)})"
        return {"by": "independent", "verdict": "confirmed" if same else "refuted", "reason": reason, "cells": cells, "turns": turns}

    # -------------------------------------------------------------------------------------------- debate

    def _debate(self, question: str, need: Need, evidence: Evidence) -> dict[str, Any]:
        claim = {key: getattr(evidence, key) for key in CLAIM_KEYS}
        namespace = {"sheet": self.tools, "pd": pd, "np": np, "claim": claim}

        def prompt(turns: list[Turn], number: int, last: str) -> str:
            return DEBATE_PROMPT.format(
                question=question, need_id=need.id, need=need.description, claim=_json(claim),
                transcript=transcript(turns, "This check so far"), turn=number, max_turns=MAX_TURNS, last=last,
            )

        reply, turns, _ = self._notebook_check(
            "debate", DEBATE_SYSTEM_PROMPT, prompt, namespace, need, evidence, _check_objections, _check_last_objections,
            LAST_OBJECTIONS,
        )
        if reply is None:
            return _unsure("debate", turns)
        objections = []
        for item in reply["objections"]:
            shown = _shown_cells(self.tools, item["cells"])
            if shown:
                objections.append({**item, "shown": shown})
        objections = objections[:MAX_OBJECTIONS]
        if not objections:
            return {"by": "debate", "verdict": "confirmed", "reason": "no objection names cells that show it", "cells": [], "turns": turns}
        return {
            "by": "debate",
            "verdict": "refuted",
            "reason": f"{len(objections)} objection(s)",
            "cells": [cell for item in objections for cell in item["cells"]],
            "objections": objections,
            "turns": turns,
        }

    # ---------------------------------------------------------------------------------------- notebook

    def _notebook_check(
        self, kind: str, system_prompt: str, prompt, namespace: dict[str, Any], need: Need, evidence: Evidence,
        validate, validate_last, last_note: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], Notebook]:
        """Turns of code until a reply says done; the last turn must give the result."""
        notebook = Notebook(namespace)
        turns: list[Turn] = []
        reply = None
        for number in range(1, MAX_TURNS + 1):
            if self.llm.used >= MAX_LLM_CALLS:
                break
            last = number == MAX_TURNS
            try:
                answer = ask_json(
                    self.llm, f"check_{kind}", system_prompt,
                    prompt(turns, number, last_note if last else ""),
                    validate_last if last else validate,
                )
            except ReplyError as exc:
                turns.append(Turn(number, error=str(exc)))
                continue
            turn = Turn(number, answer["thought"], answer.get("code", ""))
            turns.append(turn)
            if turn.code:
                cell = notebook.run(turn.code, f"{need.id}_attempt{evidence.attempt}_{kind}_turn{number}")
                turn.output, turn.error = clip(cell.output, OUTPUT_CHARS), cell.error
            self.llm.log.event("check_turn", need_id=need.id, attempt=evidence.attempt, by=kind, **asdict(turn))
            if answer["done"] and not turn.error:
                reply = answer
                break
        self.cells += notebook.cells
        return reply, [asdict(turn) for turn in turns], notebook


class Restricted:
    """The `sheet` of a check view: only the named SheetTools methods."""

    def __init__(self, tools: SheetTools, names: tuple[str, ...]):
        for name in names:
            setattr(self, name, getattr(tools, name))


def check_method(llm: CountingLLM, ledger: Ledger, answer: dict[str, Any]) -> dict[str, str]:
    """Whether the synthesis code computes what the question asks: {"verdict": "sound" or "flawed", "problem"}."""
    evidence = "\n".join(
        f"- `{need.variable}` ({need.description}): value {_clip(_json(ledger.latest(need.id).value), CLAIM_VALUE_CHARS)}"
        for need in ledger.found_needs()
    )
    prompt = METHOD_PROMPT.format(
        question=ledger.question, answer_form=ledger.answer_form, slots=ledger.slots_text(),
        evidence=evidence, code=answer["code"], answer=answer["answer"],
    )
    try:
        return ask_json(llm, "check_method", METHOD_SYSTEM_PROMPT, prompt, _check_method)
    except ReplyError as exc:
        return {"verdict": "unsure", "problem": str(exc)}


# ------------------------------------------------------------------------------------------- reverse facts

def reverse_facts(tools: SheetTools, ledger: Ledger, evidence: Evidence) -> list[str]:
    """What the runner finds from the evidence cells, as lines: meaning, structure.yaml descriptions, values,
    rows left out, shared cells."""
    default_sheet = tools.workbook.sheetnames[0]
    cells, problems = expand(evidence.cells, default_sheet)
    facts = [f"cell ref problem: {problem}" for problem in problems]
    if not cells:
        return facts + ["no readable cells were given"]
    described = cells[:MAX_DESCRIBED_CELLS]
    facts.append(f"meaning of {'each cell' if len(described) == len(cells) else f'the first {len(described)} of {len(cells)} cells'}:")
    facts += [f"  {headers_line(tools.headers(cell_to_a1(row, col), sheet, look=False))}" for sheet, row, col in described]
    descriptions = {
        f"{entry.kind} {' > '.join(entry.path)}": entry.description
        for sheet, row, col in described
        for entry in tools.structure_entries_at(row, col, sheet)
        if entry.description
    }
    if descriptions:
        facts.append("structure.yaml describes the entries holding these cells:")
        facts += [f"  {name}: {text}" for name, text in list(descriptions.items())[:MAX_DESCRIPTIONS]]
    facts += _value_facts(tools, evidence.value, cells)
    facts += _left_out_facts(tools, cells)
    for need in ledger.found_needs():
        other = ledger.latest(need.id)
        if need.id == evidence.need_id or other is None:
            continue
        common = set(cells) & set(expand(other.cells, default_sheet)[0])
        if common:
            refs = ", ".join(sorted(cell_to_a1(row, col) for _, row, col in common)[:MAX_LISTED])
            facts.append(f"shared with the evidence of {need.id}: {refs} (combining both counts them twice)")
    return facts


def expand(refs: list[str], sheet: str) -> tuple[list[tuple[str, int, int]], list[str]]:
    """(sheet, row, col) of every cell the refs cover (at most MAX_CHECKED_CELLS), and refs that do not parse."""
    cells: list[tuple[str, int, int]] = []
    problems = []
    for ref in refs:
        try:
            cell_range = parse_a1_range(str(ref), sheet)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        for row in range(cell_range.start_row, cell_range.end_row + 1):
            for col in range(cell_range.start_col, cell_range.end_col + 1):
                if len(cells) < MAX_CHECKED_CELLS:
                    cells.append((cell_range.sheet or sheet, row, col))
    return cells, problems


def _value_facts(tools: SheetTools, value: Any, cells: list[tuple[str, int, int]]) -> list[str]:
    """Whether the cells hold the reported values, element by element."""
    reported = flat(value)
    if not isinstance(value, (list, tuple, dict)):
        if len(cells) > 1:
            return [f"one value reported from {len(cells)} cells: it is computed from them, not read"]
    elif len(cells) == MAX_CHECKED_CELLS:
        return [f"more than {MAX_CHECKED_CELLS} cells: the values were not checked"]
    elif len(reported) != len(cells):
        return [f"{len(reported)} value(s) reported but {len(cells)} cell(s) given: the values cannot be paired with cells"]
    differing = [
        f"{cell_to_a1(row, col)} holds {text_of(actual)!r}, reported {text_of(item)!r}"
        for (sheet, row, col), item in zip(cells, reported)
        if not same_value(actual := tools.value(row, col, sheet), item)
    ]
    if not differing:
        return ["the cells hold the reported values"]
    return [f"values differ: {'; '.join(differing[:MAX_LISTED])}" + (f" (+{len(differing) - MAX_LISTED} more)" if len(differing) > MAX_LISTED else "")]


def _left_out_facts(tools: SheetTools, cells: list[tuple[str, int, int]]) -> list[str]:
    """For a set of cells in one column: the filled cells of the same section (or block) that were not taken."""
    if len(cells) < 2 or len({(sheet, col) for sheet, _, col in cells}) != 1:
        return []
    sheet, _, col = cells[0]
    taken = {row for _, row, _ in cells}
    spans = set()
    for row in sorted(row for row in taken if row - 1 not in taken):  # the first row of each run of taken rows
        found = tools.headers(cell_to_a1(row, col), sheet, look=False)
        span = found["sections"][0]["rows"] if found["sections"] else (found["block"] or {}).get("rows")
        if span:
            spans.add(tuple(int(part) for part in span.split("-")))
    left_out = [
        row
        for first, last in sorted(spans)
        for row in range(first, last + 1)
        if row not in taken and tools.value(row, col, sheet) not in (None, "")
    ]
    if not left_out:
        return ["no other filled cell in the column within the sections of the cells"] if spans else []
    shown = [f"  {tools.describe(cell_to_a1(row, col), sheet)}" for row in left_out[:MAX_LISTED]]
    more = f" (showing {MAX_LISTED})" if len(left_out) > MAX_LISTED else ""
    rows = ", ".join(f"{first}-{last}" for first, last in sorted(spans))
    return [f"{len(left_out)} filled cell(s) in the column within rows {rows} were not taken{more}:", *shown]


def _shown_cells(tools: SheetTools, refs: list[str]) -> str:
    """The values of the cells an objection names, e.g. `D50 = 2230`; empty when none can be read."""
    cells, _ = expand(refs, tools.workbook.sheetnames[0])
    return "; ".join(f"{cell_to_a1(row, col)} = {text_of(tools.value(row, col, sheet))}" for sheet, row, col in cells[:5])


# ------------------------------------------------------------------------------------------------ values

def flat(value: Any) -> list[Any]:
    """The scalar elements of a value: a dict gives its values, nested lists are flattened."""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return [item for element in value for item in flat(element)]
    return [value]


def same_values(reported: Any, own: Any) -> bool:
    """Whether two values hold the same elements, in any order (numbers within rounding)."""
    left, right = flat(reported), flat(own)
    if len(left) != len(right):
        return False
    unmatched = list(right)
    for item in left:
        match = next((index for index, other in enumerate(unmatched) if same_value(item, other)), None)
        if match is None:
            return False
        unmatched.pop(match)
    return True


def same_value(actual: Any, reported: Any) -> bool:
    actual_number, reported_number = _number(actual), _number(reported)
    if actual_number is not None and reported_number is not None:
        return math.isclose(actual_number, reported_number, rel_tol=1e-6, abs_tol=1e-9)
    return text_of(actual).casefold() == text_of(reported).casefold()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


# ------------------------------------------------------------------------------------------------ replies

def _check_turn(data: Any) -> dict[str, Any]:
    data = require_dict(data)
    code, done = data.get("code", ""), data.get("done", False)
    if not isinstance(code, str) or not isinstance(done, bool):
        raise ValueError('"code" must be a string and "done" a boolean')
    if not done and not code.strip():
        raise ValueError('"code" must not be empty unless "done" is true')
    return {**data, "thought": str(data.get("thought") or "").strip(), "code": code.strip(), "done": done}


def _check_done(data: Any) -> dict[str, Any]:
    reply = _check_turn(data)
    if not reply["done"]:
        raise ValueError('this is the last turn: set `found` and set "done" to true')
    return reply


def _check_reply(data: Any) -> dict[str, Any]:
    """A turn that either explores (code) or gives the verdict (done)."""
    reply = _check_turn(data)
    return _check_verdict(reply) if reply["done"] else reply


def _check_verdict(data: Any) -> dict[str, Any]:
    """A reply with a verdict; any code in it is not run."""
    data = require_dict(data)
    if data.get("verdict") not in ("confirmed", "refuted"):
        raise ValueError('"verdict" must be "confirmed" or "refuted"')
    cells = data.get("cells") or []
    return {
        "thought": str(data.get("thought") or "").strip(),
        "done": True,
        "verdict": data["verdict"],
        "reason": str(data.get("reason") or "").strip(),
        "cells": [str(cell) for cell in (cells if isinstance(cells, list) else [cells])],
    }


def _check_objections(data: Any) -> dict[str, Any]:
    data = require_dict(data)
    if not data.get("done") and "objections" not in data:
        return _check_turn(data)
    objections = data.get("objections")
    if not isinstance(objections, list) or not all(isinstance(item, dict) for item in objections):
        raise ValueError('"objections" must be a list of {"objection", "cells"} objects')
    return {
        "thought": str(data.get("thought") or "").strip(),
        "done": True,
        "objections": [
            {
                "objection": str(item.get("objection") or "").strip(),
                "cells": [str(cell) for cell in (item.get("cells") if isinstance(item.get("cells"), list) else [])],
            }
            for item in objections
            if str(item.get("objection") or "").strip()
        ],
    }


def _check_last_objections(data: Any) -> dict[str, Any]:
    if "objections" not in require_dict(data):
        raise ValueError('this is the last turn: give "objections" (an empty list when the evidence holds)')
    return _check_objections(data)


def _check_method(data: Any) -> dict[str, Any]:
    data = require_dict(data)
    if data.get("verdict") not in ("sound", "flawed"):
        raise ValueError('"verdict" must be "sound" or "flawed"')
    return {"verdict": data["verdict"], "problem": str(data.get("problem") or "").strip()}


def _unsure(by: str, turns: list[dict[str, Any]], reason: str = "The check ended without a verdict.") -> dict[str, Any]:
    return {"by": by, "verdict": "unsure", "reason": reason, "cells": [], "turns": turns}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
