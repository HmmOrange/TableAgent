"""One inspection: the Inspector establishes one need by exploring the workbook in code over several turns.

A turn is one model reply (thought, code, done) and one notebook cell. Errors are observations for
the next turn. `done` ends the inspection once the evidence variable is set as the contract asks. When the
turns are used up without evidence, one report turn sets it from what the turns printed. An attempt that
found nothing hands its last thoughts to the next attempt. The notebook holds the formats in use in the
variables `legends` and `formats` (tools/formats.py).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from TableAgent.src.ledger import MAX_LLM_CALLS, Evidence, Ledger, Need, evidence_lines
from TableAgent.src.prompts import (
    FORMATS_VARIABLE_TEXT,
    INSPECTOR_LEGEND_TEXT,
    INSPECTOR_PROMPT,
    INSPECTOR_SYSTEM_PROMPT,
    STRATEGY_TEXTS,
)
from TableAgent.src.tools.formats import format_marks, legend_fills, legend_notes, legends
from TableAgent.src.tools.notebook import Notebook
from TableAgent.src.tools.sheet import SheetTools
from TableAgent.src.utils.llm import CountingLLM, ReplyError, ask_json, require_dict
from TableAgent.utils.excel_utils import parse_a1_range

MAX_TURNS = 12  # the default of QASettings.inspector_max_turns
REPORT_TURNS = 1  # turns after the last one that only report what was already printed
OUTPUT_CHARS = 3000  # of one turn's printed output, in the transcript
TRANSCRIPT_CHARS = 24000  # beyond this the oldest outputs are left out first


@dataclass
class Turn:
    number: int
    thought: str = ""
    code: str = ""
    output: str = ""
    error: str = ""
    runner_note: str = ""  # why the runner did not accept `done`, shown to the next turn


class Inspector:
    def __init__(
        self, llm: CountingLLM, notebook: Notebook, tools: SheetTools,
        system_prompt: str = INSPECTOR_SYSTEM_PROMPT, max_turns: int = MAX_TURNS,
        strategy_texts: dict[str, str] = STRATEGY_TEXTS,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.strategy_texts = strategy_texts
        self.notebook = notebook
        self.tools = tools
        found = legends(tools.workbook.worksheets)
        self.formats = INSPECTOR_LEGEND_TEXT.format(legends=legend_notes(found)) if found else ""  # added to the instruction
        if found:
            notebook.namespace["legends"] = legend_fills(found)
        notebook.namespace["formats"] = format_marks(tools.workbook.worksheets)
        self.formats += FORMATS_VARIABLE_TEXT

    def inspect(
        self, question: str, ledger: Ledger, need: Need, strategy: str, instruction: str, step: int,
    ) -> Evidence:
        attempt = need.inspections + 1
        log = self.llm.log
        log.event("inspection_start", need_id=need.id, attempt=attempt, strategy=strategy, instruction=instruction)
        # A value left by an earlier attempt must not pass for this attempt's evidence.
        self.notebook.namespace.pop(need.variable, None)
        turns: list[Turn] = []
        for number in range(1, self.max_turns + REPORT_TURNS + 1):
            if self.llm.used >= MAX_LLM_CALLS:
                break
            prompt = self._prompt(question, ledger, need, strategy, instruction, turns, number)
            try:
                reply = ask_json(self.llm, "inspect", self.system_prompt, prompt, _check_turn)
            except ReplyError as exc:
                turns.append(Turn(number, error=str(exc)))
                continue
            turn = Turn(number, reply["thought"], reply["code"])
            turns.append(turn)
            if turn.code:
                cell = self.notebook.run(turn.code, f"{need.id}_attempt{attempt}_turn{number}")
                turn.output, turn.error = clip(cell.output, OUTPUT_CHARS), cell.error
            if reply["done"]:
                turn.runner_note = _contract_problem(
                    _read_evidence(self.notebook.namespace.get(need.variable)), need.variable, turn.error,
                )
            log.event("inspection_turn", need_id=need.id, attempt=attempt, **asdict(turn))
            if reply["done"] and not turn.runner_note:
                break
        evidence = self._evidence(need, attempt, step, strategy, instruction, turns)
        log.event("inspection_evidence", **asdict(evidence))
        return evidence

    def _evidence(
        self, need: Need, attempt: int, step: int, strategy: str, instruction: str, turns: list[Turn],
    ) -> Evidence:
        evidence = Evidence(need.id, attempt, step, strategy, instruction)
        evidence.turns = [asdict(turn) for turn in turns]
        evidence.findings = next((turn.output for turn in reversed(turns) if turn.output), "")
        reported = _read_evidence(self.notebook.namespace.get(need.variable))
        if _contract_problem(reported, need.variable, ""):
            last = turns[-1] if turns else None
            evidence.note = "The inspection ended without valid evidence." + (
                f" Last error: {last.error}" if last and last.error else ""
            )
            return evidence
        evidence.value = reported["value"]
        evidence.found = reported["value"] not in (None, "", [], {})  # plain Python after _read_evidence
        cells = reported.get("cells") or []
        evidence.cells = [str(cell) for cell in (cells if isinstance(cells, list) else [cells])]
        evidence.where = str(reported.get("where") or "")
        evidence.note = str(reported.get("note") or "")
        self._as_shown(evidence)
        return evidence

    def _as_shown(self, evidence: Evidence) -> None:
        """A number read from one cell becomes the number as the cell's format shows it."""
        value = evidence.value
        if not isinstance(value, (int, float)) or isinstance(value, bool) or len(evidence.cells) != 1:
            return
        try:
            cells = parse_a1_range(evidence.cells[0], "")
            ws = self.tools.worksheet(cells.sheet)
        except ValueError:
            return
        if (cells.start_row, cells.start_col) != (cells.end_row, cells.end_col):
            return
        raw = self.tools.value(cells.start_row, cells.start_col, ws.title)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or abs(raw - value) > 1e-9:
            return
        fmt = ws.cell(cells.start_row, cells.start_col).number_format
        shown = shown_number(raw, fmt)
        if shown is not None and shown != raw:
            evidence.value = shown
            evidence.note = (evidence.note + f" Shown as {shown} by the cell's number format {fmt} (stored {raw}).").strip()

    def _prompt(
        self, question: str, ledger: Ledger, need: Need, strategy: str, instruction: str, turns: list[Turn], turn: int,
    ) -> str:
        earlier = [
            line for item in ledger.evidence if item.need_id == need.id
            for line in evidence_lines(item) + _seen_lines(item)
        ]
        return INSPECTOR_PROMPT.format(
            question=question,
            hypothesis=ledger.hypothesis,
            need_id=need.id,
            need=need.description,
            strategy=strategy,
            strategy_text=self.strategy_texts[strategy],
            instruction=(instruction or "(none)") + self.formats,
            earlier=("Earlier attempts for this need:\n" + "\n".join(earlier) + "\n") if earlier else "",
            sheets="; ".join(self.tools.sheets()) + " (listed here: do not call sheet.sheets())",
            variable=need.variable,
            transcript=transcript(turns),
            turn=turn,
            max_turns=self.max_turns,
            last=_last_text(need.variable, turn, self.max_turns),
        )


def _last_text(variable: str, turn: int, max_turns: int) -> str:
    """What the turn line adds: the pace, the last turn's order to report, or the report turn's."""
    if turn > max_turns:
        return (
            f" The turns are used up and `{variable}` is not set. Report turn: look nothing up; set `{variable}` from what"
            " the turns above printed (value None, with why in note, if they do not show it) and set done to true."
        )
    if turn == max_turns:
        return (
            f" This is the last turn: set `{variable}` from what you have found (value None, with why in note,"
            " if nothing fits) and set done to true."
        )
    pace = " Most needs take 3 to 6 turns: as" if max_turns >= 6 else " As"  # the pace only where 6 turns fit
    return (
        f"{pace} soon as you have seen where the value is, set `{variable}` and set done"
        " to true. Establish this need only; the other needs are inspected separately."
    )


def _seen_lines(item: Evidence) -> list[str]:
    """What an attempt that found nothing had worked out by its end: its last thoughts, for the next attempt."""
    if item.found:
        return []
    thoughts = [turn["thought"] for turn in item.turns if turn.get("thought")][-2:]
    return [f"  it ended thinking: {clip(' '.join(thought.split()), 500)}" for thought in thoughts]


def shown_number(value: float, fmt: str) -> float | None:
    """The number a fixed-decimal number format shows, or None for General and text formats."""
    section = re.sub(r'"[^"]*"|\\.|_.|\*.', "", fmt.split(";")[0])
    if not re.search(r"[0#]", section):
        return None
    decimals = re.search(r"\.([0#]+)", section)
    places = len(decimals.group(1).replace("#", "")) if decimals else 0
    if "%" in section:
        return round(value, places + 2)
    return round(value, places) if places else round(value)


def _check_turn(data: Any) -> dict[str, Any]:
    data = require_dict(data)
    code, done = data.get("code", ""), data.get("done", False)
    if not isinstance(code, str) or not isinstance(done, bool):
        raise ValueError('"code" must be a string and "done" a boolean')
    if not done and not code.strip():
        raise ValueError('"code" must not be empty unless "done" is true')
    return {"thought": str(data.get("thought") or "").strip(), "code": code.strip(), "done": done}


def _read_evidence(reported: Any) -> Any:
    """The evidence dict with a numpy or pandas value turned into plain Python (lists, dicts, numbers)."""
    if not isinstance(reported, dict) or "value" not in reported:
        return reported
    value = reported["value"]
    if hasattr(value, "tolist"):  # numpy arrays and scalars, pandas Series
        value = value.tolist()
    elif hasattr(value, "to_dict"):  # pandas DataFrame
        value = value.to_dict()
    return {**reported, "value": value}


def _contract_problem(reported: Any, variable: str, error: str) -> str:
    """Why a `done` reply cannot end the inspection, or an empty string."""
    if error:
        return "done was refused: this turn's code raised an error."
    if not isinstance(reported, dict):
        return f"done was refused: `{variable}` is not set to a dict."
    missing = [key for key in ("value", "cells") if key not in reported]
    if missing:
        return f"done was refused: `{variable}` lacks " + ", ".join(f'"{key}"' for key in missing) + "."
    return ""


def transcript(turns: list[Turn], title: str = "This inspection so far") -> str:
    """The turns so far; when too long, the oldest outputs are left out first."""
    if not turns:
        return ""
    for hidden in range(len(turns) + 1):
        text = "\n\n".join(_turn_text(turn, show_output=index >= hidden) for index, turn in enumerate(turns))
        if len(text) <= TRANSCRIPT_CHARS:
            break
    return f"{title}:\n{text}\n\n"


def _turn_text(turn: Turn, show_output: bool) -> str:
    lines = [f"Turn {turn.number}", f"Thought: {turn.thought}"]
    if turn.code:
        lines.append(f"Code:\n{turn.code}")
    if turn.output or turn.error:
        body = "\n".join(part for part in (turn.output, f"Error: {turn.error}" if turn.error else "") if part)
        lines.append(f"Output:\n{body if show_output else '(left out)'}")
    if turn.runner_note:
        lines.append(f"Runner: {turn.runner_note}")
    return "\n".join(lines)


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[cut]"
