"""The evidence ledger of one question: the needs to establish, what each inspection found, and every step.

The ledger also enforces the loop's rules (which action is allowed now) and limits, and decides what the
reasoning steps see: a check that refuted a need shows in full until it is ruled on, then in one line.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Limits of one question.
MAX_STEPS = 20  # reasoning steps between inspections
MAX_INSPECTIONS_PER_NEED = 3  # counted since the need was last revised
MAX_HYPOTHESIS_REVISIONS = 1
MAX_RETURNS = 1  # times the answer review may send the loop back
MAX_LLM_CALLS = 80

STRATEGIES = ("search_sheet", "structure", "window")
ACTIONS = {
    "inspect": '"need_id", "strategy", "instruction"',
    "add_need": '"need": {"id", "description"}',
    "revise_need": '"need_id", "description"',
    "drop_need": '"need_id"',
    "revise_hypothesis": '"hypothesis", optionally "answer_form" and "answer_slots"',
    "synthesize": "no arguments",
}
REVISIONS = ("add_need", "revise_need", "drop_need", "revise_hypothesis")
RULINGS = ("keep", "reject")
NEED_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

PROMPT_VALUE_CHARS = 200
PROMPT_NOTE_CHARS = 300
PROMPT_FINDINGS_CHARS = 600
PROMPT_RULED_CHARS = 160  # a check already ruled on shows in one line


@dataclass
class Need:
    id: str
    description: str
    status: str = "open"  # open, found or dropped
    inspections: int = 0
    inspections_at_revision: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def variable(self) -> str:
        """The notebook variable an inspection of this need must set."""
        return f"evidence_{self.id}"

    @property
    def inspections_left(self) -> int:
        return MAX_INSPECTIONS_PER_NEED - (self.inspections - self.inspections_at_revision)


@dataclass
class Evidence:
    """What one inspection reported for a need, the checks of it, and the ruling on a refuting check."""

    need_id: str
    attempt: int
    step: int
    strategy: str
    instruction: str
    found: bool = False
    value: Any = None
    cells: list[str] = field(default_factory=list)
    where: str = ""
    note: str = ""
    findings: str = ""  # what the inspection's code printed last
    turns: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)  # {"by", "verdict", "reason", "cells", ...} in order
    ruling: dict[str, Any] | None = None  # {"ruling", "why", "step"} once a reasoning step ruled on a refuting check

    @property
    def refuted(self) -> bool:
        return any(check["verdict"] == "refuted" for check in self.checks)

    @property
    def disputed(self) -> bool:
        """A refuting check that no reasoning step has ruled on yet."""
        return self.found and self.refuted and self.ruling is None


@dataclass
class Step:
    index: int
    kind: str  # the abstract reasoning at the start, at a step, or at the end
    assessment: str
    action: str
    args: dict[str, Any]
    accepted: bool = True
    refusal: str = ""
    reason: str = ""
    rulings: list[dict[str, Any]] = field(default_factory=list)  # on refuting checks, applied before the action


@dataclass
class Ledger:
    question: str
    hypothesis: str = ""
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    answer_form: str = ""
    answer_slots: list[dict[str, str]] = field(default_factory=list)
    needs: dict[str, Need] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    answers: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ changes

    def start(self, plan: dict[str, Any]) -> None:
        """Start from the first reasoning: hypothesis, answer form and slots, and the needs to establish."""
        self.hypothesis = plan["hypothesis"]
        self.hypotheses = [{"step": 0, "hypothesis": self.hypothesis, "reason": "initial"}]
        self.answer_form = plan["answer_form"]
        self.answer_slots = plan["answer_slots"]
        self.needs = {item["id"]: Need(item["id"], item["description"]) for item in plan["needs"]}
        self.steps.append(Step(0, "start", plan["analysis"], "start", {}))

    def refusal(self, action: Any, args: dict[str, Any]) -> str:
        """Why an action is not allowed now, or an empty string."""
        if action not in ACTIONS:
            return f"Unknown action {action!r}; actions and their arguments: " + "; ".join(
                f"{name} ({arguments})" for name, arguments in ACTIONS.items()
            )
        if action == "synthesize":
            if self.open_needs():
                return f"Cannot synthesize while needs are open: {', '.join(f.id for f in self.open_needs())}."
            if not self.found_needs():
                return "Cannot synthesize: no need is found."
            if self.disputed_needs():
                return f"Cannot synthesize before ruling on the checks of: {', '.join(self.disputed_needs())}."
            return ""
        if action == "add_need":
            new = args.get("need")
            if not isinstance(new, dict) or not _text(new.get("description")):
                return 'add_need needs "need": {"id", "description"}.'
            if not NEED_ID.fullmatch(str(new.get("id"))) or str(new.get("id")) in self.needs:
                return f"Need id {new.get('id')!r} must be new and made of letters, digits and underscores."
            return ""
        if action == "revise_hypothesis":
            if len(self.hypotheses) > MAX_HYPOTHESIS_REVISIONS:
                return "The hypothesis was already revised; it may be revised once."
            if not _text(args.get("hypothesis")):
                return "revise_hypothesis needs a hypothesis."
            return slots_problem(args["answer_slots"]) if "answer_slots" in args else ""
        need = self.needs.get(str(args.get("need_id")))
        if need is None:
            return f"Unknown need id {args.get('need_id')!r}."
        if action == "revise_need":
            if need.history and need.history[-1]["change"] == "revised" and need.inspections == need.inspections_at_revision:
                return f"Need {need.id} was revised and not inspected since; inspect it before revising it again."
            return "" if _text(args.get("description")) else "revise_need needs a description."
        if action == "drop_need":
            others = [f for f in self.needs.values() if f is not need and f.status != "dropped"]
            return "" if others else f"Cannot drop {need.id}, the last need; revise it instead."
        if need.status == "dropped":
            return f"Need {need.id} was dropped; revise it to reopen it."
        if need.inspections_left <= 0:
            return (
                f"Need {need.id} was inspected {MAX_INSPECTIONS_PER_NEED} times since it was last revised; "
                "revise it (a new description reopens it) or drop it if the question does not need it."
            )
        if args.get("strategy") not in STRATEGIES:
            return f"Unknown strategy {args.get('strategy')!r}; choose one of: {', '.join(STRATEGIES)}."
        return ""

    def apply(self, action: str, args: dict[str, Any], step: int, reason: str) -> None:
        """Apply an allowed action that edits the ledger (all but inspect and synthesize)."""
        if action == "add_need":
            need = Need(str(args["need"]["id"]), args["need"]["description"].strip())
            need.history.append({"step": step, "change": "added", "reason": reason})
            self.needs[need.id] = need
        elif action == "revise_need":
            need = self.needs[str(args["need_id"])]
            need.history.append({"step": step, "change": "revised", "reason": reason, "before": need.description})
            need.description = args["description"].strip()
            need.status = "open"
            need.inspections_at_revision = need.inspections
        elif action == "drop_need":
            need = self.needs[str(args["need_id"])]
            need.history.append({"step": step, "change": "dropped", "reason": reason})
            need.status = "dropped"
        elif action == "revise_hypothesis":
            self.hypothesis = args["hypothesis"].strip()
            self.hypotheses.append({"step": step, "hypothesis": self.hypothesis, "reason": reason})
            if _text(args.get("answer_form")):
                self.answer_form = args["answer_form"].strip()
            if "answer_slots" in args:
                self.answer_slots = args["answer_slots"]

    def record(self, evidence: Evidence) -> None:
        need = self.needs[evidence.need_id]
        need.inspections += 1
        need.status = "found" if evidence.found else "open"
        self.evidence.append(evidence)

    def rule(self, rulings: Any, step: int) -> list[dict[str, str]]:
        """Apply the rulings of a reasoning step on refuting checks; returns the ones applied.

        Keeping the evidence leaves the need found; rejecting it reopens the need.
        """
        applied = []
        for item in rulings if isinstance(rulings, list) else []:
            if not isinstance(item, dict) or str(item.get("need_id")) not in self.disputed_needs():
                continue
            if item.get("ruling") not in RULINGS:
                continue
            need = self.needs[str(item["need_id"])]
            ruling = {"ruling": item["ruling"], "why": str(item.get("why") or "").strip(), "step": step}
            self.latest(need.id).ruling = ruling
            if ruling["ruling"] == "reject":
                need.status = "open"
            applied.append({"need_id": need.id, **ruling})
        return applied

    # -------------------------------------------------------------------- state

    def open_needs(self) -> list[Need]:
        return [need for need in self.needs.values() if need.status == "open"]

    def found_needs(self) -> list[Need]:
        return [need for need in self.needs.values() if need.status == "found"]

    def latest(self, need_id: str) -> Evidence | None:
        return next((item for item in reversed(self.evidence) if item.need_id == need_id), None)

    def disputed_needs(self) -> list[str]:
        """Found needs whose latest evidence has a refuting check not ruled on yet."""
        return [need.id for need in self.found_needs() if self.latest(need.id).disputed]

    @property
    def decisions(self) -> list[Step]:
        return [step for step in self.steps if step.kind == "step"]

    @property
    def hypothesis_revisions(self) -> int:
        return len(self.hypotheses) - 1

    @property
    def returns(self) -> int:
        return sum(1 for step in self.steps if step.kind == "end" and step.args.get("verdict") == "return")

    def revision_count(self) -> int:
        return sum(1 for step in self.decisions if step.accepted and step.action in REVISIONS)

    # ---------------------------------------------------------------- rendering

    def slots_text(self, parts: list[str] | None = None) -> str:
        lines = []
        for index, slot in enumerate(self.answer_slots, start=1):
            line = f'{index}. {slot["slot"]} (asked by: "{slot["asked_by"]}")'
            if parts is not None:
                line += f" -> {parts[index - 1] if index <= len(parts) else '(missing)'}"
            lines.append(line)
        return "\n".join(lines)

    def needs_text(self) -> str:
        """Needs with their latest evidence and its checks; earlier attempts in one line each."""
        blocks = []
        for need in self.needs.values():
            lines = [f"{need.id} [{need.status}] {need.description} (inspections left: {need.inspections_left})"]
            attempts = [item for item in self.evidence if item.need_id == need.id]
            for item in attempts[:-1]:
                lines.append(f"  attempt {item.attempt} ({item.strategy}): {_short_result(item)}")
            if attempts:
                lines += ["  " + line for line in evidence_lines(attempts[-1])]
            blocks.append("\n".join(lines))
        return "\n".join(blocks) or "(no needs)"

    def history_text(self) -> str:
        lines = []
        for step in self.decisions:
            line = f"step {step.index}: {step.action} {_args_text(step.args)}".rstrip()
            if step.rulings:
                line += " (ruled " + ", ".join(f"{item['need_id']} {item['ruling']}" for item in step.rulings) + ")"
            if step.refusal:
                line += f" -> refused: {step.refusal}"
            lines.append(line)
        return "\n".join(lines) or "(none yet)"


def evidence_lines(item: Evidence) -> list[str]:
    """The latest attempt for a need as prompt lines: what was reported, then its checks."""
    lines = [f"attempt {item.attempt} ({item.strategy}): {_short_result(item)}"]
    if item.where:
        lines.append(f"  where: {_clip(item.where, PROMPT_NOTE_CHARS)}")
    if item.note:
        lines.append(f"  note: {_clip(item.note, PROMPT_NOTE_CHARS)}")
    if item.findings:
        lines.append(f"  printed last: {_clip(item.findings, PROMPT_FINDINGS_CHARS)}")
    lines += [f"  {line}" for line in check_lines(item)]
    return lines


def check_lines(item: Evidence) -> list[str]:
    """The checks of one attempt: a refuting check in full until it is ruled on, any other in one line."""
    lines = []
    for check in item.checks:
        cells = f" (cells: {', '.join(check['cells'])})" if check.get("cells") else ""
        if check["verdict"] != "refuted" or item.ruling is not None:
            line = f"check {check['by']}: {check['verdict']} - {_clip(check['reason'], PROMPT_RULED_CHARS)}"
            lines.append(line + (f"; ruled {item.ruling['ruling']}: {_clip(item.ruling['why'], PROMPT_RULED_CHARS)}"
                                 if check["verdict"] == "refuted" else ""))
            continue
        lines.append(f"check {check['by']}: REFUTED - {_clip(check['reason'], PROMPT_NOTE_CHARS)}{cells}")
        lines += [f"  - {objection['objection']} ({objection['shown']})" for objection in check.get("objections", [])]
        lines += [f"  {fact}" for fact in check.get("facts", [])]
        lines.append("  RULE ON IT: keep or reject this evidence in \"rulings\".")
    return lines


def slots_problem(slots: Any) -> str:
    """What is wrong with a list of answer slots, or an empty string."""
    if not isinstance(slots, list) or not slots:
        return "answer_slots must be a non-empty list"
    for slot in slots:
        if not isinstance(slot, dict) or not _text(slot.get("slot")) or not _text(slot.get("asked_by")):
            return 'each answer slot must be {"slot": ..., "asked_by": "<phrase of the question>"} with both filled'
    return ""


def _short_result(item: Evidence) -> str:
    if not item.found:
        return "not found" + (f" ({_clip(item.note, 120)})" if item.note else "")
    result = f"value {_clip(_json(item.value), PROMPT_VALUE_CHARS)} from {', '.join(item.cells) or '(no cells)'}"
    if item.ruling and item.ruling["ruling"] == "reject":
        result += f"; rejected: {_clip(item.ruling['why'], PROMPT_RULED_CHARS)}"
    return result


def _args_text(args: dict[str, Any]) -> str:
    return _clip(", ".join(f"{key}={_json(value)}" for key, value in args.items()), 200)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
