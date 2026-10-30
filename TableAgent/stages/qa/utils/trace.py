"""Logging and the artifacts of one question.

Per question, in its run directory:
- trace.md      the readable story: framing, every step with its inspections and checks, synthesis, review
- ledger.json   the full ledger (needs, evidence with checks and rulings, steps, answers)
- events.jsonl  every event in order, including prompts and model calls
- events.log    the same events to read: one block per event, prompts left out
- result.json   the outcome and the list of artifacts
- cells/        each notebook cell's code, with what it printed as comments
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from TableAgent.stages.qa.ledger import MAX_LLM_CALLS, MAX_RETURNS, MAX_STEPS, Evidence, Ledger, Step

if TYPE_CHECKING:
    from TableAgent.stages.qa.orchestrator import AnswerResult
    from TableAgent.stages.qa.tools.notebook import CellRun

CELL_OUTPUT_CHARS = 3000
TRACE_TEXT_CHARS = 600
TRACE_LINE_CHARS = 200
LOG_TEXT_CHARS = 2000  # of one field in events.log
LOG_VALUE_CHARS = 500  # of a list or dict field in events.log
LOG_LEFT_OUT = ("turns",)  # repeated in events.log by the turn events


class RunLog:
    """Events of one question, and optional one-line progress messages."""

    def __init__(self, progress: Callable[[str], None] | None = None):
        self.events: list[dict[str, Any]] = []
        self.progress = progress

    def event(self, kind: str, **data: Any) -> None:
        self.events.append(
            {"timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"), "event_type": kind, **data}
        )

    def say(self, message: str) -> None:
        if self.progress is not None:
            self.progress(f"[qa] {message}")


def run_dir_name(question: str) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", question.strip()).strip("_").lower()[:60]
    return f"{stamp}_{slug or 'question'}"


def write_artifacts(run_dir: Path, result: AnswerResult, ledger: Ledger, log: RunLog, cells: list[CellRun]) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    cell_paths: dict[str, str] = {}
    for index, cell in enumerate(cells, start=1):
        name = f"{index:02d}_{cell.label}.py"
        (cells_dir / name).write_text(_cell_file(cell), encoding="utf-8")
        cell_paths[cell.label] = f"cells/{name}"
    artifacts = {
        "run_dir": str(run_dir),
        "trace_md": str(run_dir / "trace.md"),
        "ledger_json": str(run_dir / "ledger.json"),
        "events_jsonl": str(run_dir / "events.jsonl"),
        "events_log": str(run_dir / "events.log"),
        "result_json": str(run_dir / "result.json"),
        "cells_dir": str(cells_dir),
    }
    (run_dir / "trace.md").write_text(render_trace(result, ledger, cell_paths), encoding="utf-8")
    _write_json(run_dir / "ledger.json", asdict(ledger))
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in log.events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    (run_dir / "events.log").write_text(render_events(log.events), encoding="utf-8")
    _write_json(run_dir / "result.json", {
        "question": result.question,
        "success": result.success,
        "answer": result.answer,
        "error": result.error,
        "seconds": result.seconds,
        "token_usage": result.token_usage,
        "llm_call_count": len(result.llm_calls),
        "artifacts": artifacts,
    })
    return artifacts


def render_trace(result: AnswerResult, ledger: Ledger, cell_paths: dict[str, str]) -> str:
    lines = [
        "# QA trace",
        "",
        f"- **Question:** {ledger.question}",
        f"- **Answer:** {result.answer if result.answer is not None else '(none)'}",
        f"- **Success:** {result.success}" + (f" ({result.error})" if result.error else ""),
        f"- **Budget:** steps {len(ledger.decisions)}/{MAX_STEPS} | model calls {len(result.llm_calls)}/{MAX_LLM_CALLS} "
        f"| hypothesis revisions {ledger.hypothesis_revisions} | review returns {ledger.returns}/{MAX_RETURNS} "
        f"| {result.seconds:.1f}s | tokens {result.token_usage.get('prompt', 0)} in, {result.token_usage.get('completion', 0)} out",
        "",
    ]
    start = next((step for step in ledger.steps if step.kind == "start"), None)
    if start is None:
        return "\n".join(lines)
    lines += ["## Abstract reasoning: start", "", _quote(start.assessment), "", f"**Hypothesis:** {ledger.hypotheses[0]['hypothesis']}", ""]
    lines += [f"**Answer form:** {ledger.answer_form}", "", "**Answer slots:**", "", ledger.slots_text(), ""]
    lines += ["| need | description | status | inspections |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {need.id} | {_cell(need.description)} | {need.status} | {need.inspections} |" for need in ledger.needs.values()
    ]
    lines.append("")
    for step in ledger.steps:
        if step.kind == "step":
            lines += _step_lines(step)
            lines += [
                line
                for item in ledger.evidence if item.step == step.index
                for line in _evidence_lines(item, cell_paths)
            ]
        for answer in ledger.answers:
            if answer["step"] == step.index and step.kind != "end":
                lines += _answer_lines(answer)
        if step.kind == "end":
            lines += [
                f"### Abstract reasoning: end ({step.args.get('verdict')})", "",
                f"- **Assessment:** {_clip(step.assessment, TRACE_TEXT_CHARS)}",
            ]
            if step.args.get("problem"):
                lines += [f"- **Problem:** {step.args['problem']}", f"- **Instruction:** {step.args.get('instruction', '')}"]
            lines.append("")
    for revision in ledger.hypotheses[1:]:
        lines += [f"- Hypothesis revised at step {revision['step']}: {revision['hypothesis']}"]
    return "\n".join(lines).rstrip() + "\n"


def render_events(events: list[dict[str, Any]]) -> str:
    """events.jsonl to read: a line per model call, a block per other event with one field per line."""
    blocks = []
    for event in events:
        time = event["timestamp"].split("T")[-1]
        fields = {key: value for key, value in event.items() if key not in ("timestamp", "event_type")}
        if event["event_type"] == "llm_call":
            status = "ok" if fields.get("success") else f"failed ({fields.get('error_type')})"
            blocks.append(
                f"{time} llm_call #{fields.get('index')} {status} {fields.get('duration_ms')} ms, "
                f"{fields.get('prompt_tokens', 0)} in / {fields.get('completion_tokens', 0)} out"
            )
            continue
        lines = [f"{time} {event['event_type']}"]
        for key, value in fields.items():
            if key in LOG_LEFT_OUT or value in (None, "", [], {}):
                continue
            if key == "prompt":
                lines.append(f"  prompt: ({len(value)} chars, in events.jsonl)")
            elif isinstance(value, str) and "\n" in value:
                lines.append(f"  {key}:")
                lines += [f"    {line}" for line in _clip(value, LOG_TEXT_CHARS).splitlines()]
            elif isinstance(value, str):
                lines.append(f"  {key}: {_clip(value, LOG_TEXT_CHARS)}")
            else:
                lines.append(f"  {key}: {_clip(json.dumps(value, ensure_ascii=False, default=str), LOG_VALUE_CHARS)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _step_lines(step: Step) -> list[str]:
    args = ", ".join(f"{key}={_clip(json.dumps(value, ensure_ascii=False, default=str), 120)}" for key, value in step.args.items())
    lines = [f"### Step {step.index}: {step.action}" + (f" ({args})" if args else ""), ""]
    lines.append(f"- **Assessment:** {_clip(step.assessment, TRACE_TEXT_CHARS)}")
    for item in step.rulings:
        lines.append(f"- **Ruling on {item['need_id']}:** {item['ruling']} — {_clip(item['why'], TRACE_LINE_CHARS)}")
    if step.reason:
        lines.append(f"- **Reason:** {_clip(step.reason, TRACE_LINE_CHARS)}")
    if not step.accepted:
        lines.append(f"- **Refused:** {step.refusal}")
    lines.append("")
    return lines


def _evidence_lines(item: Evidence, cell_paths: dict[str, str]) -> list[str]:
    status = "found" if item.found else "not found"
    lines = [f"#### Inspection {item.need_id}, attempt {item.attempt} ({item.strategy}): {status}", ""]
    lines += _turn_lines(item.turns, f"{item.need_id}_attempt{item.attempt}_turn", cell_paths)
    lines.append("")
    if item.found:
        lines += [
            "| value | cells | where |",
            "| --- | --- | --- |",
            f"| {_cell(json.dumps(item.value, ensure_ascii=False, default=str))} | {_cell(', '.join(item.cells))} "
            f"| {_cell(item.where)} |",
            "",
        ]
    if item.note:
        lines += [f"- **Note:** {_clip(item.note, TRACE_TEXT_CHARS)}", ""]
    for check in item.checks:
        lines += [f"##### Check {check['by']}: {check['verdict']}", ""]
        lines += _turn_lines(check.get("turns", []), f"{item.need_id}_attempt{item.attempt}_{check['by']}_turn", cell_paths)
        lines += [f"- {_clip(fact, TRACE_TEXT_CHARS)}" for fact in check.get("facts", [])]
        lines += [
            f"- Objection: {_clip(objection['objection'], TRACE_LINE_CHARS)} ({objection['shown']})"
            for objection in check.get("objections", [])
        ]
        lines += [f"- **Reason:** {_clip(check['reason'], TRACE_TEXT_CHARS)}", ""]
    return lines


def _turn_lines(turns: list[dict[str, Any]], label_prefix: str, cell_paths: dict[str, str]) -> list[str]:
    lines = []
    for turn in turns:
        result = _first_line(turn["error"]) or _first_line(turn["output"]) or "(printed nothing)"
        line = f"- Turn {turn['number']}: {_clip(_first_line(turn['thought']), 150)} → {_clip(result, 150)}"
        label = f"{label_prefix}{turn['number']}"
        if label in cell_paths:
            line += f" [`{cell_paths[label]}`]"
        if turn["runner_note"]:
            line += f" — runner: {_clip(_first_line(turn['runner_note']), 150)}"
        lines.append(line)
    return lines


def _answer_lines(answer: dict[str, Any]) -> list[str]:
    if answer.get("error"):
        return [f"### Synthesis failed: {answer['error']}", ""]
    parts = " | ".join(answer["parts"])
    lines = [f"### Synthesis → {answer['answer']}", "", f"- Parts: {parts}"]
    if answer.get("method_check"):
        check = answer["method_check"]
        lines.append(f"- **Method check:** {check['verdict']}" + (f" — {_clip(check['problem'], TRACE_TEXT_CHARS)}" if check["problem"] else ""))
    return lines + [""]


def _cell_file(cell: CellRun) -> str:
    printed = cell.output[:CELL_OUTPUT_CHARS] + ("\n...[cut]" if len(cell.output) > CELL_OUTPUT_CHARS else "")
    tail = ["# --- printed ---", *(f"# {line}" for line in printed.splitlines())] if printed else []
    if cell.error:
        tail += ["# --- error ---"] + [f"# {line}" for line in cell.error.splitlines()]
    return cell.code.rstrip() + "\n" + ("\n" + "\n".join(tail) + "\n" if tail else "")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" for line in _clip(text, 2000).splitlines() or [""])


def _first_line(text: str) -> str:
    return next((line.strip() for line in str(text or "").splitlines() if line.strip()), "")


def _clip(text: Any, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(value: Any) -> str:
    return _clip("" if value is None else value, TRACE_LINE_CHARS).replace("|", "\\|").replace("\n", " ")
