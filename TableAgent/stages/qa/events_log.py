"""Render QA run events as a human-readable ``events.log``.

``events.jsonl`` stays the complete, machine-readable record; event ``#N`` in
the log is line ``N`` of that file. This view is compacted for reading:

- a summary (answer, timing, LLM calls, tokens, per-attempt plan status) first;
- prompts show their first lines once, then only the lines changed since the
  previous prompt of the same kind;
- each code block, observation, reasoning, or feedback text is printed once;
  later repeats point back to the event that printed it;
- raw LLM responses are omitted when the next event holds the parsed result;
- plans are rendered as tables.
"""

from __future__ import annotations

import ast
import datetime
import difflib
import json
import re
import textwrap
from typing import Any, Iterable

WIDTH = 120
PROMPT_HEAD_LINES = 20
TEXT_HEAD_LINES = 40
DIFF_MAX_LINES = 40
DEDUPE_MIN_CHARS = 60

_SECTION_TITLES = {
    "run_start": "RUN",
    "planning_start": "PLANNING",
    "final_answer_review_prompt": "FINAL ANSWER REVIEW",
    "run_complete": "RESULT",
}

# Raw response event -> event that carries the parsed result.
_PARSED_BY = {
    "planner_response": "planning_complete",
    "generate_response": "generate_parsed",
    "generate_repair_response": "generate_repair_parsed",
    "review_response": "review_subtask",
    "final_answer_review_response": "final_answer_review",
}

_PROMPT_FIELDS = {"prompt", "system_prompt"}
_PLAN_FIELDS = {"subtasks"}

_FIELD_ORDER = [
    "subtask_id",
    "cell_id",
    "action",
    "layer",
    "category",
    "round_num",
    "attempt",
    "repair_attempt",
    "success",
    "accepted",
    "score",
    "description",
    "question",
    "depends_on",
    "order",
    "error",
    "feedback",
    "reasoning",
    "system_prompt",
    "prompt",
    "content",
    "subtasks",
    "code",
    "stdout_preview",
    "stderr_preview",
    "error_preview",
    "observation",
    "namespace_updates",
    "final_answer",
]

_HIDDEN_FIELDS = {
    "timestamp",
    "event_type",
    "stdout_chars",
    "stderr_chars",
    "error_chars",
    "stdout_truncated",
    "stderr_truncated",
    "error_truncated",
    "namespace_updates_keys",
    "previous_plan",
    "dependencies",
}

_FIELD_LABELS = {
    "stdout_preview": "stdout",
    "stderr_preview": "stderr",
    "error_preview": "exception",
    "content": "response",
    "round_num": "round",
}

_REPR_FIELD_RE = re.compile(r"(\w+)=('(?:[^'\\]|\\.)*'|\[[^\]]*\])")


def format_events_log(events: Iterable[dict[str, Any]]) -> str:
    """Return a readable rendering of QA events: summary first, then each event."""
    return _Renderer([event for event in events if isinstance(event, dict)]).render()


class _Renderer:
    def __init__(self, events: list[dict[str, Any]]):
        self.events = events
        self.times = [_parse_timestamp(event.get("timestamp")) for event in events]
        self.start_time = next((time for time in self.times if time), None)
        self.lines: list[str] = []
        self.prompt_history: dict[tuple[str, str], tuple[int, str]] = {}
        self.text_seen: dict[str, tuple[int, str]] = {}
        self.exec_code_at: dict[str, int] = {}
        self.descriptions: dict[str, str] = {}
        for number, event in enumerate(events, start=1):
            if event.get("event_type") == "execute_code" and isinstance(event.get("code"), str):
                self.exec_code_at.setdefault(event["code"], number)
            if event.get("event_type") == "subtask_start" and event.get("subtask_id"):
                self.descriptions.setdefault(str(event["subtask_id"]), str(event.get("description") or ""))
        self.attempts, self.subtasks, self.llm = self._collect()

    # ------------------------------------------------------------------ stats

    def _collect(self) -> tuple[list[dict], list[dict], dict]:
        attempts: list[dict] = []
        subtasks: list[dict] = []
        llm = {"calls": 0, "prompt": 0, "completion": 0, "kinds": {}, "logged": False}
        current: dict | None = None
        last_kind = "llm"
        replan_started_at = 0
        for number, event in enumerate(self.events, start=1):
            event_type = str(event.get("event_type") or "")
            if event_type in ("planning_complete", "replanning_complete"):
                plan = self._plan_rows(event.get("subtasks"))
                # A replan logs planning_complete and then replanning_complete for the same plan.
                if event_type == "replanning_complete" and attempts and attempts[-1]["number"] > replan_started_at:
                    attempts[-1]["plan"] = plan or attempts[-1]["plan"]
                else:
                    attempts.append({"number": number, "plan": plan, "error": None})
            elif event_type == "replanning_start":
                replan_started_at = number
                if attempts:
                    attempts[-1]["error"] = event.get("error")
            elif event_type == "subtask_start":
                current = {
                    "id": str(event.get("subtask_id")),
                    "attempt": len(attempts),
                    "start": number,
                    "end": None,
                    "success": None,
                    "rounds": 0,
                    "calls": 0,
                    "prompt": 0,
                    "completion": 0,
                }
                subtasks.append(current)
            elif event_type == "generate_call" and current is not None:
                current["rounds"] += 1
            elif event_type in ("subtask_complete", "subtask_exception") and current is not None:
                current["end"] = number
                if event_type == "subtask_exception":
                    current["success"] = False
                elif isinstance(event.get("success"), bool):
                    current["success"] = event["success"]
                if event_type == "subtask_complete":
                    current = None
            elif event_type == "llm_call":
                llm["logged"] = True
                prompt_tokens = int(event.get("prompt_tokens") or 0)
                completion_tokens = int(event.get("completion_tokens") or 0)
                llm["calls"] += 1
                llm["prompt"] += prompt_tokens
                llm["completion"] += completion_tokens
                llm["kinds"][last_kind] = llm["kinds"].get(last_kind, 0) + 1
                if current is not None:
                    current["calls"] += 1
                    current["prompt"] += prompt_tokens
                    current["completion"] += completion_tokens
            if event_type.endswith(("_prompt", "_call")) and event_type != "llm_call":
                last_kind = re.sub(r"_(prompt|call)$", "", event_type)
        if not llm["logged"]:
            llm["calls"] = sum(
                1
                for event in self.events
                if str(event.get("event_type") or "").endswith("_response")
                or event.get("event_type") == "question_understanding"
            )
        return attempts, subtasks, llm

    def _plan_rows(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return []
        rows = []
        for item in value if isinstance(value, list) else []:
            if isinstance(item, dict):
                rows.append(item)
            elif isinstance(item, str):
                fields = {}
                for key, raw in _REPR_FIELD_RE.findall(item):
                    try:
                        fields[key] = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        fields[key] = raw
                if "id" in fields:
                    fields.setdefault("description", self.descriptions.get(str(fields["id"]), ""))
                    rows.append(fields)
        return rows

    # ----------------------------------------------------------------- output

    def render(self) -> str:
        self._render_summary()
        subtask_by_start = {record["start"]: record for record in self.subtasks}
        subtask_counter: dict[int, int] = {}
        for number, event in enumerate(self.events, start=1):
            event_type = str(event.get("event_type") or "event")
            if event_type in _SECTION_TITLES:
                self._banner(_SECTION_TITLES[event_type], "=")
            elif event_type == "replanning_start":
                self._banner(f"REPLANNING (attempt {event.get('attempt', '?')})", "=")
            elif event_type == "subtask_start":
                record = subtask_by_start.get(number, {})
                attempt = record.get("attempt", 0)
                subtask_counter[attempt] = subtask_counter.get(attempt, 0) + 1
                self._banner(self._subtask_title(event, record, subtask_counter[attempt]), "-")
            if event_type == "llm_call":
                self.lines.append(self._header(number, event_type, event) + "  " + _llm_call_text(event))
                continue
            self.lines.append(self._header(number, event_type, event))
            if event_type == "subtask_complete":
                record = next((r for r in self.subtasks if r["end"] == number), None)
                if record:
                    self.lines.append(f"    {self._subtask_stats(record)}")
            self._render_fields(number, event_type, event)
            self.lines.append("")
        return "\n".join(self.lines).rstrip() + "\n"

    def _render_summary(self) -> None:
        run_start = next((e for e in self.events if e.get("event_type") == "run_start"), {})
        run_complete = next((e for e in reversed(self.events) if e.get("event_type") == "run_complete"), {})
        self._banner("SUMMARY", "=")
        rows = [
            ("question", run_start.get("question")),
            ("final answer", _to_text(run_complete.get("final_answer"))),
            ("status", _status_word(run_complete.get("success"))),
            ("error", run_complete.get("error")),
            ("duration", _seconds(run_complete.get("execution_time")) or self._elapsed_text(len(self.events))),
            ("replans", run_complete.get("replan_count")),
            ("llm calls", self._llm_summary()),
            ("tokens", self._token_summary()),
            ("events", f"{len(self.events)} (#N = line N of events.jsonl)"),
        ]
        for label, value in rows:
            if value in (None, ""):
                continue
            text = str(value)
            first, *rest = text.splitlines() or [""]
            self.lines.append(f"{label:>13}: {first}")
            self.lines.extend(f"{'':>15}{line}" for line in rest)

        for index, attempt in enumerate(self.attempts, start=1):
            records = {r["id"]: r for r in self.subtasks if r["attempt"] == index}
            label = "initial plan" if index == 1 else "replan"
            self.lines.append("")
            self.lines.append(f"Attempt {index} ({label}, #{attempt['number']:03d}):")
            table = [["#", "id", "layer", "depends_on", "status", "rounds", "time", "llm", "tokens in/out", "events"]]
            for position, row in enumerate(attempt["plan"], start=1):
                record = records.get(str(row.get("id")))
                table.append([
                    str(position),
                    str(row.get("id", "")),
                    str(row.get("layer", "")),
                    ", ".join(row.get("depends_on") or []) or "-",
                    _status_word(record["success"]) if record else "not run",
                    str(record["rounds"]) if record else "",
                    self._span_text(record) if record else "",
                    str(record["calls"]) if record and self.llm["logged"] else "",
                    f"{record['prompt']:,}/{record['completion']:,}" if record and self.llm["logged"] else "",
                    f"#{record['start']:03d}-#{record['end'] or 0:03d}" if record else "",
                ])
            self.lines.extend("  " + line for line in _table(table))
            if attempt["error"]:
                self.lines.append(f"  -> failed: {_first_line(attempt['error'])}")

    def _llm_summary(self) -> str:
        total = self.llm["calls"]
        if not self.llm["logged"]:
            return f"{total} (per-call timing/tokens not logged in this run)"
        kinds = ", ".join(f"{kind} {count}" for kind, count in self.llm["kinds"].items())
        return f"{total} ({kinds})" if kinds else str(total)

    def _token_summary(self) -> str | None:
        if not self.llm["logged"]:
            return None
        prompt, completion = self.llm["prompt"], self.llm["completion"]
        return f"{prompt:,} in + {completion:,} out = {prompt + completion:,}"

    def _subtask_title(self, event: dict, record: dict, position: int) -> str:
        plan = self.attempts[record["attempt"] - 1]["plan"] if record.get("attempt") else []
        total = f"/{len(plan)}" if plan else ""
        tags = "/".join(str(event[key]) for key in ("layer", "category") if event.get(key))
        title = f"SUBTASK {position}{total}: {event.get('subtask_id', '?')}"
        if tags:
            title += f" ({tags})"
        if record.get("attempt"):
            title += f" · attempt {record['attempt']}"
        if record.get("end"):
            title += f" · {_status_word(record['success'])} · {self._span_text(record)}"
        return title

    def _subtask_stats(self, record: dict) -> str:
        stats = f"subtask: {self._span_text(record)}, {record['rounds']} round(s)"
        if self.llm["logged"]:
            stats += (
                f", {record['calls']} LLM call(s), "
                f"{record['prompt']:,} in / {record['completion']:,} out tokens"
            )
        return stats

    def _span_text(self, record: dict | None) -> str:
        if not record or not record.get("end"):
            return ""
        start, end = self.times[record["start"] - 1], self.times[record["end"] - 1]
        return _seconds((end - start).total_seconds()) if start and end else ""

    def _elapsed_text(self, number: int) -> str:
        time = self.times[number - 1] if number else None
        if time and self.start_time:
            return _seconds((time - self.start_time).total_seconds())
        return ""

    def _header(self, number: int, event_type: str, event: dict) -> str:
        time = self.times[number - 1]
        if time is None:
            stamp = "--:--:--"
        else:
            previous = next((t for t in reversed(self.times[: number - 1]) if t), time)
            stamp = (
                f"{time.strftime('%H:%M:%S.%f')[:-3]} | "
                f"t={(time - self.start_time).total_seconds():.2f}s | "
                f"Δ{(time - previous).total_seconds():.2f}s"
            )
        status = ""
        for key in ("success", "accepted"):
            if event_type != "llm_call" and isinstance(event.get(key), bool):
                status = "  [OK]" if event[key] else "  [FAIL]"
                break
        return f"#{number:03d} [{stamp}] {event_type.upper().replace('_', ' ')}{status}"

    def _banner(self, title: str, char: str) -> None:
        if self.lines and self.lines[-1] != "":
            self.lines.append("")
        self.lines.append(char * WIDTH)
        self.lines.append(f"{char * 2} {title}")
        self.lines.append(char * WIDTH)

    # ----------------------------------------------------------------- fields

    def _render_fields(self, number: int, event_type: str, event: dict) -> None:
        keys = [key for key in _FIELD_ORDER if key in event]
        keys += [key for key in event if key not in keys]
        for key in keys:
            value = event[key]
            if key in _HIDDEN_FIELDS or _is_empty(value):
                continue
            label = _FIELD_LABELS.get(key, key)
            if key.endswith("_preview") and event.get(key.replace("_preview", "_truncated")):
                label += " (truncated)"

            if key == "content" and event_type in _PARSED_BY:
                parsed_at = self._next_event(number, _PARSED_BY[event_type])
                if parsed_at is not None:
                    self.lines.append(f"    {label}: (raw text omitted; parsed result in #{parsed_at:03d})")
                    continue
            if key in _PROMPT_FIELDS and isinstance(value, str):
                self._render_prompt(number, event_type, key, value)
                continue
            if key in _PLAN_FIELDS:
                rows = self._plan_rows(value)
                if rows:
                    table = _plan_table(rows)
                    seen = self.text_seen.setdefault("\n".join(table), (number, "plan"))
                    if seen[0] != number:
                        self.lines.append(f"    {label}: (same plan as #{seen[0]:03d})")
                    else:
                        self._block(label, table)
                    continue
            if key == "order" and isinstance(value, list):
                self.lines.append(f"    {label}: {' -> '.join(map(str, value))}")
                continue
            if key == "namespace_updates" and isinstance(value, str) and "; " in value:
                self._block(label, value.split("; "))
                continue

            text = _to_text(value)
            if key == "code":
                executed_at = self.exec_code_at.get(text)
                if executed_at is not None and executed_at > number:
                    self.lines.append(f"    {label}: (shown with execution in #{executed_at:03d})")
                    continue
            if len(text) >= DEDUPE_MIN_CHARS or "\n" in text:
                seen = self.text_seen.get(text)
                if seen is not None:
                    self.lines.append(f"    {label}: (same as #{seen[0]:03d} {seen[1]})")
                    continue
                self.text_seen[text] = (number, label)
            if key == "content":
                self._block(label, _head(text, TEXT_HEAD_LINES, number))
            else:
                self._field(label, text)

    def _render_prompt(self, number: int, event_type: str, key: str, text: str) -> None:
        history_key = (event_type, key)
        previous = self.prompt_history.get(history_key)
        self.prompt_history[history_key] = (number, text)
        if previous is not None and previous[1] == text:
            self.lines.append(f"    {key}: (same as #{previous[0]:03d})")
            return
        if previous is None:
            self._block(key, _head(self._elide_seen(text), PROMPT_HEAD_LINES, number))
            return
        diff = []
        before, after = self._elide_seen(previous[1]), self._elide_seen(text)
        for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0):
            if line.startswith(("---", "+++")):
                continue
            diff.append("⋯" if line.startswith("@@") else line)
        added = sum(1 for line in diff if line.startswith("+"))
        removed = sum(1 for line in diff if line.startswith("-"))
        if len(diff) > DIFF_MAX_LINES:
            diff = diff[:DIFF_MAX_LINES] + [
                f"… +{len(diff) - DIFF_MAX_LINES} more diff lines (full text: events.jsonl line {number})"
            ]
        self._block(f"{key} (diff vs #{previous[0]:03d}: +{added} -{removed} lines)", diff)

    def _elide_seen(self, text: str) -> str:
        """Replace multi-line blocks already printed (code, stdout, ...) with a reference."""
        for seen_text, (number, label) in self.text_seen.items():
            block = seen_text.strip()
            if "\n" in block and block in text:
                text = text.replace(block, f"[{label} from #{number:03d}]")
        return text

    def _next_event(self, number: int, event_type: str) -> int | None:
        for offset, event in enumerate(self.events[number:], start=number + 1):
            kind = event.get("event_type")
            if kind == event_type:
                return offset
            if kind != "llm_call":
                return None
        return None

    def _field(self, label: str, text: str) -> None:
        if "\n" not in text and len(text) <= WIDTH - len(label) - 6:
            self.lines.append(f"    {label}: {text}")
        else:
            self._block(label, text.rstrip("\n").split("\n"))

    def _block(self, label: str, lines: list[str]) -> None:
        self.lines.append(f"    ┌─ {label}")
        self.lines.extend(f"    │ {line}".rstrip() for line in lines)
        self.lines.append("    └─")


def _plan_table(rows: list[dict[str, Any]]) -> list[str]:
    table = [["#", "id", "layer", "category", "depends_on"]]
    for position, row in enumerate(rows, start=1):
        table.append([
            str(position),
            str(row.get("id", "")),
            str(row.get("layer", "")),
            str(row.get("category", "")),
            ", ".join(row.get("depends_on") or []) or "-",
        ])
    lines = _table(table)
    indent = " " * 5
    lines.append("")
    for position, row in enumerate(rows, start=1):
        description = str(row.get("description") or "").strip()
        wrapped = textwrap.wrap(description, WIDTH - 12) or [""]
        lines.append(f"{position:>2}. {wrapped[0]}")
        lines.extend(indent + line for line in wrapped[1:])
    return lines


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    keep = [col for col, width in enumerate(widths) if any(row[col] for row in rows[1:])]
    render = lambda row: " | ".join(row[col].ljust(widths[col]) for col in keep).rstrip()
    return [render(rows[0]), "-+-".join("-" * widths[col] for col in keep), *map(render, rows[1:])]


def _head(text: str, limit: int, number: int) -> list[str]:
    lines = text.rstrip("\n").split("\n")
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"… +{len(lines) - limit} more lines (full text: events.jsonl line {number})"]


def _llm_call_text(event: dict) -> str:
    duration = _seconds((event.get("duration_ms") or 0) / 1000)
    text = f"{duration} · {int(event.get('prompt_tokens') or 0):,} in / {int(event.get('completion_tokens') or 0):,} out tokens"
    if event.get("token_capped"):
        text += " · TOKEN CAPPED"
    if event.get("success") is False:
        text += f" · FAILED ({event.get('error_type') or 'error'})"
    return text


def _status_word(success: Any) -> str:
    if success is True:
        return "OK"
    if success is False:
        return "FAIL"
    return "running"


def _seconds(value: Any) -> str:
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return ""


def _first_line(value: Any, limit: int = 200) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, (str, int, float)) for item in value):
        return ", ".join(str(item) for item in value)
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _parse_timestamp(value: Any) -> datetime.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
