"""A sandboxed notebook: model-written code runs in one persistent namespace.

Imports are limited to data libraries, dunder access that could escape the sandbox is refused,
and every loop in a cell checks a deadline so a cell that never ends cannot hold the run.
"""

from __future__ import annotations

import ast
import builtins
import io
import sys
import time
import traceback
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

ALLOWED_MODULES = frozenset({
    "math", "statistics", "datetime", "time", "re", "json", "collections", "itertools", "functools",
    "operator", "traceback", "openpyxl", "pandas", "numpy",
})
# Imported at run time by allowed C functions (datetime.strptime, time.strptime) through the cell's __import__.
RUNTIME_MODULES = frozenset({"_strptime"})
BLOCKED_NAMES = frozenset({"__import__", "__builtins__", "__globals__", "__code__", "__subclasses__"})
CELL_TIME_LIMIT_SECONDS = 300


@dataclass
class CellRun:
    """One executed cell: its code, what it printed, and the error it raised (empty when none)."""

    label: str
    code: str
    output: str
    error: str

    @property
    def ok(self) -> bool:
        return not self.error


class Notebook:
    def __init__(self, variables: dict[str, Any]):
        self.cells: list[CellRun] = []
        self.namespace: dict[str, Any] = dict(variables)
        view = MappingProxyType(self.namespace)
        self._output = io.StringIO()
        self.namespace["__builtins__"] = {
            **_SAFE_BUILTINS, "locals": lambda: view, "globals": lambda: view, "print": self._print,
        }

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """`print` inside cells writes to this notebook's cell output; sys.stdout is shared by concurrent runs."""
        kwargs.pop("file", None)
        print(*args, file=self._output, **kwargs)

    def run(self, code: str, label: str) -> CellRun:
        """Execute `code`; `label` names the cell in the artifacts (e.g. `n1_try1_turn2`)."""
        self._output = io.StringIO()
        error = ""
        try:
            _check_code(code)
            _run_with_time_limit(code, self.namespace)
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            if not isinstance(exc, (ImportError, PermissionError, SyntaxError)):
                error = _short_traceback(exc, code)
        cell = CellRun(label, code, self._output.getvalue().strip(), error)
        self.cells.append(cell)
        return cell


def _short_traceback(exc: Exception, code: str) -> str:
    """The lines of the cell's own code that led to the error, plus the error; library frames are dropped."""
    source = code.splitlines()
    frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == "<cell>"]
    lines = [
        f"line {frame.lineno}: {source[frame.lineno - 1].strip()}"
        for frame in frames
        if frame.lineno and frame.lineno <= len(source)
    ]
    lines.append("".join(traceback.format_exception_only(type(exc), exc)).strip())
    return "\n".join(lines)


def _check_code(code: str) -> None:
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _check_module(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            _check_module(node.module)
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_NAMES:
            raise PermissionError(f"Access to attribute '{node.attr}' is not allowed.")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise PermissionError(f"Access to '{node.id}' is not allowed.")
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value in BLOCKED_NAMES:
            raise PermissionError(f"Access to key '{node.slice.value}' is not allowed.")


def _check_module(name: str) -> None:
    if name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"Import of module '{name}' is not allowed; allowed: {', '.join(sorted(ALLOWED_MODULES))}.")


def _safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    if name not in RUNTIME_MODULES:
        _check_module(name)
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_getattr(obj: Any, name: str, *default: Any) -> Any:
    if name in BLOCKED_NAMES:
        raise PermissionError(f"Access to attribute '{name}' is not allowed.")
    return getattr(obj, name, *default)


_SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable", "chr", "classmethod",
        "complex", "dict", "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset", "hasattr",
        "hash", "hex", "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
        "next", "object", "oct", "ord", "pow", "print", "property", "range", "repr", "reversed", "round", "set",
        "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip",
    )
}
_SAFE_BUILTINS.update({
    name: value for name, value in vars(builtins).items() if isinstance(value, type) and issubclass(value, BaseException)
})
_SAFE_BUILTINS["__import__"] = _safe_import
_SAFE_BUILTINS["getattr"] = _safe_getattr

# A cell runs in the caller's thread and cannot be killed, so each loop iteration of the cell's own
# code checks a deadline; library code is not monitored and runs at full speed.
_MONITOR = sys.monitoring
_MONITOR_TOOL = next((tool for tool in range(6) if _MONITOR.get_tool(tool) is None), None)
_deadlines: dict[Any, float] = {}


def _check_deadline(code: Any, offset: int, destination: int) -> None:
    deadline = _deadlines.get(code)
    if deadline is not None and destination <= offset and time.monotonic() > deadline:
        raise TimeoutError(
            f"Execution stopped after {CELL_TIME_LIMIT_SECONDS} seconds; a loop in this code does not end."
        )


if _MONITOR_TOOL is not None:
    _MONITOR.use_tool_id(_MONITOR_TOOL, "TableAgent cell time limit")
    _MONITOR.register_callback(_MONITOR_TOOL, _MONITOR.events.JUMP, _check_deadline)


def _run_with_time_limit(source: str, namespace: dict[str, Any]) -> None:
    compiled = compile(source, "<cell>", "exec")
    if _MONITOR_TOOL is None:
        exec(compiled, namespace)
        return
    codes = [compiled]
    for code in codes:
        codes.extend(const for const in code.co_consts if hasattr(const, "co_code"))
    deadline = time.monotonic() + CELL_TIME_LIMIT_SECONDS
    for code in codes:
        _deadlines[code] = deadline
        _MONITOR.set_local_events(_MONITOR_TOOL, code, _MONITOR.events.JUMP)
    try:
        exec(compiled, namespace)
    finally:
        for code in codes:
            _deadlines.pop(code, None)
            _MONITOR.set_local_events(_MONITOR_TOOL, code, 0)
