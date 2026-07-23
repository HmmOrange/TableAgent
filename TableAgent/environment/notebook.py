from __future__ import annotations
import io
import sys
import ast
import traceback
import builtins
import datetime
import json
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Tuple, Optional

try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_notebook, new_output
except Exception:  # pragma: no cover - nbformat is optional at import time.
    nbformat = None
    new_code_cell = None
    new_notebook = None
    new_output = None

@dataclass
class CellResult:
    cell_id: str
    code: str
    stdout: str
    stderr: str
    error: str
    success: bool
    namespace_updates: Dict[str, Any]

@dataclass
class ObservationView:
    cell_id: str
    success: bool
    stdout_preview: str
    stderr_preview: str
    error_preview: str
    namespace_summary: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error_truncated: bool = False
    full_output_available: bool = True

    def format(self) -> str:
        parts = [
            f"Cell: {self.cell_id}",
            f"Success: {self.success}",
            f"Namespace updates: {self.namespace_summary or 'None'}",
        ]
        if self.stdout_preview:
            suffix = " [truncated]" if self.stdout_truncated else ""
            parts.append(f"Stdout{suffix}:\n{self.stdout_preview}")
        if self.stderr_preview:
            suffix = " [truncated]" if self.stderr_truncated else ""
            parts.append(f"Stderr{suffix}:\n{self.stderr_preview}")
        if self.error_preview:
            suffix = " [truncated]" if self.error_truncated else ""
            parts.append(f"Error{suffix}:\n{self.error_preview}")
        if self.full_output_available and (self.stdout_truncated or self.stderr_truncated or self.error_truncated):
            parts.append("Full output is stored in notebook history; inspect a smaller slice or summary if needed.")
        return "\n".join(parts)

def _truncate_text(text: str, max_chars: int, *, tail: bool = False) -> tuple[str, bool]:
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = "\n...[truncated]...\n"
    marker_len = len(marker)
    if max_chars <= marker_len + 20:
        return text[:max_chars] + "...", True
    if tail:
        return marker + text[-(max_chars - marker_len):], True
    head_len = (max_chars - marker_len) // 2
    tail_len = max_chars - marker_len - head_len
    return text[:head_len] + marker + text[-tail_len:], True

def validate_code_imports(code: str) -> None:
    allowed_roots = {
        "math", "statistics", "datetime", "time", "re", "json", "collections", "itertools", "functools", "operator",
        "openpyxl", "pandas", "pd", "numpy", "np", "TableAgent",
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    for node in ast.walk(tree):
        # Reject import statements of unallowed modules
        if isinstance(node, ast.Import):
            for name in node.names:
                top_level = name.name.split('.')[0]
                if top_level not in allowed_roots:
                    raise ImportError(f"Import of module '{name.name}' is restricted.")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level = node.module.split('.')[0]
                if top_level not in allowed_roots and node.level == 0:
                    raise ImportError(f"Import from module '{node.module}' is restricted.")
        
        # Reject direct calls to __import__
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                raise PermissionError("Direct calls to '__import__' are restricted.")
            
        # Reject attributes/access named __import__, __builtins__, __globals__, __code__, __subclasses__
        if isinstance(node, ast.Attribute) and node.attr in {
            "__import__", "__builtins__", "__globals__", "__code__", "__subclasses__"
        }:
            raise PermissionError(f"Access to attribute '{node.attr}' is restricted.")

        # Reject Name __builtins__
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            raise PermissionError("Access to '__builtins__' is restricted.")

        # Reject dictionary lookups of __import__, __builtins__, __globals__, __code__, __subclasses__
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and node.slice.value in {
                "__import__", "__builtins__", "__globals__", "__code__", "__subclasses__"
            }:
                raise PermissionError(f"Access to '{node.slice.value}' key is restricted.")


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if not isinstance(name, str):
        raise TypeError("module name must be a string")
    
    top_level = name.split('.')[0]
    
    allowed_roots = {
        "math", "statistics", "datetime", "time", "re", "json", "collections", "itertools", "functools",
        "operator", "openpyxl", "pandas", "pd", "numpy", "np", "TableAgent",
    }
    
    if top_level not in allowed_roots:
        raise ImportError(f"Import of module '{name}' is restricted.")
        
    return builtins.__import__(name, globals, locals, fromlist, level)

def safe_getattr(obj, name, *args):
    if isinstance(name, str) and name in {
        "__import__", "__builtins__", "__globals__", "__code__", "__subclasses__"
    }:
        raise PermissionError(f"Access to attribute '{name}' is restricted.")
    return getattr(obj, name, *args)

# Define safe builtins
safe_builtins = {}
allowed_builtin_names = {
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray", "bytes", "callable", "chr",
    "classmethod", "complex", "dict", "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "object", "oct", "ord", "pow", "print", "property", "range", "repr", "reversed",
    "round", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type",
    "vars", "zip", "__import__", "divmod", "pow"
}
# Exception classes
exceptions = [
    ArithmeticError, AssertionError, AttributeError, BaseException, BlockingIOError, BrokenPipeError,
    BufferError, BytesWarning, ChildProcessError, ConnectionAbortedError, ConnectionError,
    ConnectionRefusedError, ConnectionResetError, DeprecationWarning, EOFError, EnvironmentError,
    Exception, FileExistsError, FileNotFoundError, FloatingPointError, FutureWarning, GeneratorExit,
    IOError, ImportError, ImportWarning, IndentationError, IndexError, InterruptedError,
    IsADirectoryError, KeyError, KeyboardInterrupt, LookupError, MemoryError, ModuleNotFoundError,
    NameError, NotADirectoryError, NotImplementedError, OSError, OverflowError, PendingDeprecationWarning,
    PermissionError, ProcessLookupError, RecursionError, ReferenceError, ResourceWarning, RuntimeError,
    RuntimeWarning, StopAsyncIteration, StopIteration, SyntaxError, SyntaxWarning, SystemError,
    SystemExit, TabError, TimeoutError, TypeError, UnboundLocalError, UnicodeDecodeError,
    UnicodeEncodeError, UnicodeError, UnicodeTranslateError, UnicodeWarning, UserWarning, ValueError,
    Warning, ZeroDivisionError
]
for name in allowed_builtin_names:
    if hasattr(builtins, name):
        safe_builtins[name] = getattr(builtins, name)
for exc in exceptions:
    safe_builtins[exc.__name__] = exc

# Inject our custom safe import and getattr
safe_builtins["__import__"] = safe_import
safe_builtins["getattr"] = safe_getattr

class Notebook:
    def __init__(
        self,
        initial_namespace: Dict[str, Any],
        max_observation_chars: int = 2000,
        max_error_chars: int = 2000,
        max_value_repr_chars: int = 800,
        max_history_size: int = 100,
    ):
        self.cells: List[CellResult] = []
        self.max_history_size = max_history_size
        self.max_observation_chars = max_observation_chars
        self.max_error_chars = max_error_chars
        self.max_value_repr_chars = max_value_repr_chars
        self.namespace = dict(initial_namespace)
        workspace_view = MappingProxyType(self.namespace)
        notebook_builtins = dict(safe_builtins)
        notebook_builtins["locals"] = lambda: workspace_view
        notebook_builtins["globals"] = lambda: workspace_view
        self.namespace["namespace"] = workspace_view
        self.namespace["__builtins__"] = notebook_builtins
        self.nb = new_notebook() if new_notebook is not None else None

    def execute_cell(self, cell_id: str, code: str) -> CellResult:
        # Validate code imports using AST
        try:
            validate_code_imports(code)
        except (ImportError, PermissionError) as e:
            result = CellResult(
                cell_id=cell_id,
                code=code,
                stdout="",
                stderr="",
                error=str(e),
                success=False,
                namespace_updates={}
            )
            self.cells.append(result)
            self._append_nb_cell(result)
            return result

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        success = True
        error_msg = ""

        # Capture pre-execution keys and identities to detect updates safely
        pre_keys = set(self.namespace.keys())
        pre_values = {k: self.namespace[k] for k in pre_keys if k != "__builtins__"}

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                # Execute in the persistent namespace
                exec(code, self.namespace)
        except Exception as e:
            success = False
            # Get traceback
            error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()

        # Capture namespace updates (ignoring special system keys and initial module definitions)
        updates = {}
        for k, v in self.namespace.items():
            if k.startswith("__"):
                continue
            if k in {"pd", "openpyxl", "env", "operators", "Cell", "CellRange", "Header", "np", "namespace"}:
                continue
            if k not in pre_keys:
                updates[k] = v
            else:
                if v is not pre_values.get(k):
                    updates[k] = v

        result = CellResult(
            cell_id=cell_id,
            code=code,
            stdout=stdout_val,
            stderr=stderr_val,
            error=error_msg,
            success=success,
            namespace_updates=updates
        )
        self.cells.append(result)
        if len(self.cells) > self.max_history_size:
            self.cells.pop(0)
        self._append_nb_cell(result)
        return result

    def _append_nb_cell(self, result: CellResult) -> None:
        if self.nb is None or new_code_cell is None or new_output is None:
            return
        cell = new_code_cell(source=result.code, metadata={"cell_id": result.cell_id})
        outputs = []
        if result.stdout:
            outputs.append(new_output(output_type="stream", name="stdout", text=result.stdout))
        if result.stderr:
            outputs.append(new_output(output_type="stream", name="stderr", text=result.stderr))
        if result.error:
            traceback_lines = result.error.splitlines()
            outputs.append(new_output(
                output_type="error",
                ename="ExecutionError",
                evalue=traceback_lines[-1] if traceback_lines else result.error,
                traceback=traceback_lines,
            ))
        cell.outputs = outputs
        cell.execution_count = len(self.cells)
        self.nb.cells.append(cell)
        if len(self.nb.cells) > self.max_history_size:
            self.nb.cells.pop(0)

    def export_ipynb(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.nb is not None and nbformat is not None:
            nbformat.write(self.nb, output_path)
            return output_path

        notebook = {
            "cells": [self._cell_to_ipynb_dict(cell, index) for index, cell in enumerate(self.cells, start=1)],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "pygments_lexer": "ipython3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def _cell_to_ipynb_dict(self, cell: CellResult, execution_count: int) -> dict[str, Any]:
        outputs = []
        if cell.stdout:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": cell.stdout,
            })
        if cell.stderr:
            outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": cell.stderr,
            })
        if cell.error:
            traceback_lines = cell.error.splitlines()
            outputs.append({
                "output_type": "error",
                "ename": "ExecutionError",
                "evalue": traceback_lines[-1] if traceback_lines else cell.error,
                "traceback": traceback_lines,
            })
        return {
            "cell_type": "code",
            "execution_count": execution_count,
            "metadata": {"cell_id": cell.cell_id},
            "outputs": outputs,
            "source": cell.code,
        }

    def observation_for_cell(self, cell: CellResult) -> ObservationView:
        stdout_preview, stdout_truncated = _truncate_text(cell.stdout, self.max_observation_chars)
        stderr_preview, stderr_truncated = _truncate_text(cell.stderr, self.max_observation_chars)
        error_preview, error_truncated = _truncate_text(cell.error, self.max_error_chars, tail=True)
        namespace_summary = self.summarize_updates(cell.namespace_updates)
        return ObservationView(
            cell_id=cell.cell_id,
            success=cell.success,
            stdout_preview=stdout_preview,
            stderr_preview=stderr_preview,
            error_preview=error_preview,
            namespace_summary=namespace_summary,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            error_truncated=error_truncated,
            full_output_available=True,
        )

    def summarize_updates(self, updates: Dict[str, Any]) -> str:
        if not updates:
            return ""
        parts = []
        for name, value in updates.items():
            parts.append(f"{name}: {self.summarize_value(value)}")
        return "; ".join(parts)

    def summarize_value(self, value: Any, *, max_chars: Optional[int] = None) -> str:
        max_chars = self.max_value_repr_chars if max_chars is None else max_chars
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None

        if pd is not None:
            if isinstance(value, pd.DataFrame):
                cols = [str(c) for c in value.columns[:8]]
                suffix = ", ..." if len(value.columns) > 8 else ""
                return f"DataFrame(shape={value.shape}, columns=[{', '.join(cols)}{suffix}])"
            if isinstance(value, pd.Series):
                return f"Series(len={len(value)}, name={value.name!r}, dtype={value.dtype})"

        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None:
            return f"{type(value).__name__}(shape={shape}, dtype={dtype})"
        if isinstance(value, dict):
            keys = list(value.keys())
            return f"dict(len={len(value)}, keys={keys[:8]}{'...' if len(keys) > 8 else ''})"
        if isinstance(value, (list, tuple, set, frozenset)):
            preview = list(value)[:5]
            return f"{type(value).__name__}(len={len(value)}, preview={preview!r}{'...' if len(value) > 5 else ''})"

        text = repr(value)
        text, truncated = _truncate_text(text, max_chars)
        return text + (" [truncated]" if truncated else "")

    def preview_variable(self, name: str, rows: int = 5, max_chars: Optional[int] = None) -> str:
        if name not in self.namespace:
            raise KeyError(f"Variable '{name}' is not defined in notebook namespace.")
        value = self.namespace[name]
        max_chars = self.max_value_repr_chars if max_chars is None else max_chars
        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None
        if pd is not None and isinstance(value, pd.DataFrame):
            text = f"{self.summarize_value(value)}\n{value.head(rows).to_string()}"
            return _truncate_text(text, max_chars)[0]
        if pd is not None and isinstance(value, pd.Series):
            text = f"{self.summarize_value(value)}\n{value.head(rows).to_string()}"
            return _truncate_text(text, max_chars)[0]
        if isinstance(value, (list, tuple)):
            text = f"{self.summarize_value(value)}\npreview={list(value)[:rows]!r}"
            return _truncate_text(text, max_chars)[0]
        return self.summarize_value(value, max_chars=max_chars)

    def get_history(
        self,
        last_n: Optional[int] = None,
        include_output: bool = True,
        max_code_len: int = 1000,
        max_output_len: int = 800,
        only_success: Optional[bool] = None,
    ) -> str:
        records = self.cells
        if only_success is not None:
            records = [cell for cell in records if cell.success == only_success]
        if last_n is not None:
            records = records[-last_n:]
        if not records:
            return "No notebook history."

        chunks = []
        for cell in records:
            code, code_truncated = _truncate_text(cell.code, max_code_len)
            status = "SUCCESS" if cell.success else "ERROR"
            chunk = [
                f"Cell {cell.cell_id} | {status}",
                "Code" + (" [truncated]:" if code_truncated else ":"),
                code,
                f"Namespace updates: {self.summarize_updates(cell.namespace_updates) or 'None'}",
            ]
            if include_output:
                stdout, stdout_truncated = _truncate_text(cell.stdout, max_output_len)
                stderr, stderr_truncated = _truncate_text(cell.stderr, max_output_len)
                error, error_truncated = _truncate_text(cell.error, max_output_len, tail=True)
                if stdout:
                    chunk.append("Stdout" + (" [truncated]:" if stdout_truncated else ":"))
                    chunk.append(stdout)
                if stderr:
                    chunk.append("Stderr" + (" [truncated]:" if stderr_truncated else ":"))
                    chunk.append(stderr)
                if error:
                    chunk.append("Error" + (" [truncated]:" if error_truncated else ":"))
                    chunk.append(error)
            chunks.append("\n".join(chunk))
        return "\n\n".join(chunks)

    def get_cell_output(self, cell_id: str, max_chars: Optional[int] = None) -> str:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                output = cell.stdout
                if cell.stderr:
                    output += ("\n" if output else "") + "Stderr:\n" + cell.stderr
                if cell.error:
                    output += ("\n" if output else "") + "Error:\n" + cell.error
                if max_chars is not None:
                    output = _truncate_text(output, max_chars)[0]
                return output
        raise KeyError(f"Cell '{cell_id}' not found in notebook history.")
