from __future__ import annotations

import ast
from typing import Any, Optional

from TableAgent.stages.qa.actions.base_action import (
    BaseCodeExecutionAction,
    CodeExecutionRequest,
    CodeExecutionResult,
)


def positional_cell_access(code: str) -> list[str]:
    """Find `.iloc[<int>, <int>]` -- addressing a cell by fixed physical position.

    A single positional index such as `.iloc[0]` after a filter is ordinary and stays
    unflagged. Two constant indices mean the code is asserting where a field physically
    sits, which the verified structure is there to answer instead. This reports rather
    than blocks: a collapsed or malformed table sometimes leaves no other way in.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not (isinstance(value, ast.Attribute) and value.attr in {"iloc", "values"}):
            continue
        index = node.slice
        if not isinstance(index, ast.Tuple) or len(index.elts) != 2:
            continue
        if all(isinstance(element, ast.Constant) and isinstance(element.value, int)
               for element in index.elts):
            row, column = (element.value for element in index.elts)
            findings.append(f".{value.attr}[{row}, {column}]")
    return findings


class ExecuteNotebookCodeAction(BaseCodeExecutionAction):
    """Action that executes generated Python code in the shared QA notebook."""
    name = "execute_notebook_code"
    desc = "Run Python code in the QAEnvironment notebook and return a compact observation."

    def __init__(self, env: Optional[Any] = None):
        self.env = env

    def run(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        if not self.env:
            raise ValueError("Environment not set on ExecuteNotebookCodeAction.")
        output, error, success, updates = self.env.execute_code(
            request.code,
            cell_id=request.cell_id,
        )
        positional = positional_cell_access(request.code)
        if positional and success:
            output = (
                f"{output}\n[structure advisory] This cell addressed cells by fixed "
                f"physical position ({', '.join(dict.fromkeys(positional))}). Positions are "
                "not verified evidence: confirm the value belongs to the requested header "
                "with operators.get_header/resolve_header_columns before relying on it."
            ).strip()
        return CodeExecutionResult(
            output=output,
            error=error,
            success=success,
            namespace_updates=updates,
        )

