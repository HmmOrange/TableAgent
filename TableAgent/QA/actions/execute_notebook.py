from __future__ import annotations

from typing import Any, Optional

from TableAgent.QA.actions.base_action import (
    BaseCodeExecutionAction,
    CodeExecutionRequest,
    CodeExecutionResult,
)


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
        return CodeExecutionResult(
            output=output,
            error=error,
            success=success,
            namespace_updates=updates,
        )

