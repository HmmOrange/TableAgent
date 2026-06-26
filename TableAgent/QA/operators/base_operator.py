from __future__ import annotations
from abc import ABC
from typing import Any

class BaseOperator(ABC):
    """Base class for specialized spreadsheet operators."""
    name = "base"
    description = "Base spreadsheet operator."
    examples: tuple[str, ...] = ()

    def __init__(self, env: Any):
        self.env = env

    @classmethod
    def describe(cls) -> str:
        lines = [f"{cls.name}: {cls.description}"]
        lines.extend(f"- `{example}`" for example in cls.examples)
        return "\n".join(lines)

if __name__ == "__main__":
    print("BaseOperator is abstract. Run a concrete module, e.g. `python -m TableAgent.QA.operators.table_operator`.")
