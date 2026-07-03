from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from TableAgent.schema.subtask import SubTask

@dataclass
class AgentOutput:
    subtask_id: str
    description: str
    code: str
    success: bool
    observation: str
    reasoning: str = ""
    namespace_updates: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QAResult:
    question: str
    plan: List[SubTask]
    subtask_outputs: List[AgentOutput] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    execution_time: float = 0.0
    artifacts: Dict[str, str] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"QAResult(success={self.success}, final_answer='{self.final_answer}')"
