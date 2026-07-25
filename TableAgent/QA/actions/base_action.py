from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from TableAgent.schema.subtask import SubTask

@dataclass(frozen=True)
class CodeGenerationRequest:
    question: str
    subtask_id: str
    layer: str
    round_num: int
    subtask: Optional[Any] = None


@dataclass(frozen=True)
class CodeGenerationResult:
    code: str
    description: str
    reasoning: str = ""

    def as_tuple(self) -> Tuple[str, str]:
        return self.code, self.description


@dataclass(frozen=True)
class CodeExecutionRequest:
    code: str
    cell_id: Optional[str] = None


@dataclass(frozen=True)
class CodeExecutionResult:
    output: str
    error: str
    success: bool
    namespace_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanGenerationRequest:
    question: str
    table_id: Optional[str] = None
    failure_context: Optional[str] = None
    previous_plan: Optional[List[Dict[str, Any]]] = None


@dataclass(frozen=True)
class PlanGenerationResult:
    subtasks: List[SubTask]
    raw_response: str = ""


@dataclass(frozen=True)
class ReviewRequest:
    question: str
    subtask: SubTask
    code: str
    description: str
    execution: CodeExecutionResult
    round_num: int
    require_final_answer: bool = False


@dataclass(frozen=True)
class ReviewResult:
    accepted: bool
    feedback: str
    score: float = 0.0


class BaseAction(ABC):
    """Abstract base class for concrete QA actions."""
    name = ""
    desc = ""

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the action and return its typed result."""
        raise NotImplementedError


class BaseCodeGenerationAction(BaseAction):
    """
    Action that writes executable notebook code for a QA subtask.

    `generate` is kept as a compatibility adapter for older callers; new code
    should call `run(CodeGenerationRequest(...))`.
    """

    @abstractmethod
    def run(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        raise NotImplementedError

    def generate(
        self,
        question: str,
        subtask_id: str,
        layer: str,
        round_num: int,
        subtask: Optional[Any] = None,
    ) -> Tuple[str, str]:
        request = CodeGenerationRequest(
            question=question,
            subtask_id=subtask_id,
            layer=layer,
            round_num=round_num,
            subtask=subtask,
        )
        return self.run(request).as_tuple()


class BaseCodeExecutionAction(BaseAction):
    """Action that executes notebook code and returns the observation."""

    @abstractmethod
    def run(self, request: CodeExecutionRequest) -> CodeExecutionResult:
        raise NotImplementedError


class BasePlanAction(BaseAction):
    """Action that writes or repairs a QA plan."""

    @abstractmethod
    def run(self, request: PlanGenerationRequest) -> PlanGenerationResult:
        raise NotImplementedError


class BaseReviewAction(BaseAction):
    """Action that reviews whether a subtask attempt is complete."""

    @abstractmethod
    def run(self, request: ReviewRequest) -> ReviewResult:
        raise NotImplementedError
