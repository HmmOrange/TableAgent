from __future__ import annotations
from abc import ABC, abstractmethod

from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.schema.subtask import SubTask
from TableAgent.schema.qa import AgentOutput
from TableAgent.QA.actions.base_action import BaseCodeExecutionAction, BaseCodeGenerationAction, BaseReviewAction
from TableAgent.QA.actions.execute_notebook import ExecuteNotebookCodeAction
from TableAgent.QA.actions.review import ReviewSubtaskAction

class BaseReActAgent(ABC):
    """
    Abstract base class for ReAct-style agents.
    Provides shared attributes and defines the contract for executing subtasks.
    """

    def __init__(
        self,
        env: QAEnvironment,
        code_action: BaseCodeGenerationAction | None = None,
        execute_action: BaseCodeExecutionAction | None = None,
        review_action: BaseReviewAction | None = None,
        max_retries: int = 3,
        policy: BaseCodeGenerationAction | None = None,
    ):
        code_action = code_action or policy
        if code_action is None:
            raise ValueError("A code generation action must be provided.")
        self.env = env
        self.code_action = code_action
        self.execute_action = execute_action or ExecuteNotebookCodeAction(env)
        self.review_action = review_action or ReviewSubtaskAction(env)
        self.policy = code_action
        self.max_retries = max_retries

    @abstractmethod
    def run_subtask(self, question: str, subtask: SubTask) -> AgentOutput:
        """
        Executes a single subtask within the ReAct loop:
        Reasoning/Action -> Observation -> Decision (success/retry/fail)
        """
        pass
