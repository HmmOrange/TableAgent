from __future__ import annotations
from typing import List, Optional, Any

from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.schema.subtask import SubTask
from TableAgent.QA.actions.base_action import PlanGenerationRequest
from TableAgent.QA.actions.write_plan import WriteQAPlanAction

class TableQAPlanner:
    """
    Planner that analyzes a user question and loaded table structures to build a two-layer execution plan:
    1. Field inspect layer: Extracts/selects fields, ranges, and data areas.
    2. Synthesis layer: Aggregates previous results and writes executable code to compute the final answer.
    """
    def __init__(self, env: QAEnvironment, llm_client: Optional[Any] = None):
        self.env = env
        self.llm_client = llm_client
        self.write_plan_action = WriteQAPlanAction(env, llm_client=llm_client)

    def plan(
        self,
        question: str,
        table_id: Optional[str] = None,
        *,
        failure_context: str | None = None,
        previous_plan: list[dict[str, Any]] | None = None,
    ) -> List[SubTask]:
        result = self.write_plan_action.run(PlanGenerationRequest(
            question=question,
            table_id=table_id,
            failure_context=failure_context,
            previous_plan=previous_plan,
        ))
        return result.subtasks
