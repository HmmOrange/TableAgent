from __future__ import annotations
from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.schema.subtask import SubTask
from TableAgent.schema.qa import AgentOutput
from TableAgent.schema.experience import ExperienceRecord
from TableAgent.QA.actions.base_action import (
    BaseCodeExecutionAction,
    BaseCodeGenerationAction,
    BaseReviewAction,
    CodeExecutionRequest,
    CodeGenerationRequest,
    ReviewRequest,
)
from TableAgent.QA.agents.base_agent import (
    BaseReActAgent,
    actionable_failure_feedback,
    code_fingerprint,
)

class TableQASynthesisAgent(BaseReActAgent):
    """
    Synthesis agent that processes inspection outputs and experience to generate synthesis code,
    executes it in the same notebook environment, and ensures `final_answer` is set.
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
            raise ValueError("A code generation action must be provided to TableQASynthesisAgent.")
        super().__init__(
            env=env,
            code_action=code_action,
            execute_action=execute_action,
            review_action=review_action,
            max_retries=max_retries,
        )

    def run_subtask(self, question: str, subtask: SubTask) -> AgentOutput:
        subtask.status = "running"
        round_num = 1
        success = False
        observation = ""
        code = ""
        description = ""
        reasoning = ""
        last_updates = {}
        failure_feedback = str(
            (subtask.metadata or {}).get("replan_failure_context", "")
        ).strip()
        attempted_code: set[str] = set()

        while round_num <= self.max_retries and not success:
            try:
                code_result = self.code_action.run(CodeGenerationRequest(
                    question=question,
                    subtask_id=subtask.id,
                    layer=subtask.layer,
                    round_num=round_num,
                    subtask=subtask,
                    failure_feedback=failure_feedback or None,
                ))
            except Exception as exc:
                code = ""
                description = "Code generation failed."
                reasoning = str(exc)
                observation = f"Code generation failed:\n{exc}"
                subtask.status = "failed"
                subtask.observation = observation
                self.env.experience_pool.add(ExperienceRecord(
                    subtask_id=subtask.id,
                    description=description,
                    code=code,
                    observation=observation,
                    reasoning=reasoning,
                    score=0.0,
                    round=round_num,
                ))
                failure_feedback = observation
                round_num += 1
                continue

            code = code_result.code
            description = code_result.description
            reasoning = code_result.reasoning
            fingerprint = code_fingerprint(code)
            if fingerprint in attempted_code:
                observation = (
                    "Equivalent synthesis retry skipped: generated code repeats a previously "
                    "rejected attempt without changing its inputs or calculation."
                )
                failure_feedback = (
                    "REPEATED_ATTEMPT: Generate materially different synthesis code that uses "
                    "the reviewer feedback, preserves the question qualifiers, and prints the "
                    "new input values, calculation, and result."
                )
                subtask.status = "failed"
                subtask.code_attempt = code
                subtask.observation = observation
                self.env.experience_pool.add(ExperienceRecord(
                    subtask_id=subtask.id,
                    description=description,
                    code=code,
                    observation=observation,
                    reasoning=reasoning,
                    score=0.0,
                    round=round_num,
                ))
                round_num += 1
                continue
            attempted_code.add(fingerprint)
            
            # 2. Observation: Execute code in the shared notebook
            execution = self.execute_action.run(CodeExecutionRequest(code=code))
            output = execution.output
            error = execution.error
            run_success = execution.success
            updates = execution.namespace_updates
            review = self.review_action.run(ReviewRequest(
                question=question,
                subtask=subtask,
                code=code,
                description=description,
                execution=execution,
                round_num=round_num,
                require_final_answer=True,
            ))
            
            # For synthesis, we also require `final_answer` to be set in the execution namespace!
            final_answer_set = "final_answer" in self.env.execution_namespace
            
            if run_success and final_answer_set and review.accepted:
                success = True
                observation = output if output else "Synthesis completed successfully."
                subtask.status = "success"
                subtask.code_attempt = code
                subtask.observation = observation
                score = review.score
                last_updates = updates
            else:
                success = False
                if not run_success:
                    observation = f"Error during synthesis execution:\n{error}"
                    failure_feedback = actionable_failure_feedback(
                        error,
                        fallback="Synthesis execution failed without details; regenerate it with explicit runtime evidence.",
                    )
                elif not review.accepted:
                    observation = f"Review rejected synthesis attempt:\n{review.feedback}"
                    failure_feedback = actionable_failure_feedback(
                        review.feedback,
                        fallback=(
                            "Reviewer rejected synthesis without a specific repair step. Recheck "
                            "the question qualifiers, input labels and values, and calculation, "
                            "then generate materially different code with printed runtime evidence."
                        ),
                    )
                else:
                    observation = "Error: synthesis completed, but 'final_answer' variable was not set in namespace."
                    failure_feedback = observation
                subtask.status = "failed"
                subtask.code_attempt = code
                subtask.observation = observation
                score = review.score
            
            # Record this attempt in experience pool
            record = ExperienceRecord(
                subtask_id=subtask.id,
                description=description,
                code=code,
                observation=observation,
                reasoning=reasoning,
                score=score,
                round=round_num
            )
            self.env.experience_pool.add(record)
            
            round_num += 1

        return AgentOutput(
            subtask_id=subtask.id,
            description=description,
            code=code,
            success=success,
            observation=observation,
            reasoning=reasoning,
            namespace_updates=last_updates,
            attempt_count=max(0, round_num - 1),
        )
