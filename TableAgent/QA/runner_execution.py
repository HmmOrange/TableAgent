from __future__ import annotations

import time
from typing import Any

from TableAgent.schema.qa import AgentOutput, QAResult
from TableAgent.schema.subtask import SubTask


class QAExecutionMixin:
    """Plan and execute QA subtasks for ``TableQARunner``."""

    def run(self, question: str) -> QAResult:
        event_start_index = len(self.env.logger.events)
        run_id = self._make_run_id(question)
        run_dir = self.qa_artifact_root / run_id
        self.env.logger.log_event("run_start", {"question": question})
        self.env.logger.log_event(
            "run_artifact_start",
            {"run_id": run_id, "artifact_dir": str(run_dir)},
        )
        self._progress(f"[qa] run start | artifact_dir={run_dir}")
        start_time = time.time()

        all_table_ids = self.env.operators.list_tables()
        table_id = self.table_id
        if not table_id and len(all_table_ids) == 1:
            table_id = all_table_ids[0]
            self.env.logger.log_event(
                "single_table_auto_selected", {"table_id": table_id}
            )
        self.env.execution_namespace["all_table_ids"] = all_table_ids
        self.env.execution_namespace["selected_table_ids"] = (
            [table_id] if table_id else []
        )
        if table_id:
            self._set_active_tables([table_id])

        replan_count = 0
        planning_failure = None
        while True:
            try:
                plan = self.planner.plan(
                    question,
                    table_id=table_id,
                    failure_context=planning_failure,
                    previous_plan=[],
                )
                self._progress(
                    "[qa] planning done | subtasks="
                    f"{[(subtask.id, subtask.layer) for subtask in plan]}"
                )
                break
            except Exception as exc:
                if replan_count >= self.max_replans:
                    plan = []
                    planning_error = exc
                    break
                replan_count += 1
                planning_failure = f"Planning attempt failed: {exc}"
                self.env.logger.log_event(
                    "replanning_start",
                    {
                        "attempt": replan_count,
                        "max_replans": self.max_replans,
                        "error": planning_failure,
                        "previous_plan": [],
                    },
                )

        if not plan:
            execution_time = time.time() - start_time
            error_msg = f"Planning failed: {planning_error}"
            result = QAResult(
                question=question,
                plan=[],
                subtask_outputs=[],
                final_answer=None,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                token_usage=self._token_usage(),
                replan_count=replan_count,
                subtask_retry_count=0,
                qa_max_retries=self.max_retries,
                llm_calls=self._llm_call_metrics(),
            )
            self.env.logger.log_event(
                "run_complete",
                {
                    "success": False,
                    "final_answer": None,
                    "error": error_msg,
                    "execution_time": execution_time,
                },
            )
            result.logs = self.env.logger.events
            self._persist_run_artifacts(result, run_dir, event_start_index)
            return result

        subtask_outputs = []
        success = False
        error_msg = None
        final_answer = None
        baseline_namespace = dict(self.env.execution_namespace)
        execution_attempt = 0

        while True:
            if execution_attempt:
                self.env.execution_namespace.clear()
                self.env.execution_namespace.update(baseline_namespace)
            execution_attempt += 1
            final_answer = None
            self.env.execution_namespace.pop("final_answer", None)
            try:
                execution_plan = self._topological_sort(plan)
                attempt_outputs, success, error_msg = self._execute_plan(
                    question, plan, execution_plan
                )
                subtask_outputs.extend(attempt_outputs)
            except ValueError as exc:
                success = False
                error_msg = str(exc)
                attempt_outputs = []

            if success:
                final_answer = self._final_answer(execution_plan, plan)
                if final_answer is None:
                    success = False
                    error_msg = (
                        "Synthesis layer completed, but 'final_answer' variable was not "
                        "set in namespace."
                    )
                elif (
                    self.enable_final_answer_review
                    and not self._is_pure_common_info_plan(plan)
                ):
                    final_review = self.final_answer_review.run(
                        question=question,
                        plan=plan,
                        outputs=attempt_outputs,
                        final_answer=final_answer,
                    )
                    self.env.logger.log_event(
                        "final_answer_review",
                        {
                            "accepted": final_review.accepted,
                            "score": final_review.score,
                            "feedback": final_review.feedback,
                        },
                    )
                    if not final_review.accepted:
                        success = False
                        error_msg = (
                            "Final answer review rejected the plan: "
                            f"{final_review.feedback}"
                        )

            if success or replan_count >= self.max_replans:
                break

            replan_count += 1
            failure_context = self._replanning_context(error_msg, attempt_outputs)
            self._progress(
                f"[qa] replanning start | attempt={replan_count}/{self.max_replans} "
                f"| error={error_msg}"
            )
            self.env.logger.log_event(
                "replanning_start",
                {
                    "attempt": replan_count,
                    "max_replans": self.max_replans,
                    "error": error_msg,
                    "previous_plan": self._plan_payload(plan),
                },
            )
            try:
                plan = self.planner.plan(
                    question,
                    table_id=table_id,
                    failure_context=failure_context,
                    previous_plan=self._plan_payload(plan),
                )
            except Exception as exc:
                error_msg = (
                    f"Replanning failed after execution error ({error_msg}): {exc}"
                )
                self.env.logger.log_event(
                    "replanning_error",
                    {"attempt": replan_count, "error": str(exc)},
                )
                break
            for subtask in plan:
                if not subtask.metadata:
                    subtask.metadata = {}
                subtask.metadata["replan_failure_context"] = failure_context
            self.env.logger.log_event(
                "replanning_complete",
                {"attempt": replan_count, "subtasks": self._plan_payload(plan)},
            )
            self._progress(
                "[qa] replanning done | subtasks="
                f"{[(subtask.id, subtask.layer) for subtask in plan]}"
            )

        execution_time = time.time() - start_time
        result = QAResult(
            question=question,
            plan=plan,
            subtask_outputs=subtask_outputs,
            final_answer=final_answer,
            success=success,
            error=error_msg,
            execution_time=execution_time,
            token_usage=self._token_usage(),
            subtask_retry_count=sum(
                max(0, int(getattr(output, "attempt_count", 1) or 1) - 1)
                for output in subtask_outputs
            ),
            qa_max_retries=self.max_retries,
            llm_calls=self._llm_call_metrics(),
        )
        result.replan_count = replan_count
        result.logs = self.env.logger.events

        self.env.logger.log_event(
            "run_complete",
            {
                "success": success,
                "final_answer": final_answer,
                "error": error_msg,
                "execution_time": execution_time,
                "replan_count": replan_count,
            },
        )
        self._persist_run_artifacts(result, run_dir, event_start_index)
        self._progress(f"[qa] run done | success={success} | artifact_dir={run_dir}")
        return result

    def _execute_plan(
        self,
        question: str,
        plan: list[SubTask],
        execution_plan: list[SubTask],
    ) -> tuple[list[Any], bool, str | None]:
        subtasks_by_id = {subtask.id: subtask for subtask in plan}
        accepted_updates: dict[str, tuple[str, ...]] = {}
        outputs = []
        completed: set[str] = set()
        self.env.logger.log_event(
            "execution_plan",
            {
                "order": [subtask.id for subtask in execution_plan],
                "dependencies": {
                    subtask.id: subtask.depends_on for subtask in execution_plan
                },
            },
        )

        for subtask in execution_plan:
            missing_deps = [
                dependency
                for dependency in subtask.depends_on
                if dependency not in completed
            ]
            if missing_deps:
                return (
                    outputs,
                    False,
                    f"Subtask '{subtask.id}' has unfinished dependencies: {missing_deps}",
                )

            self._progress(
                f"[qa] subtask start | id={subtask.id} | layer={subtask.layer} "
                f"| category={subtask.category}"
            )
            self.env.logger.log_event(
                "subtask_start",
                {
                    "subtask_id": subtask.id,
                    "layer": subtask.layer,
                    "category": subtask.category,
                    "description": subtask.description,
                    "depends_on": subtask.depends_on,
                },
            )

            if subtask.layer == "synthesis":
                if not subtask.metadata:
                    subtask.metadata = {}
                subtask.metadata["dependency_variables"] = self._dependency_variables(
                    subtask, subtasks_by_id, accepted_updates
                )

            try:
                output = self._run_subtask(question, subtask)
            except Exception as exc:
                output = AgentOutput(
                    subtask_id=subtask.id,
                    description=subtask.description,
                    code=subtask.code_attempt or "",
                    success=False,
                    observation=f"Unhandled subtask error: {exc}",
                    reasoning=(
                        "The subtask raised outside its normal execution/review loop."
                    ),
                )
                self.env.logger.log_event(
                    "subtask_exception",
                    {"subtask_id": subtask.id, "error": str(exc)},
                )

            self._progress(
                f"[qa] subtask done | id={subtask.id} | success={output.success} | "
                f"updates={list(output.namespace_updates.keys())}"
            )
            output.layer = subtask.layer
            output.category = subtask.category
            outputs.append(output)
            selected_exp = self.env.experience_pool.select()
            self.env.logger.log_event(
                "subtask_complete",
                {
                    "subtask_id": subtask.id,
                    "success": output.success,
                    "observation": output.observation,
                    "code": output.code,
                    "namespace_updates": list(output.namespace_updates.keys()),
                    "experience_count": len(selected_exp),
                },
            )
            if not output.success:
                return (
                    outputs,
                    False,
                    f"Failed at subtask '{subtask.id}': {output.observation}",
                )
            accepted_updates[subtask.id] = tuple(output.namespace_updates.keys())
            completed.add(subtask.id)

        return outputs, True, None

    def _run_subtask(self, question: str, subtask: SubTask) -> AgentOutput:
        if subtask.category == "common_info":
            try:
                return self.common_info_action.run(question, subtask)
            except Exception:
                if self.qa_common_info_fallback != "normal":
                    raise
                if subtask.layer == "synthesis":
                    return self.synthesis_agent.run_subtask(question, subtask)
                return self.agent.run_subtask(question, subtask)
        if subtask.layer == "synthesis":
            return self.synthesis_agent.run_subtask(question, subtask)
        if subtask.layer == "inspect":
            selected_table_ids = self._selected_table_ids()
            if not selected_table_ids:
                selected_table_ids = [self.env.default_table_id()]
                self._set_active_tables(selected_table_ids)
            if not subtask.metadata:
                subtask.metadata = {}
            subtask.metadata.setdefault("table_ids", selected_table_ids)
            subtask.metadata.setdefault("table_id", selected_table_ids[0])
        output = self.agent.run_subtask(question, subtask)
        if output.success and subtask.layer == "table_inspect":
            selected_table_ids = self._selected_table_ids() or [
                self.env.default_table_id()
            ]
            self._set_active_tables(selected_table_ids)
        return output

    @staticmethod
    def _dependency_variables(
        subtask: SubTask,
        subtasks_by_id: dict[str, SubTask],
        accepted_updates: dict[str, tuple[str, ...]],
    ) -> list[str]:
        dependency_ids: list[str] = []
        visited: set[str] = set()

        def visit(subtask_id: str) -> None:
            if subtask_id in visited:
                return
            visited.add(subtask_id)
            dependency = subtasks_by_id.get(subtask_id)
            if dependency is None:
                return
            for parent_id in dependency.depends_on:
                visit(parent_id)
            dependency_ids.append(subtask_id)

        for dependency_id in subtask.depends_on:
            visit(dependency_id)

        names: list[str] = []
        for dependency_id in dependency_ids:
            for name in accepted_updates.get(dependency_id, ()):
                if name != "final_answer" and name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _plan_payload(plan: list[SubTask]) -> list[dict[str, Any]]:
        return [
            {
                "id": subtask.id,
                "layer": subtask.layer,
                "category": subtask.category,
                "depends_on": list(subtask.depends_on),
                "description": subtask.description,
                "metadata": dict(subtask.metadata or {}),
            }
            for subtask in plan
        ]

    @staticmethod
    def _replanning_context(error_msg: str | None, outputs: list[Any]) -> str:
        recent = []
        for output in outputs[-4:]:
            observation = str(getattr(output, "observation", "") or "")
            if len(observation) > 2000:
                observation = observation[:2000] + "\n...[truncated]"
            recent.append(
                f"- subtask={getattr(output, 'subtask_id', '')} "
                f"success={getattr(output, 'success', False)}\n"
                f"  observation={observation}"
            )
        evidence = "\n".join(recent) or "No subtask output was produced."
        return (
            f"Failure: {error_msg or 'Unknown execution failure'}\n\n"
            f"Recent runtime evidence:\n{evidence}"
        )

    @staticmethod
    def _topological_sort(plan: list[SubTask]) -> list[SubTask]:
        by_id = {}
        for subtask in plan:
            if subtask.id in by_id:
                raise ValueError(f"Duplicate subtask id in plan: {subtask.id}")
            by_id[subtask.id] = subtask

        for subtask in plan:
            missing = [dep for dep in subtask.depends_on if dep not in by_id]
            if missing:
                raise ValueError(
                    f"Subtask '{subtask.id}' depends on unknown subtasks: {missing}"
                )

        ordered: list[SubTask] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(subtask_id: str) -> None:
            if subtask_id in permanent:
                return
            if subtask_id in temporary:
                raise ValueError(
                    f"Cycle detected in subtask dependencies at '{subtask_id}'"
                )
            temporary.add(subtask_id)
            for dep_id in by_id[subtask_id].depends_on:
                visit(dep_id)
            temporary.remove(subtask_id)
            permanent.add(subtask_id)
            ordered.append(by_id[subtask_id])

        for subtask in plan:
            visit(subtask.id)
        return ordered
