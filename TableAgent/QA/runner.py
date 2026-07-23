from __future__ import annotations
import datetime
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any, List

from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.QA.agents.planner import TableQAPlanner
from TableAgent.QA.agents.react_agent import TableQAAgent
from TableAgent.QA.agents.synthesis_agent import TableQASynthesisAgent
from TableAgent.QA.actions.base_action import BaseCodeGenerationAction
from TableAgent.QA.actions.common_info import CommonInfoSubtaskAction
from TableAgent.QA.actions.execute_notebook import ExecuteNotebookCodeAction
from TableAgent.QA.actions.review import ReviewSubtaskAction
from TableAgent.QA.actions.review_final_answer import ReviewFinalAnswerAction
from TableAgent.schema.qa import AgentOutput, QAResult
from TableAgent.schema.subtask import SubTask

if TYPE_CHECKING:
    from TableAgent.pipeline.retrieval import TableRetrieverContract


class TokenCountingLLM:
    """Proxy an LLM client while accumulating token usage from its responses."""
    def __init__(self, client: Any):
        self.client = client
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Any:
        response = self.client.generate(prompt, system_prompt=system_prompt)
        self.prompt_tokens += int(getattr(response, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(response, "completion_tokens", 0) or 0)
        return response

    def token_usage(self) -> dict[str, int]:
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
        }


class TableQARunner:
    """
    High-level orchestrator that coordinates the QA workflow:
    1. Loads structure and workbook into the QAEnvironment.
    2. Uses TableQAPlanner to generate a 2-layer subtask plan.
    3. Runs each subtask via TableQAAgent (inspect) or TableQASynthesisAgent (synthesis) within the shared environment.
    4. Aggregates results, persists event logs, and extracts the final computed answer.
    """
    def __init__(
        self,
        structure_path: str,
        workbook_path: str,
        llm_client: Optional[Any] = None,
        config: Optional[dict] = None,
        code_action: Optional[BaseCodeGenerationAction] = None,
        policy: Optional[BaseCodeGenerationAction] = None,
        max_experience_records: int = 5,
        max_retries: int = 3,
        table_retriever: TableRetrieverContract | None = None,
        related_structure_paths: Optional[list[str | Path]] = None,
    ):
        raw_config = config or {}
        self.settings = raw_config.get("table_agent", raw_config) if isinstance(raw_config, dict) else {}
        
        # Load parameters from configuration/settings if present
        actual_max_retries = max_retries
        actual_max_replans = 5
        actual_max_records = max_experience_records
        log_path = None
        artifact_root = Path("logs") / "qa_runs"
        max_observation_chars = 2000
        max_error_chars = 2000
        max_value_repr_chars = 800
        
        if self.settings:
            actual_max_retries = int(self.settings.get("qa_max_retries", max_retries))
            actual_max_replans = int(self.settings.get("qa_max_replans", 5))
            actual_max_records = int(self.settings.get("qa_max_experience_records", max_experience_records))
            log_path_val = self.settings.get("qa_log_path")
            if log_path_val:
                log_path = str(log_path_val)
                artifact_root = Path(log_path).parent / "qa_runs"
            max_observation_chars = int(self.settings.get("qa_max_observation_chars", max_observation_chars))
            max_error_chars = int(self.settings.get("qa_max_error_chars", max_error_chars))
            max_value_repr_chars = int(self.settings.get("qa_max_value_repr_chars", max_value_repr_chars))

        if config and isinstance(config, dict):
            agent_cfg = config.get("table_agent", {}) if isinstance(config.get("table_agent", {}), dict) else {}
            explicit_artifact_root = config.get("qa_artifact_dir") or agent_cfg.get("qa_artifact_dir")
            if explicit_artifact_root:
                artifact_root = Path(str(explicit_artifact_root))
        self.console_progress = bool(config.get("qa_console_progress", False)) if isinstance(config, dict) else False
        self.enable_final_answer_review = bool(self.settings.get("qa_final_answer_review", False))
        self.env_plan_category_review = bool(self.settings.get("qa_plan_category_review", False))
        self.qa_artifact_root = artifact_root
        self.max_replans = max(0, actual_max_replans)

        self.env = QAEnvironment(
            structure_path=structure_path,
            workbook_path=workbook_path,
            max_experience_records=actual_max_records,
            log_path=log_path,
            max_observation_chars=max_observation_chars,
            max_error_chars=max_error_chars,
            max_value_repr_chars=max_value_repr_chars,
            table_retriever=table_retriever,
            related_structure_paths=related_structure_paths,
        )
        self.env.excluded_sheet_names = {
            str(name).strip().casefold()
            for name in self.settings.get("qa_excluded_sheet_names", [])
            if str(name).strip()
        }
        self.env.enable_plan_category_review = self.env_plan_category_review
        
        self.env.logger.log_event("config_loaded", {
            "max_retries": actual_max_retries,
            "max_replans": self.max_replans,
            "max_experience_records": actual_max_records,
            "log_path": log_path,
            "max_observation_chars": max_observation_chars,
            "max_error_chars": max_error_chars,
            "max_value_repr_chars": max_value_repr_chars,
        })
        
        self.llm_client = TokenCountingLLM(llm_client) if llm_client is not None else None
        self.planner = TableQAPlanner(self.env, llm_client=self.llm_client)
        
        # Store table_id if provided in config
        self.table_id = None
        if config and isinstance(config, dict):
            self.table_id = config.get("table_id")
            if not self.table_id and "table_agent" in config and isinstance(config["table_agent"], dict):
                self.table_id = config["table_agent"].get("table_id")

        # Initialize default code generation action if none was provided.
        code_action = code_action or policy
        if code_action is None:
            if self.llm_client is None:
                raise ValueError("Either llm_client or code_action must be provided to TableQARunner.")
            from TableAgent.QA.actions.llm_code_generation import LLMCodeGenerationAction
            code_action = LLMCodeGenerationAction(self.llm_client, self.env)
        else:
            # Set the env on compatible actions.
            if hasattr(code_action, "env") or hasattr(code_action, "__dict__"):
                try:
                    code_action.env = self.env
                except Exception:
                    pass
            if self.llm_client is not None and getattr(code_action, "llm_client", None) is llm_client:
                try:
                    code_action.llm_client = self.llm_client
                except Exception:
                    pass

        execute_action = ExecuteNotebookCodeAction(self.env)
        review_action = ReviewSubtaskAction(self.env, llm_client=self.llm_client)
        self.agent = TableQAAgent(
            self.env,
            code_action=code_action,
            execute_action=execute_action,
            review_action=review_action,
            max_retries=actual_max_retries,
        )
        self.synthesis_agent = TableQASynthesisAgent(
            self.env,
            code_action=code_action,
            execute_action=execute_action,
            review_action=review_action,
            max_retries=actual_max_retries,
        )
        self.common_info_action = CommonInfoSubtaskAction(self.env, llm_client=self.llm_client)
        self.final_answer_review = ReviewFinalAnswerAction(self.env, llm_client=self.llm_client)

    def run(self, question: str) -> QAResult:
        event_start_index = len(self.env.logger.events)
        run_id = self._make_run_id(question)
        run_dir = self.qa_artifact_root / run_id
        self.env.logger.log_event("run_start", {"question": question})
        self.env.logger.log_event("run_artifact_start", {
            "run_id": run_id,
            "artifact_dir": str(run_dir),
        })
        self._progress(f"[qa] run start | artifact_dir={run_dir}")
        start_time = time.time()
        
        table_id = self.table_id
        all_table_ids = self.env.operators.list_tables()
        self.env.execution_namespace["all_table_ids"] = all_table_ids
        self.env.execution_namespace["selected_table_ids"] = [table_id] if table_id else []
        if table_id:
            self._set_active_tables([table_id])
            
        # 1. Plan, retrying malformed or failed planning through the same bounded replanning budget.
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
                    f"{[(subtask.id, subtask.layer, subtask.category) for subtask in plan]}"
                )
                break
            except Exception as exc:
                if replan_count >= self.max_replans:
                    plan = []
                    planning_error = exc
                    break
                replan_count += 1
                planning_failure = f"Planning attempt failed: {exc}"
                self.env.logger.log_event("replanning_start", {
                    "attempt": replan_count,
                    "max_replans": self.max_replans,
                    "error": planning_failure,
                    "previous_plan": [],
                })

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
            )
            self.env.logger.log_event("run_complete", {
                "success": False,
                "final_answer": None,
                "error": error_msg,
                "execution_time": execution_time,
            })
            result.logs = self.env.logger.events
            self._persist_run_artifacts(result, run_dir, event_start_index)
            return result
        
        # 2. Execute the plan, asking the LLM to produce a corrected complete plan after bounded failures.
        subtask_outputs = []
        success = False
        error_msg = None
        final_answer = None
        execution_plan: list[SubTask] = []
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
                attempt_outputs, success, error_msg = self._execute_plan(question, plan, execution_plan)
                subtask_outputs.extend(attempt_outputs)
            except ValueError as exc:
                success = False
                error_msg = str(exc)
                attempt_outputs = []

            if success:
                final_answer = self._final_answer(execution_plan, plan)
                if final_answer is None:
                    success = False
                    error_msg = "Synthesis layer completed, but 'final_answer' variable was not set in namespace."
                elif self.enable_final_answer_review and not self._is_pure_common_info_plan(plan):
                    final_review = self.final_answer_review.run(
                        question=question,
                        plan=plan,
                        outputs=attempt_outputs,
                        final_answer=final_answer,
                    )
                    self.env.logger.log_event("final_answer_review", {
                        "accepted": final_review.accepted,
                        "score": final_review.score,
                        "feedback": final_review.feedback,
                    })
                    if not final_review.accepted:
                        success = False
                        error_msg = f"Final answer review rejected the plan: {final_review.feedback}"

            if success or replan_count >= self.max_replans:
                break

            replan_count += 1
            failure_context = self._replanning_context(error_msg, attempt_outputs)
            self._progress(
                f"[qa] replanning start | attempt={replan_count}/{self.max_replans} | error={error_msg}"
            )
            self.env.logger.log_event("replanning_start", {
                "attempt": replan_count,
                "max_replans": self.max_replans,
                "error": error_msg,
                "previous_plan": self._plan_payload(plan),
            })
            try:
                plan = self.planner.plan(
                    question,
                    table_id=table_id,
                    failure_context=failure_context,
                    previous_plan=self._plan_payload(plan),
                )
            except Exception as exc:
                error_msg = f"Replanning failed after execution error ({error_msg}): {exc}"
                self.env.logger.log_event("replanning_error", {
                    "attempt": replan_count,
                    "error": str(exc),
                })
                break
            self.env.logger.log_event("replanning_complete", {
                "attempt": replan_count,
                "subtasks": self._plan_payload(plan),
            })
            self._progress(
                "[qa] replanning done | subtasks="
                f"{[(subtask.id, subtask.layer, subtask.category) for subtask in plan]}"
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
        )
        result.replan_count = replan_count

        # Expose logs/events on the result object
        result.logs = self.env.logger.events

        self.env.logger.log_event("run_complete", {
            "success": success,
            "final_answer": final_answer,
            "error": error_msg,
            "execution_time": execution_time,
            "replan_count": replan_count,
        })

        self._persist_run_artifacts(result, run_dir, event_start_index)
        self._progress(f"[qa] run done | success={success} | artifact_dir={run_dir}")

        return result

    def _execute_plan(
        self,
        question: str,
        plan: list[SubTask],
        execution_plan: list[SubTask],
    ) -> tuple[list[Any], bool, str | None]:
        mixed_synthesis_ids = self._mixed_synthesis_ids(plan)
        subtasks_by_id = {subtask.id: subtask for subtask in plan}
        accepted_updates: dict[str, tuple[str, ...]] = {}
        outputs = []
        completed: set[str] = set()
        self.env.logger.log_event("execution_plan", {
            "order": [subtask.id for subtask in execution_plan],
            "dependencies": {subtask.id: subtask.depends_on for subtask in execution_plan},
        })

        for subtask in execution_plan:
            missing_deps = [dependency for dependency in subtask.depends_on if dependency not in completed]
            if missing_deps:
                return outputs, False, f"Subtask '{subtask.id}' has unfinished dependencies: {missing_deps}"

            self._progress(
                f"[qa] subtask start | id={subtask.id} | layer={subtask.layer} | category={subtask.category}"
            )
            self.env.logger.log_event("subtask_start", {
                "subtask_id": subtask.id,
                "layer": subtask.layer,
                "category": subtask.category,
                "description": subtask.description,
                "depends_on": subtask.depends_on,
            })

            if subtask.layer == "synthesis":
                if not subtask.metadata:
                    subtask.metadata = {}
                subtask.metadata["dependency_variables"] = self._dependency_variables(
                    subtask,
                    subtasks_by_id,
                    accepted_updates,
                )

            try:
                if subtask.id in mixed_synthesis_ids:
                    output = self.synthesis_agent.run_subtask(question, subtask)
                elif subtask.category == "common_info":
                    output = self.common_info_action.run(question, subtask)
                elif subtask.layer == "synthesis":
                    output = self.synthesis_agent.run_subtask(question, subtask)
                else:
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
                        selected_table_ids = self._selected_table_ids() or [self.env.default_table_id()]
                        self._set_active_tables(selected_table_ids)
            except Exception as exc:
                output = AgentOutput(
                    subtask_id=subtask.id,
                    description=subtask.description,
                    code=subtask.code_attempt or "",
                    success=False,
                    observation=f"Unhandled subtask error: {exc}",
                    reasoning="The subtask raised outside its normal execution/review loop.",
                )
                self.env.logger.log_event("subtask_exception", {
                    "subtask_id": subtask.id,
                    "error": str(exc),
                })

            self._progress(
                f"[qa] subtask done | id={subtask.id} | success={output.success} | "
                f"updates={list(output.namespace_updates.keys())}"
            )
            output.layer = subtask.layer
            output.category = subtask.category
            outputs.append(output)
            selected_exp = self.env.experience_pool.select()
            self.env.logger.log_event("subtask_complete", {
                "subtask_id": subtask.id,
                "success": output.success,
                "observation": output.observation,
                "code": output.code,
                "namespace_updates": list(output.namespace_updates.keys()),
                "experience_count": len(selected_exp),
            })
            if not output.success:
                return outputs, False, f"Failed at subtask '{subtask.id}': {output.observation}"
            accepted_updates[subtask.id] = tuple(output.namespace_updates.keys())
            completed.add(subtask.id)

        return outputs, True, None

    @staticmethod
    def _dependency_variables(
        subtask: SubTask,
        subtasks_by_id: dict[str, SubTask],
        accepted_updates: dict[str, tuple[str, ...]],
    ) -> list[str]:
        """Collect accepted namespace updates from all transitive dependencies."""
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

    def _final_answer(self, execution_plan: list[SubTask], plan: list[SubTask]) -> str | None:
        final_val = self.env.execution_namespace.get("final_answer")
        if final_val is None:
            return None
        mixed_synthesis_ids = self._mixed_synthesis_ids(plan)
        synthesis_subtasks = [subtask for subtask in execution_plan if subtask.layer == "synthesis"]
        final_synthesis = synthesis_subtasks[-1] if synthesis_subtasks else None
        pure_common_info = bool(
            final_synthesis
            and final_synthesis.category == "common_info"
            and final_synthesis.id not in mixed_synthesis_ids
        )
        serialized = self._serialize_final_value(final_val)
        answer = serialized if pure_common_info else self._humanize_header_ids(serialized)
        self.env.execution_namespace["final_answer"] = answer
        return answer

    @classmethod
    def _serialize_final_value(cls, value: Any) -> str:
        """Serialize pandas values without inheriting display truncation settings."""
        try:
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                headers = [cls._markdown_cell(column) for column in value.columns]
                lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join("---" for _ in headers) + " |",
                ]
                for row in value.itertuples(index=False, name=None):
                    lines.append("| " + " | ".join(cls._markdown_cell(cell) for cell in row) + " |")
                return "\n".join(lines)
            if isinstance(value, pd.Series):
                return value.to_string(max_rows=None)
        except ImportError:
            pass
        if isinstance(value, (list, tuple)) and value and all(isinstance(item, dict) for item in value):
            headers = []
            for item in value:
                for key in item:
                    if key not in headers:
                        headers.append(key)
            lines = [
                "| " + " | ".join(cls._markdown_cell(header) for header in headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
            for item in value:
                lines.append(
                    "| " + " | ".join(cls._markdown_cell(item.get(header)) for header in headers) + " |"
                )
            return "\n".join(lines)
        return str(value)

    @staticmethod
    def _markdown_cell(value: Any) -> str:
        try:
            import pandas as pd

            if pd.isna(value):
                return ""
        except (ImportError, TypeError, ValueError):
            pass
        return str(value).replace("|", r"\|").replace("\r\n", "<br>").replace("\n", "<br>")

    def _is_pure_common_info_plan(self, plan: list[SubTask]) -> bool:
        synthesis_ids = {subtask.id for subtask in plan if subtask.layer == "synthesis"}
        return bool(
            synthesis_ids
            and all(subtask.category == "common_info" for subtask in plan)
            and not self._mixed_synthesis_ids(plan)
        )

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
                f"- subtask={getattr(output, 'subtask_id', '')} success={getattr(output, 'success', False)}\n"
                f"  observation={observation}"
            )
        evidence = "\n".join(recent) or "No subtask output was produced."
        return f"Failure: {error_msg or 'Unknown execution failure'}\n\nRecent runtime evidence:\n{evidence}"

    def close(self) -> None:
        self.env.workbook.close()

    def __enter__(self) -> "TableQARunner":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _progress(self, message: str) -> None:
        if self.console_progress:
            print(message, flush=True)

    def _selected_table_ids(self) -> list[str]:
        raw = self.env.execution_namespace.get("selected_table_ids")
        if isinstance(raw, str):
            candidates = [raw]
        elif isinstance(raw, (list, tuple, set)):
            candidates = [str(item) for item in raw]
        else:
            candidates = []
        valid = set(self.env.operators.list_tables())
        selected = []
        for table_id in candidates:
            if table_id in valid and table_id not in selected:
                selected.append(table_id)
        return selected

    def _set_active_tables(self, table_ids: list[str]) -> None:
        valid = set(self.env.operators.list_tables())
        selected = [table_id for table_id in table_ids if table_id in valid]
        if not selected:
            selected = [self.env.default_table_id()]

        self._progress(f"[qa] preload tables start | table_ids={selected}")
        table_dfs = {
            table_id: self.env.operators.read_table_as_dataframe(table_id, has_headers=True)
            for table_id in selected
        }
        primary_table_id = selected[0]
        primary_df = table_dfs[primary_table_id]

        self.env.execution_namespace["selected_table_ids"] = selected
        self.env.execution_namespace["table_ids"] = selected
        self.env.execution_namespace["table_dfs"] = table_dfs
        self.env.execution_namespace["table_id"] = primary_table_id
        self.env.execution_namespace["table_df"] = primary_df

        for table_id, table_df in table_dfs.items():
            safe_table_var = re.sub(r"[^a-zA-Z0-9_]", "_", table_id)
            self.env.execution_namespace[safe_table_var] = table_df
            spaced_table_var = re.sub(r"(?<=\D)(\d+)$", r"_\1", safe_table_var)
            self.env.execution_namespace[spaced_table_var] = table_df
        self._progress(
            "[qa] preload tables done | "
            + ", ".join(f"{table_id}:shape={getattr(table_df, 'shape', None)}" for table_id, table_df in table_dfs.items())
        )

    @staticmethod
    def _unambiguous_header_labels(table_header_labels: dict[str, dict[str, str]]) -> dict[str, str]:
        """Collapse per-table labels without guessing when the same ID has conflicting labels."""
        labels_by_id: dict[str, set[str]] = {}
        for labels in table_header_labels.values():
            for header_id, label in labels.items():
                clean_id = str(header_id).strip()
                clean_label = str(label).strip()
                if clean_id and clean_label:
                    labels_by_id.setdefault(clean_id, set()).add(clean_label)
        return {
            header_id: next(iter(labels))
            for header_id, labels in labels_by_id.items()
            if len(labels) == 1
        }

    def _humanize_header_ids(self, answer: str) -> str:
        """Replace internal header-ID tokens in a user-facing answer with verified labels."""
        table_ids = self._selected_table_ids()
        if not table_ids:
            table_ids = self.env.operators.list_tables()
        table_header_labels = {
            table_id: {
                header.id: header.label
                for header in self.env.operators.list_headers(table_id)
                if header.label
            }
            for table_id in table_ids
        }
        labels = self._unambiguous_header_labels(table_header_labels)
        humanized = answer
        for header_id in sorted(labels, key=len, reverse=True):
            label = labels[header_id]
            if header_id == label or header_id.isdecimal():
                continue
            pattern = rf"(?<![\w]){re.escape(header_id)}(?![\w])"
            humanized = re.sub(pattern, lambda _match, value=label: value, humanized)
        return humanized

    def _topological_sort(self, plan: List[SubTask]) -> List[SubTask]:
        by_id = {}
        for subtask in plan:
            if subtask.id in by_id:
                raise ValueError(f"Duplicate subtask id in plan: {subtask.id}")
            by_id[subtask.id] = subtask

        for subtask in plan:
            missing = [dep for dep in subtask.depends_on if dep not in by_id]
            if missing:
                raise ValueError(f"Subtask '{subtask.id}' depends on unknown subtasks: {missing}")

        ordered: List[SubTask] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(subtask_id: str) -> None:
            if subtask_id in permanent:
                return
            if subtask_id in temporary:
                raise ValueError(f"Cycle detected in subtask dependencies at '{subtask_id}'")
            temporary.add(subtask_id)
            for dep_id in by_id[subtask_id].depends_on:
                visit(dep_id)
            temporary.remove(subtask_id)
            permanent.add(subtask_id)
            ordered.append(by_id[subtask_id])

        for subtask in plan:
            visit(subtask.id)
        return ordered

    @staticmethod
    def _mixed_synthesis_ids(plan: List[SubTask]) -> set[str]:
        by_id = {subtask.id: subtask for subtask in plan}
        memo: dict[str, set[str]] = {}

        def dependency_categories(subtask_id: str) -> set[str]:
            if subtask_id in memo:
                return memo[subtask_id]
            categories: set[str] = set()
            for dependency_id in by_id[subtask_id].depends_on:
                dependency = by_id[dependency_id]
                categories.add(dependency.category)
                categories.update(dependency_categories(dependency_id))
            memo[subtask_id] = categories
            return categories

        return {
            subtask.id
            for subtask in plan
            if subtask.layer == "synthesis"
            and {"normal", "common_info"}.issubset(dependency_categories(subtask.id))
        }

    def _make_run_id(self, question: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", question.strip()).strip("_").lower()
        if not slug:
            slug = "qa_run"
        return f"{timestamp}_{slug[:60]}"

    def _persist_run_artifacts(self, result: QAResult, run_dir: Path, event_start_index: int) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        cells_dir = run_dir / "cells"
        cells_dir.mkdir(parents=True, exist_ok=True)

        events = self.env.logger.events[event_start_index:]
        artifacts: dict[str, str] = {
            "run_dir": str(run_dir),
            "events_jsonl": str(run_dir / "events.jsonl"),
            "plan_json": str(run_dir / "plan.json"),
            "result_json": str(run_dir / "result.json"),
            "cells_dir": str(cells_dir),
        }

        self._write_events_jsonl(run_dir / "events.jsonl", events)
        self._write_json(run_dir / "plan.json", [self._subtask_to_dict(subtask) for subtask in result.plan])

        for index, cell in enumerate(self.env.notebook.cells, start=1):
            cell_path = cells_dir / f"{index:02d}_{cell.cell_id}.py"
            cell_path.write_text(cell.code.rstrip() + "\n", encoding="utf-8")

        if self.env.notebook.cells:
            artifacts["cells_index"] = str(cells_dir)

        answer_output = self._find_answer_output(result)
        if answer_output is not None:
            answer_path = run_dir / "answer.py"
            answer_path.write_text(answer_output.code.rstrip() + "\n", encoding="utf-8")
            artifacts["answer_py"] = str(answer_path)

        generated_dir = run_dir / "generated_code"
        generated_dir.mkdir(parents=True, exist_ok=True)
        for index, output in enumerate(result.subtask_outputs, start=1):
            safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", output.subtask_id).strip("_") or f"subtask_{index}"
            code_path = generated_dir / f"{index:02d}_{safe_id}.py"
            code_path.write_text(output.code.rstrip() + "\n", encoding="utf-8")
        artifacts["generated_code_dir"] = str(generated_dir)

        try:
            notebook_path = self.env.export_notebook(run_dir / "notebook.ipynb")
            artifacts["notebook_ipynb"] = str(notebook_path)
        except Exception as exc:
            artifacts["notebook_export_error"] = str(exc)

        result.artifacts = artifacts
        self._write_json(run_dir / "result.json", {
            "question": result.question,
            "success": result.success,
            "final_answer": result.final_answer,
            "error": result.error,
            "execution_time": result.execution_time,
            "replan_count": result.replan_count,
            "token_usage": result.token_usage,
            "artifacts": artifacts,
            "plan": [self._subtask_to_dict(subtask) for subtask in result.plan],
            "subtask_outputs": [
                {
                    "subtask_id": output.subtask_id,
                    "description": output.description,
                    "success": output.success,
                    "observation": output.observation,
                    "reasoning": output.reasoning,
                    "code": output.code,
                    "layer": output.layer,
                    "category": output.category,
                    "namespace_updates": list(output.namespace_updates.keys()),
                }
                for output in result.subtask_outputs
            ],
        })

        self.env.logger.log_event("run_artifact_complete", artifacts)

    def _find_answer_output(self, result: QAResult):
        synthesis_ids = {subtask.id for subtask in result.plan if subtask.layer == "synthesis"}
        for output in reversed(result.subtask_outputs):
            if output.subtask_id in synthesis_ids and output.code.strip():
                return output
        for output in reversed(result.subtask_outputs):
            if "final_answer" in output.code and output.code.strip():
                return output
        return None

    def _subtask_to_dict(self, subtask: SubTask) -> dict[str, Any]:
        return {
            "id": subtask.id,
            "description": subtask.description,
            "layer": subtask.layer,
            "category": subtask.category,
            "depends_on": list(subtask.depends_on),
            "status": subtask.status,
            "metadata": subtask.metadata,
            "assigned_agent": subtask.assigned_agent,
            "observation": subtask.observation,
        }

    def _write_events_jsonl(self, path: Path, events: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def token_usage(self) -> dict[str, int]:
        return self._token_usage()

    def _token_usage(self) -> dict[str, int]:
        if self.llm_client is None:
            return {"prompt": 0, "completion": 0}
        return self.llm_client.token_usage()
