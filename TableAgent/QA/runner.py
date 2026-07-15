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
from TableAgent.QA.actions.execute_notebook import ExecuteNotebookCodeAction
from TableAgent.QA.actions.review import ReviewSubtaskAction
from TableAgent.schema.qa import QAResult
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
    ):
        raw_config = config or {}
        self.settings = raw_config.get("table_agent", raw_config) if isinstance(raw_config, dict) else {}
        
        # Load parameters from configuration/settings if present
        actual_max_retries = max_retries
        actual_max_records = max_experience_records
        log_path = None
        artifact_root = Path("logs") / "qa_runs"
        max_observation_chars = 2000
        max_error_chars = 2000
        max_value_repr_chars = 800
        
        if self.settings:
            actual_max_retries = int(self.settings.get("qa_max_retries", max_retries))
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
        self.qa_artifact_root = artifact_root

        self.env = QAEnvironment(
            structure_path=structure_path,
            workbook_path=workbook_path,
            max_experience_records=actual_max_records,
            log_path=log_path,
            max_observation_chars=max_observation_chars,
            max_error_chars=max_error_chars,
            max_value_repr_chars=max_value_repr_chars,
            table_retriever=table_retriever,
        )
        
        self.env.logger.log_event("config_loaded", {
            "max_retries": actual_max_retries,
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
            
        # 1. Plan
        try:
            plan = self.planner.plan(question, table_id=table_id)
            self._progress(f"[qa] planning done | subtasks={[(subtask.id, subtask.layer) for subtask in plan]}")
        except Exception as exc:
            execution_time = time.time() - start_time
            error_msg = f"Planning failed: {exc}"
            result = QAResult(
                question=question,
                plan=[],
                subtask_outputs=[],
                final_answer=None,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                token_usage=self._token_usage(),
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
        
        # 2. Execute plan subtasks in dependency order
        try:
            execution_plan = self._topological_sort(plan)
        except ValueError as exc:
            execution_time = time.time() - start_time
            result = QAResult(
                question=question,
                plan=plan,
                subtask_outputs=[],
                final_answer=None,
                success=False,
                error=str(exc),
                execution_time=execution_time,
                token_usage=self._token_usage(),
            )
            result.logs = self.env.logger.events
            self.env.logger.log_event("run_complete", {
                "success": False,
                "final_answer": None,
                "error": str(exc),
                "execution_time": execution_time,
            })
            self._persist_run_artifacts(result, run_dir, event_start_index)
            return result

        subtask_outputs = []
        success = True
        error_msg = None
        completed: set[str] = set()

        self.env.logger.log_event("execution_plan", {
            "order": [subtask.id for subtask in execution_plan],
            "dependencies": {subtask.id: subtask.depends_on for subtask in execution_plan},
        })

        for subtask in execution_plan:
            missing_deps = [dep for dep in subtask.depends_on if dep not in completed]
            if missing_deps:
                success = False
                error_msg = f"Subtask '{subtask.id}' has unfinished dependencies: {missing_deps}"
                break
            self._progress(f"[qa] subtask start | id={subtask.id} | layer={subtask.layer}")
            self.env.logger.log_event("subtask_start", {
                "subtask_id": subtask.id,
                "layer": subtask.layer,
                "description": subtask.description,
                "depends_on": subtask.depends_on,
            })
            
            if subtask.layer == "synthesis":
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
                    selected_table_ids = self._selected_table_ids()
                    if not selected_table_ids:
                        selected_table_ids = [self.env.default_table_id()]
                    self._set_active_tables(selected_table_ids)
            self._progress(
                f"[qa] subtask done | id={subtask.id} | success={output.success} | "
                f"updates={list(output.namespace_updates.keys())}"
            )
                
            subtask_outputs.append(output)
            
            selected_exp = self.env.experience_pool.select()
            self.env.logger.log_event("subtask_complete", {
                "subtask_id": subtask.id,
                "success": output.success,
                "observation": output.observation,
                "code": output.code,
                "namespace_updates": list(output.namespace_updates.keys()),
                "experience_count": len(selected_exp)
            })
            
            if not output.success:
                success = False
                error_msg = f"Failed at subtask '{subtask.id}': {output.observation}"
                break
            completed.add(subtask.id)

        # 3. Retrieve final answer from the shared namespace
        final_answer = None
        if success:
            final_val = self.env.execution_namespace.get("final_answer")
            if final_val is not None:
                final_answer = self._humanize_header_ids(str(final_val))
                self.env.execution_namespace["final_answer"] = final_answer
            else:
                success = False
                error_msg = "Synthesis layer completed, but 'final_answer' variable was not set in namespace."

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
        
        # Expose logs/events on the result object
        result.logs = self.env.logger.events
        
        self.env.logger.log_event("run_complete", {
            "success": success,
            "final_answer": final_answer,
            "error": error_msg,
            "execution_time": execution_time
        })

        self._persist_run_artifacts(result, run_dir, event_start_index)
        self._progress(f"[qa] run done | success={success} | artifact_dir={run_dir}")
        
        return result

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
            humanized = re.sub(pattern, lambda _match, value=label: value, humanized, flags=re.IGNORECASE)
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
