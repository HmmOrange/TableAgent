from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.QA.actions.base_action import BaseCodeGenerationAction
from TableAgent.QA.actions.common_info import CommonInfoSubtaskAction
from TableAgent.QA.actions.execute_notebook import ExecuteNotebookCodeAction
from TableAgent.QA.actions.review import ReviewSubtaskAction
from TableAgent.QA.actions.review_final_answer import ReviewFinalAnswerAction
from TableAgent.QA.agents.planner import TableQAPlanner
from TableAgent.QA.agents.react_agent import TableQAAgent
from TableAgent.QA.agents.synthesis_agent import TableQASynthesisAgent
from TableAgent.QA.runner_artifacts import QAArtifactMixin
from TableAgent.QA.runner_execution import QAExecutionMixin
from TableAgent.QA.runner_support import QARunnerSupportMixin

if TYPE_CHECKING:
    from TableAgent.pipeline.retrieval import TableRetrieverContract


class TokenCountingLLM:
    """Proxy an LLM client while accumulating token usage from its responses."""

    def __init__(self, client: Any):
        self.client = client
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Any:
        started_at = time.perf_counter()
        try:
            response = self.client.generate(prompt, system_prompt=system_prompt)
        except Exception as exc:
            self.calls.append(
                {
                    "index": len(self.calls) + 1,
                    "duration_ms": max(
                        0, round((time.perf_counter() - started_at) * 1000)
                    ),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "token_capped": False,
                    "success": False,
                    "error_type": type(exc).__name__,
                }
            )
            raise
        self.prompt_tokens += int(getattr(response, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(response, "completion_tokens", 0) or 0)
        self.calls.append(
            {
                "index": len(self.calls) + 1,
                "duration_ms": max(
                    0, round((time.perf_counter() - started_at) * 1000)
                ),
                "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    getattr(response, "completion_tokens", 0) or 0
                ),
                "token_capped": bool(getattr(response, "token_capped", False)),
                "success": True,
                "error_type": None,
            }
        )
        return response

    def token_usage(self) -> dict[str, int]:
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
        }

    def call_metrics(self) -> list[dict[str, Any]]:
        return [dict(call) for call in self.calls]


class TableQARunner(QAExecutionMixin, QARunnerSupportMixin, QAArtifactMixin):
    """Coordinate planning, deterministic routes, execution, and QA artifacts."""

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
        progress_callback: Callable[[str], None] | None = None,
    ):
        raw_config = config or {}
        self.settings = (
            raw_config.get("table_agent", raw_config)
            if isinstance(raw_config, dict)
            else {}
        )

        actual_max_retries = max_retries
        actual_max_replans = 5
        actual_max_records = max_experience_records
        log_path = None
        artifact_root = Path("logs") / "qa_runs"
        max_observation_chars = 2000
        max_error_chars = 2000
        max_value_repr_chars = 800

        if self.settings:
            actual_max_retries = int(
                self.settings.get("qa_max_retries", max_retries)
            )
            actual_max_replans = int(self.settings.get("qa_max_replans", 5))
            actual_max_records = int(
                self.settings.get(
                    "qa_max_experience_records", max_experience_records
                )
            )
            log_path_value = self.settings.get("qa_log_path")
            if log_path_value:
                log_path = str(log_path_value)
                artifact_root = Path(log_path).parent / "qa_runs"
            max_observation_chars = int(
                self.settings.get(
                    "qa_max_observation_chars", max_observation_chars
                )
            )
            max_error_chars = int(
                self.settings.get("qa_max_error_chars", max_error_chars)
            )
            max_value_repr_chars = int(
                self.settings.get("qa_max_value_repr_chars", max_value_repr_chars)
            )

        if isinstance(config, dict):
            agent_config = config.get("table_agent", {})
            if not isinstance(agent_config, dict):
                agent_config = {}
            explicit_artifact_root = config.get(
                "qa_artifact_dir"
            ) or agent_config.get("qa_artifact_dir")
            if explicit_artifact_root:
                artifact_root = Path(str(explicit_artifact_root))
        self.console_progress = bool(
            config.get("qa_console_progress", False)
            if isinstance(config, dict)
            else False
        )
        self.progress_callback = progress_callback
        self.enable_final_answer_review = bool(
            self.settings.get("qa_final_answer_review", False)
        )
        qa_routing = self._qa_routing_settings()
        self.qa_routing_mode = str(
            self.settings.get(
                "qa_routing_mode", self._route_value(qa_routing, "mode", "auto")
            )
        )
        self.qa_common_info_enabled = bool(
            self.settings.get(
                "qa_common_info_enabled",
                self._route_value(qa_routing, "common_info_enabled", True),
            )
        )
        self.qa_common_info_fallback = str(
            self.settings.get(
                "qa_common_info_fallback",
                self._route_value(qa_routing, "common_info_fallback", "normal"),
            )
        )
        self.qa_artifact_root = artifact_root
        self.max_replans = max(0, actual_max_replans)
        self.max_retries = max(0, actual_max_retries)

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
        self.env.qa_routing_mode = self.qa_routing_mode
        self.env.qa_common_info_enabled = self.qa_common_info_enabled
        self.env.logger.log_event(
            "config_loaded",
            {
                "max_retries": actual_max_retries,
                "max_replans": self.max_replans,
                "max_experience_records": actual_max_records,
                "log_path": log_path,
                "max_observation_chars": max_observation_chars,
                "max_error_chars": max_error_chars,
                "max_value_repr_chars": max_value_repr_chars,
                "qa_routing_mode": self.qa_routing_mode,
                "qa_common_info_enabled": self.qa_common_info_enabled,
                "qa_common_info_fallback": self.qa_common_info_fallback,
            },
        )

        self.llm_client = (
            TokenCountingLLM(llm_client) if llm_client is not None else None
        )
        self.planner = TableQAPlanner(self.env, llm_client=self.llm_client)

        self.table_id = None
        if isinstance(config, dict):
            self.table_id = config.get("table_id")
            nested = config.get("table_agent")
            if not self.table_id and isinstance(nested, dict):
                self.table_id = nested.get("table_id")

        code_action = code_action or policy
        if code_action is None:
            if self.llm_client is None:
                raise ValueError(
                    "Either llm_client or code_action must be provided to TableQARunner."
                )
            from TableAgent.QA.actions.llm_code_generation import (
                LLMCodeGenerationAction,
            )

            code_action = LLMCodeGenerationAction(self.llm_client, self.env)
        else:
            if hasattr(code_action, "env") or hasattr(code_action, "__dict__"):
                try:
                    code_action.env = self.env
                except Exception:
                    pass
            if (
                self.llm_client is not None
                and getattr(code_action, "llm_client", None) is llm_client
            ):
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
        self.common_info_action = CommonInfoSubtaskAction(
            self.env, llm_client=self.llm_client
        )
        self.final_answer_review = ReviewFinalAnswerAction(
            self.env, llm_client=self.llm_client
        )

    def _qa_routing_settings(self) -> Any:
        routing = self.settings.get("routing")
        if isinstance(routing, dict):
            return routing.get("qa", {})
        return getattr(routing, "qa", None)

    @staticmethod
    def _route_value(routing: Any, name: str, default: Any) -> Any:
        if isinstance(routing, dict):
            return routing.get(name, default)
        return getattr(routing, name, default)
