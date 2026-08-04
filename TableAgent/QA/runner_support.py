from __future__ import annotations

import re
from typing import Any

from TableAgent.schema.subtask import SubTask


class QARunnerSupportMixin:
    """Shared answer formatting, table state, and lifecycle helpers."""

    def _final_answer(
        self, execution_plan: list[SubTask], plan: list[SubTask]
    ) -> str | None:
        del execution_plan
        final_val = self.env.execution_namespace.get("final_answer")
        if final_val is None:
            return None
        serialized = self._serialize_final_value(final_val)
        answer = (
            serialized
            if self._is_pure_common_info_plan(plan)
            else self._humanize_header_ids(serialized)
        )
        self.env.execution_namespace["final_answer"] = answer
        return answer

    @classmethod
    def _serialize_final_value(cls, value: Any) -> str:
        try:
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                headers = [cls._markdown_cell(column) for column in value.columns]
                lines = [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join("---" for _ in headers) + " |",
                ]
                for row in value.itertuples(index=False, name=None):
                    lines.append(
                        "| "
                        + " | ".join(cls._markdown_cell(cell) for cell in row)
                        + " |"
                    )
                return "\n".join(lines)
            if isinstance(value, pd.Series):
                return value.to_string(max_rows=None)
        except ImportError:
            pass
        if (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, dict) for item in value)
        ):
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
                    "| "
                    + " | ".join(
                        cls._markdown_cell(item.get(header)) for header in headers
                    )
                    + " |"
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
        return (
            str(value)
            .replace("|", r"\|")
            .replace("\r\n", "<br>")
            .replace("\n", "<br>")
        )

    def close(self) -> None:
        self.env.workbook.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)
        if self.console_progress:
            print(message, flush=True)

    @staticmethod
    def _is_pure_common_info_plan(plan: list[SubTask]) -> bool:
        return bool(plan) and all(
            subtask.category == "common_info" for subtask in plan
        )

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
            table_id: self.env.operators.read_table_as_dataframe(
                table_id, has_headers=True
            )
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
            + ", ".join(
                f"{table_id}:shape={getattr(table_df, 'shape', None)}"
                for table_id, table_df in table_dfs.items()
            )
        )

    @staticmethod
    def _unambiguous_header_labels(
        table_header_labels: dict[str, dict[str, str]],
    ) -> dict[str, str]:
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
            humanized = re.sub(
                pattern, lambda _match, value=label: value, humanized
            )
        return humanized

    def token_usage(self) -> dict[str, int]:
        return self._token_usage()

    def _token_usage(self) -> dict[str, int]:
        if self.llm_client is None:
            return {"prompt": 0, "completion": 0}
        return self.llm_client.token_usage()

    def _llm_call_metrics(self) -> list[dict[str, Any]]:
        if self.llm_client is None:
            return []
        return self.llm_client.call_metrics()
