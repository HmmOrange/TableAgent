from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from TableAgent.QA.runner import TableQARunner
from TableAgent.schema.qa import QAResult


SUPPORTED_QA_WORKBOOK_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm"}


@dataclass(frozen=True)
class TableQARequest:
    """One bounded QA request over a workbook and its prepared structure."""

    question: str
    workbook_path: Path
    structure_path: Path
    artifact_dir: Path
    table_id: str | None = None


@dataclass(frozen=True)
class TableQAResponse:
    """Stable result returned to integrations without exposing runner internals."""

    success: bool
    answer: str | None
    error: str | None
    execution_time: float
    artifacts: Mapping[str, str]
    token_usage: Mapping[str, int]
    limitations: tuple[str, ...] = ()


RunnerFactory = Callable[..., TableQARunner]


class TableQAEngine:
    """Run QA against an externally supplied, ingestion-prepared structure file."""

    def __init__(
        self,
        *,
        llm_client: Any | None = None,
        code_action: Any | None = None,
        config: Mapping[str, Any] | None = None,
        runner_factory: RunnerFactory = TableQARunner,
    ) -> None:
        self._llm_client = llm_client
        self._code_action = code_action
        self._config = dict(config or {})
        self._runner_factory = runner_factory

    def answer(self, request: TableQARequest) -> TableQAResponse:
        question = request.question.strip()
        if not question:
            raise ValueError("question must be non-empty")

        workbook_path = request.workbook_path.expanduser().resolve()
        structure_path = request.structure_path.expanduser().resolve()
        artifact_dir = request.artifact_dir.expanduser().resolve()
        self._validate_workbook(workbook_path)
        self._validate_structure(structure_path)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        config = dict(self._config)
        config["qa_artifact_dir"] = str(artifact_dir)
        if request.table_id:
            config["table_id"] = request.table_id

        runner = self._runner_factory(
            structure_path=str(structure_path),
            workbook_path=str(workbook_path),
            llm_client=self._llm_client,
            config=config,
            code_action=self._code_action,
        )
        result = runner.run(question)
        if not isinstance(result, QAResult):
            raise TypeError("TableAgent runner must return QAResult")

        return TableQAResponse(
            success=result.success,
            answer=result.final_answer,
            error=result.error,
            execution_time=result.execution_time,
            artifacts=dict(result.artifacts),
            token_usage=dict(result.token_usage),
        )

    @staticmethod
    def _validate_workbook(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Workbook not found: {path}")
        if path.suffix.lower() not in SUPPORTED_QA_WORKBOOK_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_QA_WORKBOOK_EXTENSIONS))
            raise ValueError(
                f"Unsupported QA workbook extension {path.suffix!r}; expected one of: {supported}"
            )

    @staticmethod
    def _validate_structure(path: Path) -> None:
        if path.name != "structure.yaml":
            raise ValueError("Sheet structure file must be named structure.yaml")
        if not path.is_file():
            raise FileNotFoundError(f"Structure file not found: {path}")
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid structure YAML: {path}") from exc
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Structure YAML must contain at least one table mapping")


__all__ = [
    "SUPPORTED_QA_WORKBOOK_EXTENSIONS",
    "TableQAEngine",
    "TableQARequest",
    "TableQAResponse",
]
