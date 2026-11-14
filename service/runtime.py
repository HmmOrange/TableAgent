"""The TableAgent service: answer questions over one workbook, keeping each job's artifacts in its own directory."""

from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import openpyxl

from TableAgent import QASettings, WorkbookQA
from TableAgent.src.prompts import DIRECT_ANSWER_PROMPT
from TableAgent.src.utils.workbook_view import workbook_preview
from service.clients import create_model_client
from service.config import load_config

WORKBOOK_SUFFIXES = (".xlsx", ".xlsm", ".xltx", ".xltm")


class TableAgentService:
    """Runs QA jobs: one workbook, one or more questions, answered `max_workers` at a time.

    The answer model is the config's `llm`; the vision model (`vlm`), when the config names one, lets the exploration
    look at rendered cell ranges. Clients may be injected instead (tests, custom endpoints).
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        llm_client: Any | None = None,
        vision_client: Any | None = None,
        llm_profile: str | None = None,
        vlm_profile: str | None = None,
        root_dir: str | Path | None = None,
    ):
        service = config.get("service") or {}
        if not isinstance(service, dict):
            raise ValueError("service configuration must be a mapping")
        self.root_dir = Path(root_dir or service.get("root_dir", "outputs")).expanduser().resolve()
        self.max_workers = max(1, int(service.get("max_workers", 1)))
        self.max_upload_bytes = max(1, int(service.get("max_upload_mb", 100))) * 1024 * 1024
        self.api_key = str(service["api_key"]) if service.get("api_key") else None
        self.allow_local_paths = bool(service.get("allow_local_paths", False))
        self.allowed_input_roots = tuple(Path(value).expanduser().resolve() for value in service.get("allowed_input_roots", []))
        self.settings = QASettings.from_config({**(config.get("table_agent") or {}), "max_workers": self.max_workers})
        llm = llm_client or create_model_client(config, kind="llm", profile=llm_profile)
        if vision_client is None and (vlm_profile or config.get("vlm")):
            vision_client = create_model_client(config, kind="vlm", profile=vlm_profile)
        self.agent = WorkbookQA(llm, self.settings, vision_client)

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml", **kwargs: Any) -> "TableAgentService":
        return cls(load_config(path), **kwargs)

    def run(self, workbook: str | Path, questions: list[str], *, job_id: str | None = None) -> dict[str, Any]:
        """Answer every question over the workbook; the workbook is copied into the job directory first."""
        questions = [str(question).strip() for question in questions if str(question).strip()]
        if not questions:
            raise ValueError("At least one non-empty question is required")
        source = self.validate_workbook(workbook)
        job_dir = self.root_dir / (job_id or new_job_id())
        job_dir.mkdir(parents=True, exist_ok=True)
        workbook_path = job_dir / source.name
        if source.resolve() != workbook_path.resolve():
            shutil.copy2(source, workbook_path)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(questions))) as executor:
            answers = list(executor.map(
                lambda item: self._answer(workbook_path, job_dir / "qa" / f"q{item[0]}", item[1]),
                enumerate(questions, start=1),
            ))
        return {
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "workbook": workbook_path.name,
            "answers": answers,
            "artifacts": sorted(path.relative_to(job_dir).as_posix() for path in job_dir.rglob("*") if path.is_file()),
        }

    def validate_workbook(self, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        if path.suffix.lower() not in WORKBOOK_SUFFIXES:
            raise ValueError(f"Unsupported workbook type {path.suffix!r}; use one of {', '.join(WORKBOOK_SUFFIXES)}")
        if not path.is_file():
            raise FileNotFoundError(f"Workbook not found: {path}")
        return path

    def validate_local_workbook(self, value: str | Path) -> Path:
        """A workbook named by path in a request: only when local paths are allowed, and only under an allowed root."""
        if not self.allow_local_paths:
            raise PermissionError("Local workbook paths are disabled; upload the workbook instead")
        path = self.validate_workbook(value)
        if self.allowed_input_roots and not any(path.is_relative_to(root) for root in self.allowed_input_roots):
            raise PermissionError(f"Workbook path is outside the allowed input roots: {path}")
        return path

    def _answer(self, workbook_path: Path, qa_dir: Path, question: str) -> dict[str, Any]:
        response, qa = self.agent.run(
            question=question,
            workbook_path=workbook_path,
            qa_artifact_dir=qa_dir,
            fallback_prompt=lambda: DIRECT_ANSWER_PROMPT.format(
                question=question, table_context=_preview(workbook_path, question),
            ),
        )
        trace = (qa.get("artifacts") or {}).get("trace_md")
        return {
            "question": question,
            "answer": response.content,
            "token_usage": {"prompt": response.prompt_tokens, "completion": response.completion_tokens},
            "trace": str(Path(trace).relative_to(qa_dir.parent.parent)) if trace else None,
            "qa": {key: value for key, value in qa.items() if key not in ("artifacts", "llm_calls")},
        }


def _preview(workbook_path: Path, question: str) -> str:
    """The workbook as a fallback answer reads it; empty when it cannot be read."""
    try:
        workbook = openpyxl.load_workbook(workbook_path, data_only=True)
    except Exception:
        return ""
    try:
        return workbook_preview(workbook, workbook_path.name, question=question)
    finally:
        workbook.close()


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]
