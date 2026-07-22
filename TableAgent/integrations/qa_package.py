"""Runnable QA package boundary built on the public TableQAEngine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from TableAgent.configs import load_config

from .models import create_model_client
from .qa import TableQAEngine, TableQARequest, TableQAResponse


QA_PACKAGE_SCHEMA_VERSION = 1


class TableQAPackage:
    """Own the configured answer client and expose request-to-response QA."""

    def __init__(
        self,
        engine: TableQAEngine,
        *,
        owned_client: Any | None = None,
    ) -> None:
        self.engine = engine
        self._owned_client = owned_client

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config.yaml",
        *,
        llm_profile: str = "table_agent",
        session: Any | None = None,
    ) -> "TableQAPackage":
        config = load_config(config_path)
        client = create_model_client(
            config,
            kind="llm",
            profile=llm_profile,
            session=session,
        )
        return cls(
            TableQAEngine(llm_client=client, config=config),
            owned_client=client,
        )

    def run(self, request: TableQARequest) -> TableQAResponse:
        return self.answer(request)

    def answer(self, request: TableQARequest) -> TableQAResponse:
        """Match TableQAEngine so integrations can use the package as a drop-in."""

        return self.engine.answer(request)

    def close(self) -> None:
        if self._owned_client is None:
            return
        close = getattr(self._owned_client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "TableQAPackage":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_qa_request(path: str | Path) -> TableQARequest:
    """Load a versioned JSON request with paths relative to the request file."""

    request_path = Path(path).expanduser().resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("QA request JSON must contain an object")
    if payload.get("schema_version") != QA_PACKAGE_SCHEMA_VERSION:
        raise ValueError(
            f"QA request schema_version must be {QA_PACKAGE_SCHEMA_VERSION}"
        )
    base_dir = request_path.parent
    return TableQARequest(
        question=_required_text(payload, "question"),
        workbook_path=_request_path(payload, "workbook_path", base_dir),
        structure_path=_request_path(payload, "structure_path", base_dir),
        artifact_dir=_request_path(payload, "artifact_dir", base_dir),
        table_id=_optional_text(payload, "table_id"),
    )


def qa_response_payload(response: TableQAResponse) -> dict[str, Any]:
    """Convert the stable package response to JSON-safe primitives."""

    return {
        "schema_version": QA_PACKAGE_SCHEMA_VERSION,
        "success": response.success,
        "answer": response.answer,
        "error": response.error,
        "execution_time": response.execution_time,
        "artifacts": dict(response.artifacts),
        "token_usage": dict(response.token_usage),
        "limitations": list(response.limitations),
    }


def _request_path(
    payload: Mapping[str, Any], field: str, base_dir: Path
) -> Path:
    value = _required_text(payload, field)
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"QA request field '{field}' must be a non-empty string")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"QA request field '{field}' must be a string")
    return value.strip() or None


__all__ = [
    "QA_PACKAGE_SCHEMA_VERSION",
    "TableQAPackage",
    "load_qa_request",
    "qa_response_payload",
]
