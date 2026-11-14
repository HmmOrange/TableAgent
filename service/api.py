"""HTTP API of the TableAgent service: ask questions about a workbook, by upload or by local path."""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from service.runtime import TableAgentService


class PathJobRequest(BaseModel):
    workbook: str = Field(min_length=1)
    questions: list[str] = Field(min_length=1)


class UploadJobRequest(BaseModel):
    questions: list[str] = Field(min_length=1)


def create_app(service: TableAgentService | None = None, *, config_path: str | Path | None = None) -> FastAPI:
    resolved_service = service or TableAgentService.from_config(config_path or _default_config_path())

    app = FastAPI(
        title="TableAgent API",
        version=_package_version(),
        description="Question answering over Excel workbooks.",
    )
    app.state.service = resolved_service

    @app.middleware("http")
    async def authenticate(request, call_next):
        if request.url.path.startswith("/v1/") and resolved_service.api_key:
            provided = request.headers.get("X-API-Key")
            if not provided or not secrets.compare_digest(provided, resolved_service.api_key):
                return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Invalid or missing X-API-Key"})
        return await call_next(request)

    @app.get("/health")
    @app.get("/health/live")
    def health_live() -> dict[str, Any]:
        return {"status": "ok", "service": "table-agent", "version": _package_version()}

    @app.get("/health/ready")
    def health_ready() -> dict[str, Any]:
        return {"status": "ready"}

    @app.get("/v1/status")
    def service_status() -> dict[str, Any]:
        return {"status": "ok", "version": _package_version(), "workers": resolved_service.max_workers}

    @app.post("/v1/jobs")
    def create_path_job(request: PathJobRequest) -> dict[str, Any]:
        try:
            workbook = resolved_service.validate_local_workbook(request.workbook)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        try:
            return resolved_service.run(workbook, request.questions)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/v1/jobs/upload")
    async def create_upload_job(payload: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
        try:
            request = UploadJobRequest.model_validate_json(payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid payload: {exc}") from exc
        with tempfile.TemporaryDirectory(prefix="table-agent-upload-") as upload_dir:
            target = Path(upload_dir) / _safe_name(Path(file.filename or "workbook.xlsx").name)
            try:
                size = 0
                with target.open("wb") as handle:
                    while chunk := await file.read(1024 * 1024):
                        size += len(chunk)
                        if size > resolved_service.max_upload_bytes:
                            limit = resolved_service.max_upload_bytes // (1024 * 1024)
                            raise ValueError(f"Upload '{target.name}' exceeds the {limit} MB limit")
                        handle.write(chunk)
                resolved_service.validate_workbook(target)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            finally:
                await file.close()
            try:
                return await asyncio.to_thread(resolved_service.run, target, request.questions)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return app


def _default_config_path() -> Path:
    configured = os.environ.get("TABLE_AGENT_CONFIG")
    if configured:
        return Path(configured)
    private_config = Path("config.yaml")
    return private_config if private_config.is_file() else Path("config.example.yaml")


def _package_version() -> str:
    try:
        return version("table-agent")
    except PackageNotFoundError:
        return "0.1.0"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or "workbook.xlsx"
