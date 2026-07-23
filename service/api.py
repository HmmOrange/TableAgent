from __future__ import annotations

import asyncio
import os
import secrets
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from TableAgent.pipeline.common import safe_name
from service.runtime import Stage, TableAgentService


class PathJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    stage: Stage = "all"
    queries: list[str] = Field(default_factory=list)
    workbooks: list[str] = Field(min_length=1)
    embed: bool = False
    sheets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_queries(self) -> "PathJobRequest":
        _require_queries(self.stage, self.queries)
        return self


class UploadJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    stage: Stage = "all"
    queries: list[str] = Field(default_factory=list)
    embed: bool = False
    sheets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_queries(self) -> "UploadJobRequest":
        _require_queries(self.stage, self.queries)
        return self


def create_app(
    service: TableAgentService | None = None,
    *,
    config_path: str | Path | None = None,
) -> FastAPI:
    resolved_service = service or TableAgentService.from_config(config_path or _default_config_path())

    app = FastAPI(
        title="TableAgent API",
        version=_package_version(),
        description="Ephemeral spreadsheet structure extraction and question answering.",
    )
    app.state.service = resolved_service

    @app.middleware("http")
    async def authenticate(request, call_next):
        if request.url.path.startswith("/v1/") and resolved_service.api_key:
            provided = request.headers.get("X-API-Key")
            if not provided or not secrets.compare_digest(provided, resolved_service.api_key):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing X-API-Key"},
                )
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
        return {
            "status": "ok",
            "version": _package_version(),
            "workers": resolved_service.max_workers,
            "persistence": False,
        }

    @app.post("/v1/jobs")
    def create_path_job(request: PathJobRequest) -> dict[str, Any]:
        try:
            workbooks = [resolved_service.validate_local_workbook(path) for path in request.workbooks]
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        try:
            return resolved_service.run(
                stage=request.stage,
                queries=request.queries,
                workbooks=workbooks,
                embed=request.embed,
                sheets=request.sheets,
                persist=False,
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/v1/jobs/upload")
    async def create_upload_job(
        payload: str = Form(...),
        files: list[UploadFile] = File(...),
    ) -> dict[str, Any]:
        try:
            request = UploadJobRequest.model_validate_json(payload)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid payload JSON: {exc}") from exc
        if not files:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one workbook is required")
        with tempfile.TemporaryDirectory(prefix="table-agent-upload-") as upload_text:
            upload_dir = Path(upload_text)
            saved: list[Path] = []
            try:
                for index, upload in enumerate(files, start=1):
                    filename = safe_name(Path(upload.filename or f"workbook-{index}.xlsx").name)
                    target = upload_dir / filename
                    if target.exists():
                        target = upload_dir / f"{index:03d}_{filename}"
                    size = 0
                    with target.open("wb") as handle:
                        while chunk := await upload.read(1024 * 1024):
                            size += len(chunk)
                            if size > resolved_service.max_upload_bytes:
                                raise ValueError(
                                    f"Upload '{filename}' exceeds the "
                                    f"{resolved_service.max_upload_bytes // (1024 * 1024)} MB limit"
                                )
                            handle.write(chunk)
                    resolved_service._validate_workbook(target)
                    saved.append(target)
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            finally:
                for upload in files:
                    await upload.close()

            try:
                return await asyncio.to_thread(
                    resolved_service.run,
                    stage=request.stage,
                    queries=request.queries,
                    workbooks=saved,
                    embed=request.embed,
                    sheets=request.sheets,
                    persist=False,
                )
            except (RuntimeError, ValueError) as exc:
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


def _require_queries(stage: Stage, queries: list[str]) -> None:
    if stage in {"qa", "all"} and not any(str(query).strip() for query in queries):
        raise ValueError("At least one non-empty query is required for qa and all stages")


__all__ = ["PathJobRequest", "UploadJobRequest", "create_app"]
