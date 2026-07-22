from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator

from TableAgent.pipeline.common import safe_name
from service.runtime import Stage, TableAgentService


class PathJobRequest(BaseModel):
    stage: Stage = "all"
    queries: list[str] = Field(default_factory=list)
    workbooks: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_queries(self) -> "PathJobRequest":
        _require_queries(self.stage, self.queries)
        return self


class UploadJobRequest(BaseModel):
    stage: Stage = "all"
    queries: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_queries(self) -> "UploadJobRequest":
        _require_queries(self.stage, self.queries)
        return self


class JobManager:
    def __init__(self, service: TableAgentService):
        self.service = service
        self.executor = ThreadPoolExecutor(max_workers=service.max_workers, thread_name_prefix="table-agent")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()
        self._load_jobs()

    def submit(
        self,
        *,
        stage: Stage,
        queries: list[str],
        workbooks: list[Path],
        cleanup_dir: Path | None = None,
    ) -> str:
        job_id = uuid.uuid4().hex
        now = _utc_now()
        record = {
            "job_id": job_id,
            "stage": stage,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "workbook_count": len(workbooks),
            "query_count": len(queries),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = record
            self._persist(record)
            self._futures[job_id] = self.executor.submit(
                self._execute,
                job_id,
                stage,
                queries,
                workbooks,
                cleanup_dir,
            )
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record is not None else None

    def wait(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            raise KeyError(job_id)
        future.result()
        record = self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    def counts(self) -> dict[str, int]:
        counts = {name: 0 for name in ("queued", "running", "succeeded", "failed")}
        with self._lock:
            for record in self._jobs.values():
                counts[record["status"]] += 1
        return counts

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def _execute(
        self,
        job_id: str,
        stage: Stage,
        queries: list[str],
        workbooks: list[Path],
        cleanup_dir: Path | None,
    ) -> None:
        self._update(job_id, status="running")
        try:
            result = self.service.run(
                stage=stage,
                queries=queries,
                workbooks=workbooks,
                job_id=job_id,
            )
        except Exception as exc:
            self._update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        else:
            self._update(job_id, status="succeeded", result=result)
        finally:
            if cleanup_dir is not None:
                resolved = cleanup_dir.resolve()
                upload_root = (self.service.root_dir / "uploads").resolve()
                if resolved.is_relative_to(upload_root):
                    shutil.rmtree(resolved, ignore_errors=True)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.update(changes)
            record["updated_at"] = _utc_now()
            self._persist(record)

    def _persist(self, record: dict[str, Any]) -> None:
        job_dir = self._job_dir(str(record["job_id"]))
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / "status.json"
        staging = job_dir / ".status.json.tmp"
        staging.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        staging.replace(path)

    def _load_jobs(self) -> None:
        if not self.service.jobs_dir.is_dir():
            return
        for status_path in self.service.jobs_dir.glob("*/status.json"):
            try:
                record = json.loads(status_path.read_text(encoding="utf-8"))
                job_id = str(record["job_id"])
                job_dir = self._job_dir(job_id)
            except (KeyError, OSError, ValueError, json.JSONDecodeError, TypeError):
                continue
            if status_path.parent.resolve() != job_dir:
                continue
            if record.get("status") in {"queued", "running"}:
                record["status"] = "failed"
                record["error"] = "Service restarted before the job completed"
                record["updated_at"] = _utc_now()
                self._persist(record)
            self._jobs[job_id] = record

    def _job_dir(self, job_id: str) -> Path:
        if job_id != safe_name(job_id) or job_id in {".", ".."}:
            raise ValueError("Invalid job id")
        path = (self.service.jobs_dir / job_id).resolve()
        if not path.is_relative_to(self.service.jobs_dir.resolve()):
            raise ValueError("Invalid job id")
        return path


def create_app(
    service: TableAgentService | None = None,
    *,
    config_path: str | Path | None = None,
) -> FastAPI:
    resolved_service = service or TableAgentService.from_config(config_path or _default_config_path())
    manager = JobManager(resolved_service)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="TableAgent API",
        version=_package_version(),
        description="Asynchronous spreadsheet structure extraction and question answering.",
        lifespan=lifespan,
    )
    app.state.service = resolved_service
    app.state.jobs = manager

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
        ready = all(path.is_dir() for path in (resolved_service.root_dir, resolved_service.jobs_dir))
        if not ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service storage is unavailable")
        return {"status": "ready"}

    @app.get("/v1/status")
    def service_status() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": _package_version(),
            "workers": resolved_service.max_workers,
            "jobs": manager.counts(),
        }

    @app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    def create_path_job(
        request: PathJobRequest,
        wait: bool = Query(False),
    ) -> dict[str, Any]:
        try:
            workbooks = [resolved_service.validate_local_workbook(path) for path in request.workbooks]
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        job_id = manager.submit(stage=request.stage, queries=request.queries, workbooks=workbooks)
        return manager.wait(job_id) if wait else manager.get(job_id) or {"job_id": job_id}

    @app.post("/v1/jobs/upload", status_code=status.HTTP_202_ACCEPTED)
    async def create_upload_job(
        payload: str = Form(...),
        files: list[UploadFile] = File(...),
        wait: bool = Query(False),
    ) -> dict[str, Any]:
        try:
            request = UploadJobRequest.model_validate_json(payload)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid payload JSON: {exc}") from exc
        if not files:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one workbook is required")
        upload_dir = resolved_service.root_dir / "uploads" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=False)
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
                                f"Upload '{filename}' exceeds the {resolved_service.max_upload_bytes // (1024 * 1024)} MB limit"
                            )
                        handle.write(chunk)
                resolved_service._validate_workbook(target)
                saved.append(target)
        except (OSError, ValueError) as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        finally:
            for upload in files:
                await upload.close()

        job_id = manager.submit(
            stage=request.stage,
            queries=request.queries,
            workbooks=saved,
            cleanup_dir=upload_dir,
        )
        return await asyncio.to_thread(manager.wait, job_id) if wait else manager.get(job_id) or {"job_id": job_id}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        return record

    @app.get("/v1/jobs/{job_id}/artifacts")
    def list_artifacts(job_id: str) -> dict[str, Any]:
        if manager.get(job_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        job_dir = resolved_service.jobs_dir / job_id
        artifacts = sorted(path.relative_to(job_dir).as_posix() for path in job_dir.rglob("*") if path.is_file())
        return {"job_id": job_id, "artifacts": artifacts}

    @app.get("/v1/jobs/{job_id}/artifacts/{artifact_path:path}", response_class=FileResponse)
    def get_artifact(job_id: str, artifact_path: str) -> FileResponse:
        if manager.get(job_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        job_dir = (resolved_service.jobs_dir / job_id).resolve()
        path = (job_dir / artifact_path).resolve()
        if not path.is_relative_to(job_dir) or not path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
        return FileResponse(path)

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_queries(stage: Stage, queries: list[str]) -> None:
    if stage in {"qa", "all"} and not any(str(query).strip() for query in queries):
        raise ValueError("At least one non-empty query is required for qa and all stages")


__all__ = ["JobManager", "PathJobRequest", "UploadJobRequest", "create_app"]
