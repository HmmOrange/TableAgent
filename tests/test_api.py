from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from service.api import create_app


class FakeService:
    def __init__(self, root: Path):
        self.root_dir = root
        self.jobs_dir = root / "jobs"
        self.jobs_dir.mkdir(parents=True)
        self.max_workers = 1
        self.max_upload_bytes = 1024 * 1024
        self.api_key = None
        self.calls = []
        self.accept_local_paths = False

    def run(self, *, stage, queries, workbooks, schema, metadata, sheets, job_id):
        self.calls.append(
            {
                "stage": stage,
                "queries": queries,
                "schema": schema,
                "metadata": metadata,
                "sheets": sheets,
            }
        )
        return {
            "job_id": job_id,
            "stage": stage,
            "workbooks": [Path(path).name for path in workbooks],
            "structures": [],
            "answers": [{"query": query, "answer": "ok"} for query in queries],
            "artifacts": [],
        }

    def validate_local_workbook(self, value):
        if not self.accept_local_paths:
            raise PermissionError("Server-side workbook paths are disabled; upload the workbook instead")
        return Path(value)

    @staticmethod
    def _validate_workbook(path: Path):
        if path.suffix.lower() != ".xlsx":
            raise ValueError("Unsupported workbook")


def test_health_status_and_upload_job(tmp_path: Path):
    service = FakeService(tmp_path / "service")
    app = create_app(service)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/health/ready").json()["status"] == "ready"

        response = client.post(
            "/v1/jobs/upload?wait=true",
            data={
                "payload": (
                    '{"stage":"all","queries":["question"],"schema":true,'
                    '"metadata":true,"sheets":["Summary,Detail","Archive"]}'
                )
            },
            files={"files": ("book.xlsx", b"workbook-bytes", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        body = response.json()

        assert response.status_code == 202
        assert body["status"] == "succeeded"
        assert body["result"]["answers"][0]["answer"] == "ok"
        job_id = body["job_id"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}Z", job_id)
        assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "succeeded"
        assert client.get("/v1/status").json()["jobs"]["succeeded"] == 1
        assert service.calls == [
            {
                "stage": "all",
                "queries": ["question"],
                "schema": True,
                "metadata": True,
                "sheets": ["Summary,Detail", "Archive"],
            }
        ]


def test_server_side_paths_are_forbidden_by_default(tmp_path: Path):
    app = create_app(FakeService(tmp_path / "service"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            json={"stage": "structure", "queries": [], "workbooks": [str(tmp_path / "book.xlsx")]},
        )

    assert response.status_code == 403


def test_path_jobs_forward_artifact_and_sheet_options(tmp_path: Path):
    service = FakeService(tmp_path / "service")
    service.accept_local_paths = True
    app = create_app(service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs?wait=true",
            json={
                "stage": "structure",
                "queries": [],
                "workbooks": [str(tmp_path / "book.xlsx")],
                "schema": True,
                "metadata": False,
                "sheets": ["Summary,Detail"],
            },
        )

    assert response.status_code == 202
    assert service.calls == [
        {
            "stage": "structure",
            "queries": [],
            "schema": True,
            "metadata": False,
            "sheets": ["Summary,Detail"],
        }
    ]


def test_all_stage_requires_a_query(tmp_path: Path):
    app = create_app(FakeService(tmp_path / "service"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            json={"stage": "all", "queries": [], "workbooks": [str(tmp_path / "book.xlsx")]},
        )

    assert response.status_code == 422


def test_v1_endpoints_support_optional_api_key_authentication(tmp_path: Path):
    service = FakeService(tmp_path / "service")
    service.api_key = "test-key"
    app = create_app(service)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/status").status_code == 401
        response = client.get("/v1/status", headers={"X-API-Key": "test-key"})

    assert response.status_code == 200
