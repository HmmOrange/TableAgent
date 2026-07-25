from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from service.api import create_app


class FakeService:
    def __init__(self, root: Path):
        self.root_dir = root
        self.max_workers = 1
        self.max_upload_bytes = 1024 * 1024
        self.api_key = None
        self.calls = []
        self.accept_local_paths = False
        self.indexed_calls = []

    def run(self, *, stage, queries, workbooks, embed, sheets, qa_max_replans, persist):
        workbook_names = [Path(path).name for path in workbooks]
        self.calls.append(
            {
                "stage": stage,
                "queries": queries,
                "embed": embed,
                "sheets": sheets,
                "qa_max_replans": qa_max_replans,
                "persist": persist,
                "workbooks": [str(path) for path in workbooks],
            }
        )
        return {
            "job_id": "ephemeral-run",
            "stage": stage,
            "workbooks": workbook_names,
            "structures": [],
            "answers": [{"query": query, "answer": "ok"} for query in queries],
            "retrieval_artifacts": (
                [
                    {
                        "workbook": workbook_names[0],
                        "retrieval_cards": [
                            {
                                "id": f"{workbook_names[0]}:metadata",
                                "embedding": {"model": "mock-hash-embedding", "dimension": 1, "values": [1.0]},
                            }
                        ],
                    }
                ]
                if embed
                else []
            ),
            "artifacts": [],
        }

    def validate_local_workbook(self, value):
        if not self.accept_local_paths:
            raise PermissionError("Server-side workbook paths are disabled; upload the workbook instead")
        return Path(value)

    def run_indexed_qa(
        self,
        *,
        query,
        workbooks,
        artifacts,
        answer_instruction,
        expected_output,
        retrieval_top_k,
        qa_max_replans,
        qa_enable_final_review,
        mode,
    ):
        self.indexed_calls.append(
            {
                "query": query,
                "workbooks": [str(path) for path in workbooks],
                "artifacts": artifacts,
                "answer_instruction": answer_instruction,
                "expected_output": expected_output,
                "retrieval_top_k": retrieval_top_k,
                "qa_max_replans": qa_max_replans,
                "qa_enable_final_review": qa_enable_final_review,
                "mode": mode,
            }
        )
        return {"answers": [{"query": query, "answer": "indexed"}], "artifacts": []}

    def select_indexed_artifact(self, *, query, artifacts, mode):
        self.selection_mode = mode
        return {
            "selected_artifact_id": artifacts[0]["id"],
            "document_id": artifacts[0].get("document_id", ""),
            "workbook": artifacts[0].get("upload_name", ""),
            "sheet": artifacts[0].get("sheet", ""),
            "retrieval": {"mode": "table_agent_hybrid"},
        }

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
            "/v1/jobs/upload",
            data={
                "payload": (
                    '{"stage":"all","queries":["question"],'
                    '"embed":true,"sheets":["Summary,Detail","Archive"],'
                    '"qa_max_replans":2}'
                )
            },
            files={"files": ("book.xlsx", b"workbook-bytes", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        body = response.json()

        assert response.status_code == 200
        assert body["answers"][0]["answer"] == "ok"
        assert body["retrieval_artifacts"][0]["retrieval_cards"][0]["embedding"]["values"] == [1.0]
        assert client.get("/v1/status").json()["persistence"] is False
        assert service.calls[0]["stage"] == "all"
        assert service.calls[0]["queries"] == ["question"]
        assert service.calls[0]["embed"] is True
        assert service.calls[0]["sheets"] == ["Summary,Detail", "Archive"]
        assert service.calls[0]["qa_max_replans"] == 2
        assert service.calls[0]["persist"] is False
        uploaded_path = Path(service.calls[0]["workbooks"][0])
        assert client.get("/v1/jobs/ephemeral-run").status_code == 404
        assert client.get("/v1/jobs/ephemeral-run/artifacts").status_code == 404
    assert not uploaded_path.exists()
    assert not service.root_dir.exists()


def test_server_side_paths_are_forbidden_by_default(tmp_path: Path):
    app = create_app(FakeService(tmp_path / "service"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            json={"stage": "structure", "queries": [], "workbooks": [str(tmp_path / "book.xlsx")]},
        )

    assert response.status_code == 403


def test_indexed_retrieval_select_endpoint(tmp_path: Path):
    app = create_app(FakeService(tmp_path / "service"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/retrieval/select",
            json={
                "query": "revenue",
                "artifacts": [
                    {
                        "id": "sales:summary",
                        "document_id": "doc-sales",
                        "upload_name": "sales.xlsx",
                        "sheet": "Summary",
                    }
                ],
                "mode": "instant",
            },
        )

    assert response.status_code == 200
    assert response.json()["document_id"] == "doc-sales"
    assert response.json()["retrieval"]["mode"] == "table_agent_hybrid"
    assert app.state.service.selection_mode == "instant"


def test_upload_job_routes_indexed_artifacts_to_qa_only_runtime(tmp_path: Path):
    service = FakeService(tmp_path / "service")
    app = create_app(service)
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/upload",
            data={
                "payload": (
                    '{"stage":"qa","queries":["question"],"qa_max_replans":2,'
                    '"mode":"instant",'
                    '"answer_instruction":"Compare regions",'
                    '"expected_output":"Return a markdown table",'
                    '"retrieval_top_k":3,'
                    '"qa_enable_final_review":false,"artifacts":'
                    '[{"upload_name":"book.xlsx","sheet":"Sheet","structure_yaml":"table1: {}"}]}'
                )
            },
            files={"files": ("book.xlsx", b"workbook-bytes", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    assert response.status_code == 200
    assert response.json()["answers"][0]["answer"] == "indexed"
    assert not service.calls
    assert service.indexed_calls[0]["query"] == "question"
    assert service.indexed_calls[0]["answer_instruction"] == "Compare regions"
    assert service.indexed_calls[0]["expected_output"] == "Return a markdown table"
    assert service.indexed_calls[0]["retrieval_top_k"] == 3
    assert service.indexed_calls[0]["qa_max_replans"] == 2
    assert service.indexed_calls[0]["qa_enable_final_review"] is False
    assert service.indexed_calls[0]["mode"] == "instant"


def test_indexed_upload_job_rejects_multiple_queries(tmp_path: Path):
    app = create_app(FakeService(tmp_path / "service"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs/upload",
            data={
                "payload": (
                    '{"stage":"qa","queries":["one","two"],"artifacts":'
                    '[{"upload_name":"book.xlsx","sheet":"Sheet","structure_yaml":"table1: {}"}]}'
                )
            },
            files={"files": ("book.xlsx", b"workbook-bytes", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]


def test_path_jobs_forward_artifact_and_sheet_options(tmp_path: Path):
    service = FakeService(tmp_path / "service")
    service.accept_local_paths = True
    app = create_app(service)

    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            json={
                "stage": "structure",
                "queries": [],
                "workbooks": [str(tmp_path / "book.xlsx")],
                "embed": True,
                "sheets": ["Summary,Detail"],
                "qa_max_replans": 0,
            },
        )

    assert response.status_code == 200
    assert service.calls[0]["stage"] == "structure"
    assert service.calls[0]["queries"] == []
    assert service.calls[0]["embed"] is True
    assert service.calls[0]["sheets"] == ["Summary,Detail"]
    assert service.calls[0]["qa_max_replans"] == 0
    assert service.calls[0]["persist"] is False


def test_all_stage_requires_a_query(tmp_path: Path):
    app = create_app(FakeService(tmp_path / "service"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            json={"stage": "all", "queries": [], "workbooks": [str(tmp_path / "book.xlsx")]},
        )

    assert response.status_code == 422


def test_api_rejects_negative_qa_max_replans(tmp_path: Path):
    app = create_app(FakeService(tmp_path / "service"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/jobs",
            json={
                "stage": "qa",
                "queries": ["question"],
                "workbooks": [str(tmp_path / "book.xlsx")],
                "qa_max_replans": -1,
            },
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
