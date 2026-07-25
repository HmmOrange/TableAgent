from __future__ import annotations

import pickle
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest
import yaml

from TableAgent.configs import load_config
from TableAgent.pipeline import TableAgentPipeline
from TableAgent.pipeline.base import PipelineOutput
from TableAgent.llm import LLMResponse
from service.runtime import TableAgentService


class FakePipeline:
    instances = []

    def __init__(self, llm_client, layout_vlm_client, config):
        self.llm_client = llm_client
        self.layout_vlm_client = layout_vlm_client
        self.config = config
        self.prepared = []
        self.runs = []
        self.forces = []
        type(self).instances.append(self)

    def verify_samples(self, samples, force=False):
        self.forces.append(force)
        workbook_path = Path(samples[0].table_path.split(";")[0])
        workbook = openpyxl.load_workbook(workbook_path, read_only=True)
        try:
            sheet_names = list(workbook.sheetnames)
        finally:
            workbook.close()
        selected = samples[0].raw.get("selected_sheets") or sheet_names
        records = []
        for sheet_name in selected:
            structure_path = (
                Path(self.config["source_artifact_dir"])
                / "fake"
                / sheet_name
                / "structure.yaml"
            )
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_text(
                f"table1:\n  name: {sheet_name}\n  headers: []\n",
                encoding="utf-8",
            )
            records.append(
                SimpleNamespace(
                    workbook_path=workbook_path,
                    sheet_name=sheet_name,
                    structure_path=structure_path,
                    status="good",
                    cache_hit=False,
                )
            )
        return records

    def prepare_samples(self, samples):
        self.prepared.extend(samples)

    def run(self, sample):
        self.runs.append(sample)
        workbook_path = sample.table_path.split(";")[0]
        return PipelineOutput(
            sample_id=sample.sample_id,
            structured_table="table1:\n  headers: []\n",
            predicted_answer=f"answer: {sample.question}",
            latency=0.25,
            token_usage={"prompt": 4, "completion": 2},
            metadata={
                "workbook_path": workbook_path,
                "workbook_sheets": ["Sheet"],
                "verification": {"status": "good"},
                "qa": {"success": True, "artifacts": {"private": "path"}},
            },
        )


class FakeSummaryClient:
    def generate(self, prompt, system_prompt=None):
        description = "Workbook summary" if "workbook as a whole" in prompt else "Sheet summary"
        return LLMResponse(content=f'{{"description": "{description}"}}')


class FakeEmbeddingClient:
    model = "test-embedding"

    def __init__(self):
        self.calls = []

    async def encode(self, texts):
        values = list(texts) if isinstance(texts, list) else [texts]
        self.calls.append(values)
        return [[1.0, float(index + 1)] for index, _ in enumerate(values)]


class FakeIndexedPipeline:
    instances = []
    calls = []

    def __init__(self, llm_client, layout_vlm_client, config):
        self.llm_client = llm_client
        self.layout_vlm_client = layout_vlm_client
        self.config = config
        self.qa_agent = SimpleNamespace(
            run=lambda prompt: LLMResponse(content="combined indexed answer", prompt_tokens=2, completion_tokens=1)
        )
        type(self).instances.append(self)

    def _run_verified_qa(self, **kwargs):
        type(self).calls.append(kwargs)
        assert Path(kwargs["structure_path"]).is_file()
        return (
            LLMResponse(content="indexed answer", prompt_tokens=3, completion_tokens=2),
            {"success": True, "fallback_used": False, "replan_count": 0},
        )


def _workbook(path: Path) -> Path:
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = "value"
    workbook.save(path)
    workbook.close()
    return path


def _multi_sheet_workbook(path: Path) -> Path:
    workbook = openpyxl.Workbook()
    workbook.active.title = "Summary"
    workbook.create_sheet("Detail")
    workbook.create_sheet("Archive")
    workbook.save(path)
    workbook.close()
    return path


def test_service_runs_structure_once_and_answers_all_queries(tmp_path: Path):
    FakePipeline.instances = []
    answer_client = FakeSummaryClient()
    layout_client = object()
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=answer_client,
        layout_vlm_client=layout_client,
        pipeline_factory=FakePipeline,
    )

    result = service.run(
        stage="all",
        workbooks=[source],
        queries=["first question", "second question"],
        job_id="job-one",
    )

    assert len(FakePipeline.instances) == 2
    assert FakePipeline.instances[0].layout_vlm_client is layout_client
    assert FakePipeline.instances[1].llm_client is answer_client
    assert len(FakePipeline.instances[1].runs) == 2
    assert result["workbooks"] == ["book.xlsx"]
    assert result["structures"][0]["artifact"].endswith(".yaml")
    assert result["schema_artifacts"] == [
        {"workbook": "book.xlsx", "artifact": "workbooks/book.xlsx/schema.yaml"}
    ]
    assert result["metadata_artifacts"] == [
        {"workbook": "book.xlsx", "artifact": "workbooks/book.xlsx/metadata.json"}
    ]
    assert result["retrieval_artifacts"] == []
    assert [item["answer"] for item in result["answers"]] == [
        "answer: first question",
        "answer: second question",
    ]
    assert "artifacts" not in result["answers"][0]["qa"]
    assert (service.root_dir / "job-one" / "run.json").is_file()
    assert not (service.root_dir / "jobs").exists()
    assert not (service.root_dir / "inputs").exists()
    assert not (service.root_dir / "structure").exists()


def test_service_runs_structure_and_qa_units_concurrently(tmp_path: Path):
    state = {
        "structure_active": 0,
        "structure_max": 0,
        "qa_active": 0,
        "qa_max": 0,
    }
    state_lock = threading.Lock()
    instances = []

    def enter(phase: str) -> None:
        with state_lock:
            active_key = f"{phase}_active"
            max_key = f"{phase}_max"
            state[active_key] += 1
            state[max_key] = max(state[max_key], state[active_key])

    def leave(phase: str) -> None:
        with state_lock:
            state[f"{phase}_active"] -= 1

    class ConcurrentPipeline:
        def __init__(self, llm_client, layout_vlm_client, config):
            self.config = config
            instances.append(self)

        def verify_samples(self, samples, force=False):
            sample = samples[0]
            workbook_path = Path(sample.table_path)
            sheet_name = sample.raw["selected_sheets"][0]
            enter("structure")
            try:
                time.sleep(0.08)
                structure_dir = (
                    Path(self.config["source_artifact_dir"])
                    / workbook_path.stem
                    / sheet_name
                )
                structure_dir.mkdir(parents=True, exist_ok=True)
                structure_path = structure_dir / "structure.yaml"
                structure_path.write_text(
                    f"table1:\n  name: {workbook_path.stem}\n  headers: []\n",
                    encoding="utf-8",
                )
                return [
                    SimpleNamespace(
                        workbook_path=workbook_path,
                        sheet_name=sheet_name,
                        structure_path=structure_path,
                        status="good",
                        cache_hit=False,
                    )
                ]
            finally:
                leave("structure")

        def prepare_samples(self, samples):
            pass

        def run(self, sample):
            enter("qa")
            try:
                time.sleep(0.08)
                return PipelineOutput(
                    sample_id=sample.sample_id,
                    structured_table="table1: {}\n",
                    predicted_answer=f"answer: {sample.question}",
                    latency=0.08,
                    token_usage={"prompt": 1, "completion": 1},
                    metadata={
                        "workbook_path": sample.table_path.split(";")[0],
                        "workbook_sheets": ["Sheet"],
                        "verification": {"status": "good"},
                        "qa": {"success": True},
                    },
                )
            finally:
                leave("qa")

    source = _multi_sheet_workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        pipeline_factory=ConcurrentPipeline,
    )

    result = service.run(
        stage="all",
        workbooks=[source],
        queries=["first question", "second question"],
        max_workers=2,
        persist=False,
    )

    assert state["structure_max"] == 2
    assert state["qa_max"] == 2
    assert len(instances) == 5
    assert all(instance.config["max_workers"] == 2 for instance in instances)
    assert [(item["workbook"], item["sheet"]) for item in result["structures"]] == [
        ("book.xlsx", "Summary"),
        ("book.xlsx", "Detail"),
        ("book.xlsx", "Archive"),
    ]
    assert [item["answer"] for item in result["answers"]] == [
        "answer: first question",
        "answer: second question",
    ]


def test_structure_stage_always_generates_schema_and_metadata(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    embedding_client = FakeEmbeddingClient()
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        embedding_client=embedding_client,
        pipeline_factory=FakePipeline,
    )

    result = service.run(
        stage="structure",
        workbooks=[source],
        job_id="structure-artifacts",
        embed=True,
    )

    assert len(FakePipeline.instances) == 1
    assert result["schema_artifacts"][0]["artifact"] == "workbooks/book.xlsx/schema.yaml"
    assert result["metadata_artifacts"][0]["artifact"] == "workbooks/book.xlsx/metadata.json"
    assert result["retrieval_records"]
    assert result["retrieval_records"][0]["schema_yaml"]
    sheet_records = [record for record in result["retrieval_records"] if record["sheet"]]
    assert sheet_records
    assert all(record["structure_yaml"] for record in sheet_records)
    assert all(record["embedding"]["model"] == "test-embedding" for record in result["retrieval_records"])
    assert embedding_client.calls
    schema = yaml.safe_load(result["retrieval_records"][0]["schema_yaml"])
    assert schema["Sheet"]["structure"]["table1"]["name"] == "Sheet"
    persisted_records = pickle.loads(
        (
            service.root_dir
            / "structure-artifacts"
            / "workbooks"
            / "book.xlsx"
            / "retrieval_cards.pkl"
        ).read_bytes()
    )
    persisted_sheet_records = [record for record in persisted_records if record["sheet"]]
    assert all(record["structure_yaml"] for record in persisted_sheet_records)
    assert all(
        record["embedding"]["model"] == "test-embedding"
        for record in persisted_records
    )


def test_embed_requires_a_configured_real_embedding_provider(tmp_path: Path):
    service = TableAgentService({"service": {"root_dir": str(tmp_path / "service")}})

    with pytest.raises(ValueError, match="configured real embedding model"):
        service._retrieval_embedding_backend()


def test_embed_rejects_mock_embedding_provider(tmp_path: Path):
    service = TableAgentService(
        {
            "service": {"root_dir": str(tmp_path / "service")},
            "table_agent": {"retrieval_embedding_provider": "mock"},
        }
    )

    with pytest.raises(ValueError, match="mock embeddings are not allowed"):
        service._retrieval_embedding_backend()


def test_indexed_qa_uses_persisted_structures_without_layout_extraction(tmp_path: Path):
    FakeIndexedPipeline.instances = []
    FakeIndexedPipeline.calls = []
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        pipeline_factory=FakeIndexedPipeline,
    )

    result = service.run_indexed_qa(
        query="question",
        answer_instruction="Compare regions before answering.",
        expected_output="Return a markdown table.",
        workbooks=[source],
        qa_max_replans=2,
        artifacts=[
            {
                "id": "book:Sheet:table-1",
                "upload_name": "book.xlsx",
                "document_name": "book.xlsx",
                "score": 0.9,
                "sheet": "Sheet",
                "retrieval_level": "table",
                "retrieval_card": "Workbook: book.xlsx\nSheet: Sheet",
                "structure_yaml": "table1:\n  name: Sheet\n  sheet: Sheet\n  headers: []\n",
            }
        ],
    )

    assert len(FakeIndexedPipeline.instances) == 1
    assert FakeIndexedPipeline.instances[0].layout_vlm_client is None
    assert FakeIndexedPipeline.instances[0].config["qa_max_replans"] == 2
    assert "indexed_schema_text" not in FakeIndexedPipeline.calls[0]
    assert FakeIndexedPipeline.calls[0]["question"] == (
        "question\n\nAdditional answer instructions:\nCompare regions before answering."
    )
    assert FakeIndexedPipeline.calls[0]["expected_output"] == "Return a markdown table."
    assert result["answers"][0]["answer"] == "indexed answer"
    assert result["answers"][0]["answer_instruction"] == "Compare regions before answering."
    assert result["answers"][0]["expected_output"] == "Return a markdown table."
    assert result["answers"][0]["retrieval"]["mode"] == "indexed_vector_multi_candidate"


def test_instant_indexed_qa_disables_thinking_and_uses_instant_limits(tmp_path: Path):
    FakeIndexedPipeline.instances = []
    FakeIndexedPipeline.calls = []
    source = _workbook(tmp_path / "book.xlsx")
    answer_client = SimpleNamespace(
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        max_tokens=8192,
    )
    service = TableAgentService(
        {
            "service": {"root_dir": str(tmp_path / "service")},
            "table_agent": {
                "qa_max_retries": 3,
                "generation_max_tokens": 8192,
                "qa_instant_max_retries": 1,
                "qa_instant_generation_max_tokens": 2048,
            },
        },
        llm_client=answer_client,
        layout_vlm_client=object(),
        pipeline_factory=FakeIndexedPipeline,
    )

    service.run_indexed_qa(
        query="question",
        workbooks=[source],
        mode="instant",
        artifacts=[
            {
                "id": "book:Sheet:table-1",
                "upload_name": "book.xlsx",
                "sheet": "Sheet",
                "retrieval_card": "Workbook: book.xlsx\nSheet: Sheet",
                "structure_yaml": "table1:\n  name: Sheet\n  sheet: Sheet\n  headers: []\n",
            }
        ],
    )

    pipeline = FakeIndexedPipeline.instances[0]
    assert pipeline.llm_client is not answer_client
    assert pipeline.llm_client.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert answer_client.extra_body["chat_template_kwargs"]["enable_thinking"] is True
    assert pipeline.config["qa_max_retries"] == 1
    assert pipeline.config["generation_max_tokens"] == 2048


def test_thinking_indexed_qa_disables_reasoning_and_preserves_limits(tmp_path: Path):
    FakeIndexedPipeline.instances = []
    FakeIndexedPipeline.calls = []
    source = _workbook(tmp_path / "book.xlsx")
    answer_client = SimpleNamespace(
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        max_tokens=8192,
    )
    service = TableAgentService(
        {
            "service": {"root_dir": str(tmp_path / "service")},
            "table_agent": {
                "qa_max_retries": 3,
                "generation_max_tokens": 8192,
            },
        },
        llm_client=answer_client,
        layout_vlm_client=object(),
        pipeline_factory=FakeIndexedPipeline,
    )

    service.run_indexed_qa(
        query="question",
        workbooks=[source],
        mode="thinking",
        artifacts=[
            {
                "id": "book:Sheet:table-1",
                "upload_name": "book.xlsx",
                "sheet": "Sheet",
                "retrieval_card": "Workbook: book.xlsx\nSheet: Sheet",
                "structure_yaml": "table1:\n  name: Sheet\n  sheet: Sheet\n  headers: []\n",
            }
        ],
    )

    pipeline = FakeIndexedPipeline.instances[0]
    assert pipeline.llm_client is not answer_client
    assert pipeline.llm_client.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert answer_client.extra_body["chat_template_kwargs"]["enable_thinking"] is True
    assert pipeline.config["qa_max_retries"] == 3
    assert pipeline.config["generation_max_tokens"] == 8192


def test_real_indexed_qa_hybrid_routes_only_the_matching_workbook(tmp_path: Path, monkeypatch):
    sales = _workbook(tmp_path / "sales.xlsx")
    sales_book = openpyxl.load_workbook(sales)
    sales_book.create_sheet("Archive")
    sales_book.save(sales)
    sales_book.close()
    maintenance = _workbook(tmp_path / "maintenance.xlsx")
    maintenance_book = openpyxl.load_workbook(maintenance)
    maintenance_book.active["B1"] = "maintenance"
    maintenance_book.save(maintenance)
    maintenance_book.close()
    config = load_config("config.example.yaml")
    config["service"]["root_dir"] = str(tmp_path / "service")
    config["table_agent"]["retrieval_rerank_with_llm"] = False
    config["table_agent"]["retrieval_embedding_provider"] = "mock"
    qa_calls = []

    def run_verified_qa(_pipeline, **kwargs):
        qa_calls.append(
            {
                **kwargs,
                "structure_text": kwargs["structure_path"].read_text(
                    encoding="utf-8"
                ),
                "related_structure_texts": [
                    path.read_text(encoding="utf-8")
                    for path in kwargs["related_structure_paths"]
                ],
            }
        )
        return (
            LLMResponse(content="Sales answer", prompt_tokens=3, completion_tokens=2),
            {"success": True, "fallback_used": False, "replan_count": 0},
        )

    monkeypatch.setattr(TableAgentPipeline, "_run_verified_qa", run_verified_qa)
    service = TableAgentService(
        config,
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
    )

    selection = service.select_indexed_artifact(
        query="regional revenue score",
        mode="instant",
        artifacts=[
            {
                "id": "sales:summary",
                "document_id": "doc-sales",
                "upload_name": "sales.xlsx",
                "sheet": "Sheet",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "retrieval_card": "Regional revenue score and quarterly sales results",
                "structure_yaml": "table1:\n  sheet: Sheet\n  headers: []\n",
            },
            {
                "id": "maintenance:plan",
                "document_id": "doc-maintenance",
                "upload_name": "maintenance.xlsx",
                "sheet": "Sheet",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "retrieval_card": "Equipment maintenance schedule and spare parts",
                "structure_yaml": "table1:\n  sheet: Sheet\n  headers: []\n",
            },
            {
                "id": "sales:archive",
                "document_id": "doc-sales",
                "upload_name": "sales.xlsx",
                "sheet": "Archive",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "retrieval_card": "Historical discontinued products",
                "structure_yaml": "table1:\n  sheet: Archive\n  headers: []\n",
            },
        ],
    )
    assert selection["document_id"] == "doc-sales"
    assert selection["retrieval"]["candidate_count"] == 3

    result = service.run_indexed_qa(
        query="regional revenue score",
        workbooks=[sales, maintenance],
        qa_enable_final_review=False,
        artifacts=[
            {
                "id": "sales:summary",
                "document_id": "doc-sales",
                "upload_name": "sales.xlsx",
                "sheet": "Sheet",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "retrieval_card": "Regional revenue score and quarterly sales results",
                "structure_yaml": "table1:\n  sheet: Sheet\n  headers: []\n",
            },
            {
                "id": "maintenance:plan",
                "document_id": "doc-maintenance",
                "upload_name": "maintenance.xlsx",
                "sheet": "Sheet",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "retrieval_card": "Equipment maintenance schedule and spare parts",
                "structure_yaml": "table1:\n  sheet: Sheet\n  headers: []\n",
            },
            {
                "id": "sales:archive",
                "document_id": "doc-sales",
                "upload_name": "sales.xlsx",
                "sheet": "Archive",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "retrieval_card": "Historical discontinued products",
                "structure_yaml": "table1:\n  sheet: Archive\n  headers: []\n",
            },
        ],
    )

    answer = result["answers"][0]
    assert len(qa_calls) == 1
    assert qa_calls[0]["workbook_path"].name == "sales.xlsx"
    assert qa_calls[0]["structure_text"] == (
        "table1:\n  sheet: Sheet\n  headers: []\n"
    )
    assert len(qa_calls[0]["related_structure_paths"]) == 1
    assert qa_calls[0]["related_structure_texts"] == [
        "table1:\n  sheet: Archive\n  headers: []\n"
    ]
    assert qa_calls[0]["enable_final_answer_review"] is False
    assert answer["workbook"] == "sales.xlsx"
    assert answer["workbooks"] == ["sales.xlsx"]
    assert answer["sheets"] == ["Sheet", "Archive"]
    assert answer["retrieval"]["mode"] == "table_agent_hybrid"
    assert answer["retrieval"]["document_id"] == "doc-sales"
    assert answer["retrieval"]["embedding_used"] is True
    assert answer["retrieval"]["candidate_count"] == 3
    assert answer["retrieval"]["workbook_count"] == 2
    assert sum(bool(row["selected"]) for row in answer["retrieval"]["audit"]) == 1


def test_real_indexed_qa_hybrid_runs_top_k_distinct_workbooks(tmp_path: Path, monkeypatch):
    sales = _workbook(tmp_path / "sales.xlsx")
    maintenance = _workbook(tmp_path / "maintenance.xlsx")
    maintenance_book = openpyxl.load_workbook(maintenance)
    maintenance_book.active["B1"] = "maintenance"
    maintenance_book.save(maintenance)
    maintenance_book.close()
    config = load_config("config.example.yaml")
    config["service"]["root_dir"] = str(tmp_path / "service")
    config["table_agent"]["retrieval_rerank_with_llm"] = False
    config["table_agent"]["retrieval_embedding_provider"] = "mock"
    qa_workbooks = []
    qa_calls = []

    def run_verified_qa(_pipeline, **kwargs):
        workbook_name = kwargs["workbook_path"].name
        qa_workbooks.append(workbook_name)
        qa_calls.append(kwargs)
        return (
            LLMResponse(
                content=f"Evidence from {workbook_name}",
                prompt_tokens=3,
                completion_tokens=2,
            ),
            {"success": True, "fallback_used": False, "replan_count": 0},
        )

    monkeypatch.setattr(TableAgentPipeline, "_run_verified_qa", run_verified_qa)
    service = TableAgentService(
        config,
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
    )

    result = service.run_indexed_qa(
        query="compare revenue with maintenance",
        workbooks=[sales, maintenance],
        retrieval_top_k=2,
        qa_enable_final_review=False,
        artifacts=[
            {
                "id": "sales:summary",
                "upload_name": "sales.xlsx",
                "sheet": "Sheet",
                "retrieval_card": "Revenue and sales results",
                "structure_yaml": "table1:\n  sheet: Sheet\n  headers: []\n",
            },
            {
                "id": "maintenance:plan",
                "upload_name": "maintenance.xlsx",
                "sheet": "Sheet",
                "retrieval_card": "Maintenance schedule and costs",
                "structure_yaml": "table1:\n  sheet: Sheet\n  headers: []\n",
            },
        ],
    )

    answer = result["answers"][0]
    assert set(qa_workbooks) == {"sales.xlsx", "maintenance.xlsx"}
    assert all(
        "Workbook-specific evidence pass:" in call["question"]
        for call in qa_calls
    )
    assert all(call["expected_output"] == "" for call in qa_calls)
    assert all(call["enable_final_answer_review"] is False for call in qa_calls)
    assert set(answer["workbooks"]) == {"sales.xlsx", "maintenance.xlsx"}
    assert answer["retrieval"]["mode"] == "table_agent_hybrid_top_k"
    assert answer["retrieval"]["top_k_requested"] == 2
    assert len(answer["retrieval"]["groups"]) == 2
    assert answer["answer"]


def test_indexed_qa_falls_back_when_selected_artifact_has_no_structure(tmp_path: Path):
    source = _workbook(tmp_path / "book.xlsx")
    config = load_config("config.example.yaml")
    config["service"]["root_dir"] = str(tmp_path / "service")
    config["table_agent"]["retrieval_rerank_with_llm"] = False
    config["table_agent"]["retrieval_embedding_provider"] = "mock"
    service = TableAgentService(
        config,
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
    )

    result = service.run_indexed_qa(
        query="regional revenue score",
        workbooks=[source],
        artifacts=[
            {
                "id": "book:sheet:table-1",
                "upload_name": "book.xlsx",
                "sheet": "Sheet",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "retrieval_card": "Regional revenue score and quarterly sales results",
            }
        ],
    )

    answer = result["answers"][0]
    assert answer["answer"]
    assert answer["qa"]["success"] is False
    assert answer["qa"]["fallback_used"] is True
    assert answer["qa"]["fallback_source"] == "missing_structure"
    assert "Missing structure.yaml" in answer["qa"]["error"]


def test_instant_indexed_retrieval_rejects_unrelated_candidates(tmp_path: Path):
    config = load_config("config.example.yaml")
    config["service"]["root_dir"] = str(tmp_path / "service")
    config["table_agent"]["retrieval_rerank_with_llm"] = True
    config["table_agent"]["retrieval_embedding_provider"] = "mock"
    service = TableAgentService(
        config,
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
    )

    selection = service.select_indexed_artifact(
        query="name in the CV",
        mode="instant",
        artifacts=[
            {
                "id": "maintenance:plan",
                "document_id": "doc-maintenance",
                "upload_name": "maintenance.xlsx",
                "sheet": "Plan",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "retrieval_card": "Equipment maintenance schedule and spare parts",
                "structure_yaml": "table1:\n  sheet: Plan\n  headers: []\n",
            }
        ],
    )

    assert selection["status"] == "no_evidence"
    assert selection["document_id"] is None
    assert selection["retrieval"]["selected_artifact_id"] is None
    assert selection["retrieval"]["audit"][0]["selected"] is False


def test_qa_stage_generates_fresh_structure_before_answering(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        pipeline_factory=FakePipeline,
    )

    result = service.run(stage="qa", workbooks=[source], queries=["question"])

    assert len(FakePipeline.instances) == 2
    assert FakePipeline.instances[0].layout_vlm_client is not None
    assert FakePipeline.instances[0].forces == [True]
    assert FakePipeline.instances[1].layout_vlm_client is None
    assert result["answers"][0]["answer"] == "answer: question"


def test_qa_stage_accepts_per_run_max_replans_override(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {
            "service": {"root_dir": str(tmp_path / "service")},
            "table_agent": {"qa_max_replans": 5},
        },
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        pipeline_factory=FakePipeline,
    )

    service.run(
        stage="qa",
        workbooks=[source],
        queries=["question"],
        qa_max_replans=2,
    )

    assert FakePipeline.instances[1].config["qa_max_replans"] == 2


def test_service_rejects_negative_qa_max_replans(tmp_path: Path):
    service = TableAgentService({"service": {"root_dir": str(tmp_path / "service")}})
    source = _workbook(tmp_path / "book.xlsx")

    try:
        service.run(
            stage="qa",
            workbooks=[source],
            queries=["question"],
            qa_max_replans=-1,
        )
    except ValueError as exc:
        assert "qa_max_replans" in str(exc)
    else:
        raise AssertionError("Expected a negative qa_max_replans validation error")


def test_service_always_regenerates_structure_in_a_fresh_workspace(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        pipeline_factory=FakePipeline,
    )

    service.run(stage="structure", workbooks=[source])

    assert FakePipeline.instances[0].forces == [True]


def test_service_generates_readable_timestamp_job_id(tmp_path: Path):
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        pipeline_factory=FakePipeline,
    )

    result = service.run(stage="structure", workbooks=[source])

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{6}Z", result["job_id"])


def test_service_normalizes_repeated_comma_separated_sheet_filters(tmp_path: Path):
    FakePipeline.instances = []
    source = _multi_sheet_workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        pipeline_factory=FakePipeline,
    )

    result = service.run(
        stage="structure",
        workbooks=[source],
        sheets=["Summary, Detail", "Summary"],
        job_id="selected-sheets",
    )

    assert [item["sheet"] for item in result["structures"]] == ["Summary", "Detail"]
    schema_path = service.root_dir / "selected-sheets" / result["schema_artifacts"][0]["artifact"]
    assert list(yaml.safe_load(schema_path.read_text(encoding="utf-8"))) == ["Summary", "Detail"]


def test_ephemeral_service_run_returns_embedded_retrieval_cards_without_persisting(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    output_root = tmp_path / "output"
    service = TableAgentService(
        {"service": {"root_dir": str(output_root)}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        embedding_client=FakeEmbeddingClient(),
        pipeline_factory=FakePipeline,
    )

    result = service.run(stage="structure", workbooks=[source], embed=True, persist=False)

    assert result["artifacts"] == []
    assert result["structures"][0]["artifact"] is None
    assert "table1" in result["schema_artifacts"][0]["schema"]
    assert result["metadata_artifacts"][0]["metadata"]["name"] == "book.xlsx"
    retrieval = result["retrieval_artifacts"][0]
    assert retrieval["workbook"] == "book.xlsx"
    assert retrieval["retrieval_cards"]
    for card in retrieval["retrieval_cards"]:
        embedding = card["embedding"]
        assert embedding["model"] == "test-embedding"
        assert embedding["dimension"] == 2
        assert len(embedding["values"]) == 2
    assert not output_root.exists()


def test_persisted_service_run_returns_embedded_retrieval_artifact_path(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "output")}},
        llm_client=FakeSummaryClient(),
        layout_vlm_client=object(),
        embedding_client=FakeEmbeddingClient(),
        pipeline_factory=FakePipeline,
    )

    result = service.run(stage="structure", workbooks=[source], embed=True, job_id="embedded")

    retrieval = result["retrieval_artifacts"][0]
    assert retrieval == {
        "workbook": "book.xlsx",
        "artifact": "workbooks/book.xlsx/retrieval_cards.pkl",
    }
    assert (service.root_dir / "embedded" / retrieval["artifact"]).is_file()


def test_service_deletes_selected_or_all_saved_runs(tmp_path: Path):
    root = tmp_path / "output"
    service = TableAgentService({"service": {"root_dir": str(root)}})
    for run_id in ("run-one", "run-two"):
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(f'{{"job_id": "{run_id}"}}', encoding="utf-8")
    unrelated = root / "unrelated"
    unrelated.mkdir()

    selected = service.delete_runs(["run-one", "missing"])
    all_runs = service.delete_runs(all_runs=True)

    assert selected == {"deleted": ["run-one"], "missing": ["missing"]}
    assert all_runs == {"deleted": ["run-two"], "missing": []}
    assert unrelated.is_dir()


def test_service_rejects_missing_sheet_before_pipeline_work(tmp_path: Path):
    FakePipeline.instances = []
    source = _workbook(tmp_path / "book.xlsx")
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        pipeline_factory=FakePipeline,
    )

    try:
        service.run(
            stage="structure",
            workbooks=[source],
            sheets=["Missing"],
        )
    except ValueError as exc:
        assert "book.xlsx: Missing" in str(exc)
    else:
        raise AssertionError("Expected a missing-sheet validation error")

    assert FakePipeline.instances == []


def test_service_rejects_queries_missing_from_qa_stage(tmp_path: Path):
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_client=object(),
        pipeline_factory=FakePipeline,
    )
    source = _workbook(tmp_path / "book.xlsx")

    try:
        service.run(stage="qa", workbooks=[source], queries=[])
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("Expected an empty-query validation error")


def test_local_paths_are_disabled_by_default(tmp_path: Path):
    service = TableAgentService({"service": {"root_dir": str(tmp_path / "service")}})
    source = _workbook(tmp_path / "book.xlsx")

    try:
        service.validate_local_workbook(source)
    except PermissionError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("Expected local paths to be disabled")


def test_service_uses_explicit_model_profiles(monkeypatch, tmp_path: Path):
    calls = []

    def fake_create_model_client(config, *, kind, profile):
        calls.append((kind, profile))
        return object()

    monkeypatch.setattr("service.runtime.create_model_client", fake_create_model_client)
    service = TableAgentService(
        {"service": {"root_dir": str(tmp_path / "service")}},
        llm_profile="alternate_answer",
        vlm_profile="alternate_layout",
    )

    service._answer_client()
    service._layout_client()

    assert calls == [
        ("llm", "alternate_answer"),
        ("vlm", "alternate_layout"),
    ]
