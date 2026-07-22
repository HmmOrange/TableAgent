import tempfile
import json
from pathlib import Path
import pytest
from TableAgent.configs import TableAgentConfig
from TableAgent.pipeline.retrieval import SourceRetriever
from TableAgent.schema import EvalSample
from TableAgent.llm import BaseLLM, LLMResponse

class FakeLLM(BaseLLM):
    def __init__(self, response_content: str = "selected_index: 0\nrationale: test"):
        self.response_content = response_content

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> LLMResponse:
        return LLMResponse(content=self.response_content, prompt_tokens=10, completion_tokens=10)


@pytest.fixture(autouse=True)
def resolved_table_agent_config(monkeypatch):
    from TableAgent.configs import load_config
    real_from_config = TableAgentConfig.from_config

    def resolve(config=None):
        merged = dict(load_config("config.example.yaml")["table_agent"])
        explicit = config or {}
        if "table_agent" in explicit:
            explicit = explicit["table_agent"]
        merged.update(explicit)
        return real_from_config(merged)

    monkeypatch.setattr(TableAgentConfig, "from_config", staticmethod(resolve))

@pytest.fixture
def temp_sources_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sources_path = tmp_path / "sources"
        sources_path.mkdir()
        
        # Create candidate 1 (Sheet1)
        c1_dir = sources_path / "dummy_Sheet1"
        c1_dir.mkdir()
        (c1_dir / "metadata.json").write_text(json.dumps({
            "workbook_path": "/path/to/dummy.xlsx",
            "sheet_name": "Sheet1",
            "layout_workflow_version": 4,
        }), encoding="utf-8")
        (c1_dir / "structure.yaml").write_text("""
table1:
  id: t1
  name: Equipment List
  description: List of items.
  headers:
    - id: item_id
      label: Item ID
      orientation: column
""", encoding="utf-8")
        (c1_dir / "sheet_text.txt").write_text("dry wet equipment list", encoding="utf-8")
        (c1_dir / "table.png").touch()
        (c1_dir / "table.html").touch()
        
        # Create candidate 2 (Sheet2)
        c2_dir = sources_path / "dummy_Sheet2"
        c2_dir.mkdir()
        (c2_dir / "metadata.json").write_text(json.dumps({
            "workbook_path": "/path/to/dummy.xlsx",
            "sheet_name": "Sheet2",
            "layout_workflow_version": 4,
        }), encoding="utf-8")
        (c2_dir / "structure.yaml").write_text("""
table1:
  id: t2
  name: Maintenance Plan
  description: Plan for maintenance.
  headers:
    - id: plan_id
      label: Plan ID
      orientation: column
""", encoding="utf-8")
        (c2_dir / "sheet_text.txt").write_text("maintenance statistics and plan", encoding="utf-8")
        (c2_dir / "table.png").touch()
        (c2_dir / "table.html").touch()
        
        yield sources_path

def test_hybrid_retrieval_fallback_to_lexical(temp_sources_dir):
    config = TableAgentConfig.from_config({
        "artifact_dir": str(temp_sources_dir.parent),
        "source_artifact_dir": str(temp_sources_dir.parent),
        "retrieval_rerank_with_llm": False,
        "retrieval_top_k": 3,
        "retrieval_candidate_max_chars": 1000,
    })
    llm = FakeLLM()
    retriever = SourceRetriever(config, llm, None, None)
    
    sample = EvalSample(
        index=0,
        sample_id="siflex-test",
        table_id="test",
        table_content="",
        question="maintenance plan",
        answer=[],
        sample_path="siflex",
        table_path="/path/to/dummy.xlsx",
        raw={}
    )
    
    candidates = retriever.load_candidates(sample)
    assert len(candidates) == 2
    # Candidate with "maintenance plan" should rank higher on lexical
    assert candidates[0].sheet_name == "Sheet2"
    assert candidates[0].lexical_score > 0
    assert not candidates[0].embedding_used
    assert candidates[0].embedding_score == 0.0

def test_hybrid_retrieval_with_mock_embedding(temp_sources_dir):
    config = TableAgentConfig.from_config({
        "artifact_dir": str(temp_sources_dir.parent),
        "source_artifact_dir": str(temp_sources_dir.parent),
        "retrieval_rerank_with_llm": False,
        "retrieval_top_k": 3,
        "retrieval_candidate_max_chars": 1000,
        "retrieval_embedding_provider": "mock",
        "retrieval_lexical_weight": 0.5,
        "retrieval_embedding_weight": 0.5,
    })
    llm = FakeLLM()
    retriever = SourceRetriever(config, llm, None, None)
    
    sample = EvalSample(
        index=0,
        sample_id="siflex-test",
        table_id="test",
        table_content="",
        question="equipment",
        answer=[],
        sample_path="siflex",
        table_path="/path/to/dummy.xlsx",
        raw={}
    )
    
    candidates = retriever.load_candidates(sample)
    assert len(candidates) == 2
    assert candidates[0].embedding_used
    assert candidates[0].embedding_score > 0
    # "equipment" is in c1_dir structure/text, so Sheet1 should rank first
    assert candidates[0].sheet_name == "Sheet1"

def test_no_provider_does_not_instantiate_live_embedding(temp_sources_dir):
    config = TableAgentConfig.from_config({
        "artifact_dir": str(temp_sources_dir.parent),
        "source_artifact_dir": str(temp_sources_dir.parent),
        "retrieval_rerank_with_llm": False,
        "retrieval_top_k": 3,
        "retrieval_candidate_max_chars": 1000,
        "retrieval_embedding_provider": None,
    })
    llm = FakeLLM()

    from TableAgent.pipeline import retrieval

    mock_embedding_called = False

    class DummyEmbedding:
        async def encode(self, texts):
            import numpy as np
            return np.zeros((len(texts), 128), dtype=np.float32)

    def fake_from_config(config, provider=None):
        nonlocal mock_embedding_called
        mock_embedding_called = True
        return DummyEmbedding()

    original_from_config = retrieval.OpenAICompatibleEmbeddingClient.from_config
    retrieval.OpenAICompatibleEmbeddingClient.from_config = staticmethod(fake_from_config)
    try:
        retriever = SourceRetriever(config, llm, None, None)
        assert retriever.embedding_client is None
        assert not mock_embedding_called

        config_live = TableAgentConfig.from_config({
            "artifact_dir": str(temp_sources_dir.parent),
            "source_artifact_dir": str(temp_sources_dir.parent),
            "retrieval_rerank_with_llm": False,
            "retrieval_top_k": 3,
            "retrieval_candidate_max_chars": 1000,
            "retrieval_embedding_provider": "default",
        })
        retriever_live = SourceRetriever(config_live, llm, None, None)
        assert retriever_live.embedding_client is None
        assert not mock_embedding_called

        injected = SourceRetriever(config_live, llm, None, None, embedding_client=DummyEmbedding())
        assert injected.embedding_client is not None
    finally:
        retrieval.OpenAICompatibleEmbeddingClient.from_config = original_from_config

def test_top_level_headers_structure_included(temp_sources_dir):
    c3_dir = temp_sources_dir / "dummy_Sheet3"
    c3_dir.mkdir()
    (c3_dir / "metadata.json").write_text(json.dumps({
        "workbook_path": "/path/to/dummy.xlsx",
        "sheet_name": "Sheet3",
        "layout_workflow_version": 4,
    }), encoding="utf-8")
    (c3_dir / "structure.yaml").write_text("""
headers:
  - id: item_price
    label: Price Column
    description: Unit price of item.
    orientation: column
""", encoding="utf-8")
    (c3_dir / "sheet_text.txt").write_text("simple text", encoding="utf-8")
    (c3_dir / "table.png").touch()
    (c3_dir / "table.html").touch()

    config = TableAgentConfig.from_config({
        "artifact_dir": str(temp_sources_dir.parent),
        "source_artifact_dir": str(temp_sources_dir.parent),
        "retrieval_rerank_with_llm": False,
        "retrieval_top_k": 3,
        "retrieval_candidate_max_chars": 1000,
    })
    llm = FakeLLM()
    retriever = SourceRetriever(config, llm, None, None)
    
    sample = EvalSample(
        index=0,
        sample_id="siflex-test",
        table_id="test",
        table_content="",
        question="Price Column",
        answer=[],
        sample_path="siflex",
        table_path="/path/to/dummy.xlsx",
        raw={}
    )
    
    candidates = retriever.load_candidates(sample)
    sheet3_cand = next(c for c in candidates if c.sheet_name == "Sheet3")
    assert "Price Column" in sheet3_cand.structure_text
    assert sheet3_cand.lexical_score > 0


def test_source_retriever_ranks_tables_within_one_sheet(temp_sources_dir):
    source_dir = temp_sources_dir / "dummy_Sheet3"
    source_dir.mkdir()
    (source_dir / "metadata.json").write_text(json.dumps({
        "workbook_path": "/path/to/dummy.xlsx",
        "sheet_name": "Sheet3",
        "layout_workflow_version": 4,
    }), encoding="utf-8")
    (source_dir / "structure.yaml").write_text("""
table1:
  id: table1
  name: Equipment status
  description: WET and DRY equipment repair counts and share by failure class.
  sheet: Sheet3
  headers:
    - id: failure_class
      label: Failure class
      description: Equipment failure class
      orientation: column
table2:
  id: table2
  name: Maintenance item cost
  description: Maintenance items, amount, and cost ratio.
  sheet: Sheet3
  headers:
    - id: maintenance_item
      label: Maintenance item
      description: Maintenance item name
      orientation: column
    - id: amount
      label: Amount
      description: Maintenance amount
      orientation: column
""", encoding="utf-8")
    (source_dir / "sheet_text.txt").write_text(
        "sheet has equipment and maintenance summaries",
        encoding="utf-8",
    )
    (source_dir / "table.png").touch()
    (source_dir / "table.html").touch()

    config = TableAgentConfig.from_config({
        "artifact_dir": str(temp_sources_dir.parent),
        "source_artifact_dir": str(temp_sources_dir.parent),
        "retrieval_rerank_with_llm": False,
        "retrieval_top_k": 3,
        "retrieval_candidate_max_chars": 1000,
    })
    retriever = SourceRetriever(config, FakeLLM(), None, None)
    sample = EvalSample(
        index=0,
        sample_id="siflex-test",
        table_id="test",
        table_content="",
        question="maintenance item amount cost ratio",
        answer=[],
        sample_path="siflex",
        table_path="/path/to/dummy.xlsx",
        raw={},
    )

    candidates = retriever.load_candidates(sample)
    sheet_candidates = [candidate for candidate in candidates if candidate.sheet_name == "Sheet3"]

    assert [candidate.table_id for candidate in sheet_candidates] == ["table2", "table1"]
    assert sheet_candidates[0].table_name == "Maintenance item cost"
    assert "table2:" in sheet_candidates[0].structure_text
    assert "table1:" not in sheet_candidates[0].structure_text
    assert "Maintenance item" in sheet_candidates[0].retrieval_card

@pytest.mark.xfail(reason="Entity-score fields are not present in the extracted source snapshot")
def test_candidate_prompt_text_labels(temp_sources_dir):
    from TableAgent.pipeline.prompting import PromptBuilder
    from TableAgent.pipeline.common import SourceCandidate
    
    config = TableAgentConfig.from_config({
        "retrieval_candidate_max_chars": 1000,
    })
    prompt_builder = PromptBuilder(config, None)
    
    candidate = SourceCandidate(
        directory=Path("/dummy"),
        workbook_path=Path("/dummy/book.xlsx"),
        sheet_name="SheetX",
        image_path=Path("/dummy/img.png"),
        html_path=Path("/dummy/html.html"),
        structure_text="some structure",
        sheet_text="some text",
        score=0.85,
        lexical_score=0.7,
        embedding_score=0.9,
        embedding_used=True,
        retrieval_card="Table: compact card\nHeaders: A; B",
    )
    
    def dummy_fit_context(text):
        return text
        
    prompt_text = prompt_builder.candidate_prompt_text([candidate], dummy_fit_context)
    assert "score: 0.85" in prompt_text
    assert "lexical_score: 0.7" in prompt_text
    assert "embedding_score: 0.9" in prompt_text
    assert "embedding_used: True" in prompt_text
    assert "entity_score:" in prompt_text
    assert "matched_terms:" in prompt_text
    assert "missing_terms:" in prompt_text
    assert "table_id:" in prompt_text
    assert "table_name:" in prompt_text
    assert "retrieval_card:" in prompt_text
    assert "Table: compact card" in prompt_text
    assert "some structure" not in prompt_text


@pytest.mark.xfail(reason="Entity-score fields are not present in the extracted source snapshot")
def test_retrieval_entity_match_promotes_specific_value_candidate(temp_sources_dir):
    summary_dir = temp_sources_dir / "dummy_Summary"
    summary_dir.mkdir()
    (summary_dir / "metadata.json").write_text(json.dumps({
        "workbook_path": "/path/to/dummy.xlsx",
        "sheet_name": "Summary",
        "layout_workflow_version": 4,
    }), encoding="utf-8")
    (summary_dir / "structure.yaml").write_text("""
table1:
  id: summary
  name: Maintenance amount summary
  description: Summary of maintenance amount by broad class.
  headers:
    - id: amount
      label: Amount
      orientation: column
""", encoding="utf-8")
    (summary_dir / "sheet_text.txt").write_text(
        "maintenance amount summary broad class",
        encoding="utf-8",
    )
    (summary_dir / "table.png").touch()

    detail_dir = temp_sources_dir / "dummy_Detail"
    detail_dir.mkdir()
    (detail_dir / "metadata.json").write_text(json.dumps({
        "workbook_path": "/path/to/dummy.xlsx",
        "sheet_name": "Detail",
        "layout_workflow_version": 4,
    }), encoding="utf-8")
    (detail_dir / "structure.yaml").write_text("""
table1:
  id: detail
  name: Maintenance detail
  description: Detailed maintenance rows.
  headers:
    - id: amount
      label: Amount
      orientation: column
""", encoding="utf-8")
    (detail_dir / "sheet_text.txt").write_text(
        "maintenance amount 화학동#1 No.1~10 CF54-08 detailed rows",
        encoding="utf-8",
    )
    (detail_dir / "table.png").touch()

    config = TableAgentConfig.from_config({
        "artifact_dir": str(temp_sources_dir.parent),
        "source_artifact_dir": str(temp_sources_dir.parent),
        "retrieval_rerank_with_llm": False,
        "retrieval_top_k": 3,
        "retrieval_candidate_max_chars": 1000,
        "retrieval_entity_weight": 2.0,
        "retrieval_audit_top_k": 5,
    })
    retriever = SourceRetriever(config, FakeLLM(), None, None)
    sample = EvalSample(
        index=0,
        sample_id="siflex-test",
        table_id="test",
        table_content="",
        question="maintenance amount 화학동#1 No.1~10 CF54-08",
        answer=[],
        sample_path="siflex",
        table_path="/path/to/dummy.xlsx",
        raw={},
    )

    candidates = retriever.load_candidates(sample)

    assert candidates[0].sheet_name == "Detail"
    assert candidates[0].entity_score > candidates[1].entity_score
    assert "화학동#1" in candidates[0].matched_terms
    assert candidates[0].retrieval_rank == 1
    assert candidates[0].retrieval_audit[0]["sheet"] == "Detail"
    assert candidates[0].retrieval_audit[0]["matched_terms"]
