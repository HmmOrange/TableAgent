from pathlib import Path

from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.pipeline.retrieval import (
    MockEmbeddingModel,
    OpenAICompatibleEmbeddingClient,
    SourceRetriever,
    TableCandidate,
    TableRetriever,
    TableRetrieverContract,
    TableSearchRequest,
)
from TableAgent.pipeline.retrieval.source_retriever import SourceRetriever as SourceRetrieverImplementation


SAMPLE_WORKBOOK = Path("sample/multitab.xlsx")
SAMPLE_STRUCTURE = Path("sample/multitab_structure.yaml")


class RecordingTableRetriever:
    def __init__(self):
        self.requests = []

    def search(self, request: TableSearchRequest) -> list[TableCandidate]:
        self.requests.append(request)
        return [
            TableCandidate(table_id="missing", score=1.0),
            TableCandidate(table_id="table2", score=0.9, reason="salary table"),
            TableCandidate(table_id="table1", score=0.8),
        ]


def test_retrieval_package_preserves_existing_public_imports():
    assert SourceRetriever is SourceRetrieverImplementation
    assert MockEmbeddingModel is not None
    assert OpenAICompatibleEmbeddingClient is not None


def test_table_retriever_contract_is_runtime_checkable():
    assert isinstance(RecordingTableRetriever(), TableRetrieverContract)


def test_find_tables_delegates_to_injected_retriever_and_validates_ids():
    retriever = RecordingTableRetriever()
    env = QAEnvironment(
        str(SAMPLE_STRUCTURE),
        str(SAMPLE_WORKBOOK),
        table_retriever=retriever,
    )
    try:
        candidates = env.operators.retrieve_tables("salary and KPI", top_k=2)
        table_ids = env.operators.find_tables("salary and KPI", top_k=2)
    finally:
        env.workbook.close()

    assert [candidate.table_id for candidate in candidates] == ["table2", "table1"]
    assert table_ids == ["table2", "table1"]
    assert len(retriever.requests) == 2
    request = retriever.requests[0]
    assert request.query == "salary and KPI"
    assert request.top_k == 2
    assert request.allowed_table_ids == ("table1", "table2")
    assert request.workbook_paths == (SAMPLE_WORKBOOK.resolve(),)
    assert request.sheet_names == ("Data_Synthesis",)


def test_placeholder_table_retriever_falls_back_to_lexical_routing():
    env = QAEnvironment(
        str(SAMPLE_STRUCTURE),
        str(SAMPLE_WORKBOOK),
        table_retriever=TableRetriever(),
    )
    try:
        selected = env.operators.find_tables("Lương cơ bản và Hệ số KPI")
    finally:
        env.workbook.close()

    assert selected == ["table2"]
