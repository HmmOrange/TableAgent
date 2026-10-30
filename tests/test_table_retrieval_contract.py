from TableAgent.stages.retrieval import (
    MockEmbeddingModel,
    OpenAICompatibleEmbeddingClient,
    SourceRetriever,
    TableCandidate,
    TableRetrieverContract,
    TableSearchRequest,
)
from TableAgent.stages.retrieval.source_retriever import SourceRetriever as SourceRetrieverImplementation


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
