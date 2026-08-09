"""Retrieval stage: prepared-corpus loading, ranking, and selection."""

from importlib import import_module

_EXPORTS = {
    "MockEmbeddingModel": (".embeddings", "MockEmbeddingModel"),
    "OpenAICompatibleEmbeddingClient": (".embeddings", "OpenAICompatibleEmbeddingClient"),
    "RetrievalInput": (".contracts", "RetrievalInput"),
    "RetrievalOutput": (".contracts", "RetrievalOutput"),
    "RetrievalPipelineMixin": (".pipeline", "RetrievalPipelineMixin"),
    "RetrievalStage": (".stage", "RetrievalStage"),
    "SourceCandidate": (".contracts", "SourceCandidate"),
    "SourceRetriever": (".source_retriever", "SourceRetriever"),
    "TableCandidate": (".contracts", "TableCandidate"),
    "TableRetriever": (".contracts", "TableRetriever"),
    "TableRetrieverContract": (".contracts", "TableRetrieverContract"),
    "TableSearchRequest": (".contracts", "TableSearchRequest"),
    "build_metadata_retrieval_card": ("TableAgent.stages.structure.card_builders", "build_metadata_retrieval_card"),
    "build_sheet_metadata_payload": ("TableAgent.stages.structure.card_builders", "build_sheet_metadata_payload"),
    "build_source_retrieval_card": ("TableAgent.stages.structure.card_builders", "build_source_retrieval_card"),
    "build_table_retrieval_cards": ("TableAgent.stages.structure.card_builders", "build_table_retrieval_cards"),
    "choose_from_reranker": (".reranking", "choose_from_reranker"),
    "cosine_similarity": (".scoring", "cosine_similarity"),
    "extract_columns": ("TableAgent.stages.structure.card_builders", "extract_columns"),
    "extract_headers_text": ("TableAgent.stages.structure.card_builders", "extract_headers_text"),
    "hybrid_score": (".scoring", "hybrid_score"),
    "normalize_scores": (".scoring", "normalize_scores"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    qualified_name = module_name if module_name.startswith("TableAgent.") else f"{__name__}{module_name}"
    return getattr(import_module(qualified_name), attribute)


__all__ = list(_EXPORTS)
