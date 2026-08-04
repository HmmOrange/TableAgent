from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from TableAgent.configs import TableAgentConfig
from TableAgent.llm import BaseLLM, LLMResponse
from TableAgent.pipeline.common import SourceCandidate, has_workbook_sources
from TableAgent.run_logging import Logger
from TableAgent.schema import EvalSample

from .candidate_loading import RetrievalCandidateLoadingMixin
from .indexed_guards import (
    apply_indexed_guards,
    attach_guard_trace,
    expand_metadata_selection,
)
from .perfect import PerfectRetrievalMixin
from .ranking import RetrievalRankingMixin
from .embeddings import MockEmbeddingModel


logger = Logger(__name__)


class SourceRetriever(
    PerfectRetrievalMixin,
    RetrievalRankingMixin,
    RetrievalCandidateLoadingMixin,
):
    """Workbook/sheet/table retriever over prepared TableAgent source artifacts."""

    def __init__(
        self,
        settings: TableAgentConfig,
        llm: BaseLLM | None,
        templates: object,
        prompt_builder: object,
        embedding_client: Any = None,
    ):
        self.settings = settings
        self.llm = llm
        self.templates = templates
        self.prompt_builder = prompt_builder
        self.embedding_client = embedding_client
        self._perfect_mapping: dict[str, dict[str, str]] | None = None

        if self.embedding_client is None:
            provider = self.settings.retrieval_embedding_provider
            if provider == "mock":
                self.embedding_client = MockEmbeddingModel()
            elif provider in {"default", "live", "openai_embedding"}:
                logger.warning(
                    "Live retrieval embeddings require an injected embedding_client"
                )

    def select(
        self,
        sample: EvalSample,
        responses: list[LLMResponse],
        fit_context,
    ) -> SourceCandidate | None:
        if not has_workbook_sources(sample):
            return None
        data_candidates, metadata_candidates = self._candidate_pools(sample)
        query_type = self._resolve_query_type(sample.question, responses)
        candidates = self._rank_candidates(
            self._candidates_for_query_type(
                query_type, data_candidates, metadata_candidates
            ),
            sample.question,
        )
        if not candidates:
            return None
        if (
            not self.settings.retrieval_rerank_with_llm
            or len(candidates) == 1
            or self.llm is None
        ):
            self._progress("retrieval", sample=sample.sample_id, candidate=candidates[0])
            return candidates[0]

        candidate = self._select_from_batches(
            sample.question,
            candidates,
            responses,
            fit_context,
            query_type=query_type,
        )
        self._progress("retrieval", sample=sample.sample_id, candidate=candidate)
        return candidate

    def select_indexed(
        self,
        *,
        question: str,
        artifacts: list[dict[str, Any]],
        workbook_paths: dict[str, Path],
        responses: list[LLMResponse],
        fit_context,
    ) -> SourceCandidate | None:
        """Select an ingestion-time artifact with the standard hybrid retriever."""
        data_candidates: list[SourceCandidate] = []
        metadata_candidates: list[SourceCandidate] = []
        for artifact in artifacts:
            workbook_name = str(
                artifact.get("upload_name")
                or artifact.get("document_name")
                or artifact.get("workbook")
                or ""
            ).strip()
            workbook_path = workbook_paths.get(workbook_name)
            sheet_name = str(
                artifact.get("sheet") or artifact.get("sheet_name") or ""
            ).strip()
            structure_text = str(artifact.get("structure_yaml") or "").strip()
            retrieval_card = str(artifact.get("retrieval_card") or "").strip()
            if not retrieval_card and isinstance(artifact.get("metadata"), dict):
                retrieval_card = json.dumps(
                    artifact["metadata"], ensure_ascii=False, default=str
                )
            if workbook_path is None or not sheet_name or not retrieval_card:
                continue
            retrieval_type = str(artifact.get("retrieval_type") or "data").strip()
            embedding_vector, embedding_model = self._artifact_embedding(artifact)
            artifact_metadata = artifact.get("metadata")
            if not isinstance(artifact_metadata, dict):
                artifact_metadata = {}
            try:
                indexed_score = float(artifact["score"])
            except (KeyError, TypeError, ValueError):
                indexed_score = None
            candidate = self._source_candidate(
                source_dir=workbook_path.parent,
                workbook_path=workbook_path,
                sheet_name=sheet_name,
                image_path=workbook_path,
                html_path=None,
                structure_text=structure_text,
                sheet_text=retrieval_card,
                retrieval_card=retrieval_card,
                query=question,
                table_id=str(artifact.get("table_id") or ""),
                table_name=str(artifact.get("table_name") or ""),
                table_description=str(
                    artifact.get("table_description")
                    or artifact_metadata.get("description")
                    or ""
                ),
                retrieval_type=retrieval_type,
                retrieval_level=str(artifact.get("retrieval_level") or "table"),
                artifact_id=str(artifact.get("id") or ""),
                embedding_vector=embedding_vector,
                embedding_model=embedding_model,
                embedding_score=indexed_score or 0.0,
                embedding_used=indexed_score is not None,
            )
            if retrieval_type == "metadata":
                metadata_candidates.append(candidate)
            else:
                data_candidates.append(candidate)

        data_candidates, metadata_candidates, guard_trace = apply_indexed_guards(
            question,
            data_candidates,
            metadata_candidates,
            workbook_guard_enabled=self.settings.routing.retrieval.explicit_workbook_guard,
            sheet_guard_enabled=self.settings.routing.retrieval.explicit_sheet_guard,
        )
        query_type = self._resolve_query_type(question, responses)
        candidates = self._rank_candidates(
            self._candidates_for_query_type(
                query_type, data_candidates, metadata_candidates
            ),
            question,
        )
        if not candidates and query_type != "data":
            candidates = self._rank_candidates(data_candidates, question)
        if not candidates:
            return None
        if (
            not self.settings.retrieval_rerank_with_llm
            or len(candidates) == 1
            or self.llm is None
        ):
            selected = candidates[0]
        else:
            selected = self._select_from_batches(
                question,
                candidates,
                responses,
                fit_context,
                query_type=query_type,
            )
        if not guard_trace.get("explicit_sheet_guard", {}).get("applied"):
            selected = expand_metadata_selection(selected, candidates)
        return attach_guard_trace(selected, guard_trace)
