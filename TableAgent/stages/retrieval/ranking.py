from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from TableAgent.llm import LLMResponse
from TableAgent.pipeline.common import SourceCandidate
from TableAgent.run_logging import Logger
from TableAgent.structure.layout.parsing import _parse_yaml_mapping

from .embeddings import MockEmbeddingModel
from .scoring import bm25_scores, cosine_similarity, hybrid_score, normalize_scores


logger = Logger(__name__)


class RetrievalRankingMixin:
    """Rank candidates and optionally ask an LLM to rerank bounded batches."""

    def _rank_candidates(
        self, candidates: list[SourceCandidate], query: str
    ) -> list[SourceCandidate]:
        if not candidates:
            return []

        embedding_client = (
            self.embedding_client
            if self.settings.routing.retrieval.use_embeddings
            else None
        )
        indexed_scores_available = all(
            candidate.embedding_used for candidate in candidates
        )
        stored_models = {
            candidate.embedding_model
            for candidate in candidates
            if candidate.embedding_vector and candidate.embedding_model
        }
        if embedding_client is None and stored_models == {"mock-hash-embedding"}:
            embedding_client = MockEmbeddingModel()

        if embedding_client is not None and not indexed_scores_available:
            candidates = self._with_embedding_scores(
                candidates, query, embedding_client
            )

        candidate_texts = [
            " ".join(
                part
                for part in (
                    candidate.retrieval_card,
                    candidate.table_name,
                    candidate.table_description,
                )
                if part
            )
            for candidate in candidates
        ]
        raw_bm25 = bm25_scores(query, candidate_texts)
        normalized_bm25 = normalize_scores(raw_bm25)
        normalized_lexical = normalize_scores(
            [candidate.lexical_score for candidate in candidates]
        )
        bm25_available = any(score > 0 for score in raw_bm25)
        scored = []
        for candidate, raw_bm25_score, bm25_score, lexical_score in zip(
            candidates,
            raw_bm25,
            normalized_bm25,
            normalized_lexical,
        ):
            lexical_evidence = bm25_score if bm25_available else lexical_score
            score = (
                hybrid_score(
                    lexical_evidence,
                    candidate.embedding_score,
                    lexical_weight=self.settings.retrieval_bm25_weight,
                    embedding_weight=self.settings.retrieval_embedding_weight,
                )
                if candidate.embedding_used
                else lexical_evidence
            )
            score += candidate.workbook_reference_score
            scored.append(
                replace(candidate, score=score, bm25_score=raw_bm25_score)
            )

        ranked = sorted(scored, key=lambda candidate: candidate.score, reverse=True)
        audit_top_k = max(1, int(self.settings.retrieval_audit_top_k))
        audit = tuple(
            self._audit_row(candidate, rank)
            for rank, candidate in enumerate(ranked[:audit_top_k], start=1)
        )
        return [
            replace(
                candidate,
                retrieval_rank=rank,
                retrieval_audit=audit,
                retrieval_trace=(
                    {
                        "query_type": candidate.retrieval_type,
                        "top_candidates": [
                            self._audit_row(item, item_rank)
                            for item_rank, item in enumerate(
                                ranked[:audit_top_k], start=1
                            )
                        ],
                    },
                ),
            )
            for rank, candidate in enumerate(ranked, start=1)
        ]

    def _with_embedding_scores(
        self,
        candidates: list[SourceCandidate],
        query: str,
        embedding_client: Any,
    ) -> list[SourceCandidate]:
        try:
            client_model = self._embedding_client_model(embedding_client)
            generated_indices = [
                index
                for index, candidate in enumerate(candidates)
                if not candidate.embedding_vector
                or (
                    candidate.embedding_model
                    and client_model
                    and candidate.embedding_model != client_model
                )
            ]
            vectors = self._encode_with_client(
                embedding_client,
                [query]
                + [candidates[index].retrieval_card for index in generated_indices],
            )
            query_vector = vectors[0]
            generated_vectors = {
                candidate_index: vectors[vector_index + 1]
                for vector_index, candidate_index in enumerate(generated_indices)
            }
            scored = []
            for index, candidate in enumerate(candidates):
                stored_vector = candidate.embedding_vector
                stored_compatible = bool(stored_vector) and not (
                    candidate.embedding_model
                    and client_model
                    and candidate.embedding_model != client_model
                )
                if stored_compatible and len(stored_vector) == len(query_vector):
                    candidate_vector = stored_vector
                    embedding_source = "stored"
                else:
                    candidate_vector = generated_vectors.get(index)
                    embedding_source = (
                        "generated" if candidate_vector is not None else ""
                    )
                if candidate_vector is None:
                    scored.append(candidate)
                    continue
                scored.append(
                    replace(
                        candidate,
                        embedding_score=cosine_similarity(
                            query_vector, candidate_vector
                        ),
                        embedding_used=True,
                        embedding_source=embedding_source,
                    )
                )
            return scored
        except Exception as exc:
            logger.warning("Embedding generation failed: %s", exc)
            return candidates

    def _resolve_query_type(
        self, question: str, responses: list[LLMResponse]
    ) -> str:
        configured = self.settings.retrieval_query_type
        if configured in {"data", "metadata", "both"}:
            return configured
        explicit_metadata = self._is_explicit_metadata_query(question)
        if not self.settings.retrieval_rerank_with_llm or self.llm is None:
            return self._heuristic_query_type(question)
        prompt = (
            "Classify the user's spreadsheet question into one retrieval type.\n"
            "- data: asks for cell values, rows, calculations, counts, lists from table contents.\n"
            "- metadata: asks about workbook/sheet/table names, descriptions, available sheets, structure, roles, or where information is located.\n"
            "- both: needs both table data and workbook/sheet/table metadata.\n\n"
            f"Question: {question}\n\n"
            "Output ONLY YAML:\n"
            "```yaml\n"
            "retrieval_type: <data|metadata|both>\n"
            "rationale: <brief reason>\n"
            "```"
        )
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You route spreadsheet questions to retrieval indexes.",
        )
        responses.append(response)
        parsed = _parse_yaml_mapping(response.content)
        query_type = str(parsed.get("retrieval_type") or "").strip().lower()
        if query_type not in {"data", "metadata", "both"}:
            return self._heuristic_query_type(question)
        if query_type in {"metadata", "both"} and not explicit_metadata:
            return "data"
        return query_type

    def _heuristic_query_type(self, question: str) -> str:
        return "metadata" if self._is_explicit_metadata_query(question) else "data"

    @staticmethod
    def _is_explicit_metadata_query(question: str) -> bool:
        lowered = str(question).lower()
        metadata_terms = (
            "workbook",
            "file",
            "metadata",
            "structure",
            "cấu trúc",
            "vai trò",
            "ghi chú",
            "description",
            "table nào",
            "sheet nào",
            "tên sheet",
            "tên bảng",
            "danh sách sheet",
            "có những sheet",
            "chứa sheet",
            "sheet list",
            "available sheets",
            "sheet names",
            "table names",
        )
        return any(term in lowered for term in metadata_terms)

    @staticmethod
    def _candidates_for_query_type(
        query_type: str,
        data_candidates: list[SourceCandidate],
        metadata_candidates: list[SourceCandidate],
    ) -> list[SourceCandidate]:
        if query_type == "metadata":
            return metadata_candidates
        if query_type == "both":
            return [*data_candidates, *metadata_candidates]
        return data_candidates

    def _select_from_batches(
        self,
        question: str,
        candidates: list[SourceCandidate],
        responses: list[LLMResponse],
        fit_context,
        *,
        query_type: str,
    ) -> SourceCandidate:
        top_k = max(1, int(self.settings.retrieval_top_k))
        max_batches = max(1, int(self.settings.retrieval_max_batches))
        visible: list[SourceCandidate] = []
        selected: SourceCandidate | None = None
        selected_index: int | None = None
        rationale = ""
        status = "enough"
        for batch_index in range(max_batches):
            start = batch_index * top_k
            if start >= len(candidates):
                break
            visible = candidates[: start + top_k]
            prompt = self._selection_prompt(
                question, visible, fit_context, query_type=query_type
            )
            response = self.llm.generate(
                prompt=prompt, system_prompt=self._selection_system_prompt()
            )
            responses.append(response)
            parsed = _parse_yaml_mapping(response.content)
            try:
                parsed_index = int(parsed.get("selected_index"))
            except (TypeError, ValueError):
                parsed_index = -1
            if 0 <= parsed_index < len(visible):
                selected_index = parsed_index
                selected = visible[parsed_index]
            else:
                selected_index = None
                selected = visible[0]
            rationale = str(parsed.get("rationale") or "")
            status = str(
                parsed.get("status") or parsed.get("sufficiency") or "enough"
            ).strip().lower()
            if (
                status not in {"need_more", "insufficient"}
                or len(visible) >= len(candidates)
            ):
                break
        if selected is None:
            selected = candidates[0]
        trace = (
            {
                "query_type": query_type,
                "status": status,
                "visible_count": len(visible),
                "selected_index": selected_index,
                "rationale": rationale,
                "top_candidates": [
                    self._audit_row(item, rank)
                    for rank, item in enumerate(visible, start=1)
                ],
            },
        )
        selected = replace(selected, retrieval_trace=trace)
        object.__setattr__(selected, "reranker_selected_index", selected_index)
        object.__setattr__(selected, "reranker_rationale", rationale)
        object.__setattr__(selected, "fallback_used", selected_index is None)
        return selected

    def _selection_prompt(
        self,
        question: str,
        candidates: list[SourceCandidate],
        fit_context,
        *,
        query_type: str,
    ) -> str:
        return (
            f"Question: {question}\n\n"
            f"Retrieval query type: {query_type}\n"
            "You are given ranked candidates from lexical keyword matching and optional embedding cosine similarity. "
            "Select the candidate that contains the information needed for QA. "
            "If the visible candidates are not enough, set status: need_more.\n\n"
            f"Candidates:\n{self.prompt_builder.candidate_prompt_text(candidates, fit_context)}\n\n"
            "Output ONLY YAML:\n"
            "```yaml\n"
            "selected_index: <0-based index among visible candidates>\n"
            "status: <enough|need_more>\n"
            "rationale: <brief reason>\n"
            "```"
        )

    @staticmethod
    def _selection_system_prompt() -> str:
        return (
            "You are a spreadsheet retrieval selection agent. Choose only from the "
            "visible candidates. Return valid YAML and do not answer the user question."
        )

    def _encode(self, texts: list[str]):
        return self._encode_with_client(self.embedding_client, texts)

    @staticmethod
    def _encode_with_client(embedding_client: Any, texts: list[str]):
        async def get_embeddings():
            encoder = getattr(embedding_client, "encode", None)
            if callable(encoder):
                return await encoder(texts)
            batch_encoder = getattr(embedding_client, "batch_encode", None)
            if callable(batch_encoder):
                return await batch_encoder(texts)
            raise TypeError(
                "embedding_client must implement async encode() or batch_encode()"
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, get_embeddings()).result()
        return asyncio.run(get_embeddings())

    @staticmethod
    def _embedding_client_model(embedding_client: Any) -> str:
        if isinstance(embedding_client, MockEmbeddingModel):
            return "mock-hash-embedding"
        return str(getattr(embedding_client, "model", "") or "")

    @staticmethod
    def _artifact_embedding(
        artifact: dict[str, Any],
    ) -> tuple[tuple[float, ...], str]:
        embedding = artifact.get("embedding")
        model = ""
        values: Any = None
        if isinstance(embedding, dict):
            model = str(embedding.get("model") or "")
            values = embedding.get("values")
        elif isinstance(embedding, (list, tuple)):
            values = embedding
        if not isinstance(values, (list, tuple)) or not values:
            return (), model
        try:
            return tuple(float(value) for value in values), model
        except (TypeError, ValueError):
            return (), model
