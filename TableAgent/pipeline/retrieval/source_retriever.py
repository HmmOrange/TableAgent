from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from TableAgent.llm import BaseLLM, LLMResponse
from TableAgent.run_logging import Logger
from TableAgent.schema import EvalSample

from TableAgent.artifacts import iter_sheet_artifact_dirs
from TableAgent.configs import TableAgentConfig
from TableAgent.structure.layout.parsing import _is_valid_structure
from TableAgent.pipeline.common import SourceCandidate, is_siflex
from TableAgent.utils.table_text import _lexical_overlap_score

from .cards import build_source_retrieval_card, build_table_retrieval_cards
from .embeddings import MockEmbeddingModel
from .reranking import choose_from_reranker
from .scoring import cosine_similarity, hybrid_score, normalize_scores


logger = Logger(__name__)


class SourceRetriever:
    """Compatibility-preserving workbook/sheet retriever used by prepared SiFlex sources."""

    def __init__(
        self,
        settings: TableAgentConfig,
        llm: BaseLLM,
        templates: object,
        prompt_builder: object,
        embedding_client: Any = None,
    ):
        self.settings = settings
        self.llm = llm
        self.templates = templates
        self.prompt_builder = prompt_builder
        self.embedding_client = embedding_client

        if self.embedding_client is None:
            provider = getattr(self.settings, "retrieval_embedding_provider", None)
            if provider == "mock":
                self.embedding_client = MockEmbeddingModel()
            elif provider in {"default", "live", "openai_embedding"}:
                logger.warning("Live retrieval embeddings require an injected embedding_client")

    def select(
        self,
        sample: EvalSample,
        responses: list[LLMResponse],
        fit_context,
    ) -> SourceCandidate | None:
        if not is_siflex(sample) or not sample.table_path:
            return None
        candidates = self.load_candidates(sample)
        if not candidates:
            return None
        if not self.settings.retrieval_rerank_with_llm or len(candidates) == 1:
            self._progress("retrieval", sample=sample.sample_id, candidate=candidates[0])
            return candidates[0]

        top_candidates = candidates[: max(1, self.settings.retrieval_top_k)]
        self._progress("rerank", sample=sample.sample_id, candidate=top_candidates[0])
        prompt = self.templates.reranker_user_prompt_template.format(
            question=sample.question,
            candidates_text=self.prompt_builder.candidate_prompt_text(top_candidates, fit_context),
        )
        response = self.llm.generate(prompt=prompt, system_prompt=self.templates.reranker_system_prompt)
        responses.append(response)
        candidate = self._choose_from_reranker(response.content, top_candidates)
        self._progress("retrieval", sample=sample.sample_id, candidate=candidate)
        return candidate

    def _progress(self, stage: str, *, sample: str, candidate: SourceCandidate) -> None:
        progress = getattr(self.templates, "_progress", None)
        if callable(progress):
            progress(
                stage,
                sample=sample,
                workbook=candidate.workbook_path.name,
                sheet=candidate.sheet_name,
            )

    def load_candidates(self, sample: EvalSample) -> list[SourceCandidate]:
        artifact_dir = self.settings.source_artifact_dir or self.settings.artifact_dir
        source_dirs = artifact_dir / "sources"
        if not source_dirs.is_dir():
            return []
        allowed_paths = {
            str(Path(value.strip()).resolve())
            for value in str(sample.table_path).split(";")
            if value.strip()
        }
        selected_values = sample.raw.get("selected_sheets", []) if isinstance(sample.raw, dict) else []
        selected_sheets = {str(value) for value in selected_values if str(value)}
        candidates = []
        for source_dir in iter_sheet_artifact_dirs(source_dirs):
            candidates.extend(
                self._candidates_from_dir(
                    source_dir,
                    allowed_paths,
                    sample.question,
                    selected_sheets=selected_sheets,
                )
            )
        candidates = self._deduplicate_candidates(candidates)
        if not candidates:
            return []

        embedding_used = False
        if self.embedding_client is not None:
            try:
                vectors = self._encode([sample.question] + [candidate.retrieval_card for candidate in candidates])
                query_vector = vectors[0]
                candidates = [
                    replace(
                        candidate,
                        embedding_score=cosine_similarity(query_vector, vectors[index + 1]),
                        embedding_used=True,
                    )
                    for index, candidate in enumerate(candidates)
                ]
                embedding_used = True
            except Exception as exc:
                logger.warning("Embedding generation failed: %s", exc)

        normalized_lexical = normalize_scores([candidate.lexical_score for candidate in candidates])
        lexical_weight = getattr(self.settings, "retrieval_lexical_weight", 0.5)
        embedding_weight = getattr(self.settings, "retrieval_embedding_weight", 0.5)
        scored = []
        for candidate, lexical_score in zip(candidates, normalized_lexical):
            score = (
                hybrid_score(
                    lexical_score,
                    candidate.embedding_score,
                    lexical_weight=lexical_weight,
                    embedding_weight=embedding_weight,
                )
                if embedding_used
                else candidate.lexical_score
            )
            scored.append(replace(candidate, score=score, embedding_used=embedding_used))
        return sorted(scored, key=lambda candidate: candidate.score, reverse=True)

    @staticmethod
    def _deduplicate_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        unique: dict[tuple[str, str, str], SourceCandidate] = {}
        for candidate in candidates:
            key = (
                str(candidate.workbook_path.resolve()),
                candidate.sheet_name,
                candidate.table_id,
            )
            unique.setdefault(key, candidate)
        return list(unique.values())

    def _encode(self, texts: list[str]):
        async def get_embeddings():
            return await self.embedding_client.encode(texts)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, get_embeddings()).result()
        return asyncio.run(get_embeddings())

    def _candidates_from_dir(
        self,
        source_dir: Path,
        allowed_paths: set[str],
        query: str,
        *,
        selected_sheets: set[str] | None = None,
    ) -> list[SourceCandidate]:
        if not source_dir.is_dir():
            return []
        metadata_path = source_dir / "metadata.json"
        structure_path = source_dir / "structure.yaml"
        sheet_text_path = source_dir / "sheet_text.txt"
        image_path = source_dir / "table.png"
        html_path = source_dir / "table.html"
        if not (
            metadata_path.is_file()
            and structure_path.is_file()
            and sheet_text_path.is_file()
            and image_path.is_file()
        ):
            return []

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        workbook_path = Path(metadata.get("workbook_path", ""))
        if allowed_paths and str(workbook_path.resolve()) not in allowed_paths:
            return []
        structure_text = structure_path.read_text(encoding="utf-8")
        if not _is_valid_structure(structure_text):
            return []
        sheet_text = sheet_text_path.read_text(encoding="utf-8")
        sheet_name = str(metadata.get("sheet_name", ""))
        if selected_sheets and sheet_name not in selected_sheets:
            return []
        table_cards = build_table_retrieval_cards(
            workbook_path,
            sheet_name,
            structure_text,
            sheet_text,
        )
        if table_cards:
            return [
                self._source_candidate(
                    source_dir=source_dir,
                    workbook_path=workbook_path,
                    sheet_name=sheet_name,
                    image_path=image_path,
                    html_path=html_path if html_path.is_file() else None,
                    structure_text=table_card["structure_text"],
                    sheet_text=sheet_text,
                    retrieval_card=table_card["retrieval_card"],
                    query=query,
                    table_id=table_card["table_id"],
                    table_name=table_card["table_name"],
                    table_description=table_card["description"],
                )
                for table_card in table_cards
            ]
        retrieval_card = build_source_retrieval_card(
            workbook_path,
            sheet_name,
            structure_text,
            sheet_text,
        )
        return [
            self._source_candidate(
                source_dir=source_dir,
                workbook_path=workbook_path,
                sheet_name=sheet_name,
                image_path=image_path,
                html_path=html_path if html_path.is_file() else None,
                structure_text=structure_text,
                sheet_text=sheet_text,
                retrieval_card=retrieval_card,
                query=query,
            )
        ]

    def _source_candidate(
        self,
        *,
        source_dir: Path,
        workbook_path: Path,
        sheet_name: str,
        image_path: Path,
        html_path: Path | None,
        structure_text: str,
        sheet_text: str,
        retrieval_card: str,
        query: str,
        table_id: str = "",
        table_name: str = "",
        table_description: str = "",
    ) -> SourceCandidate:
        lexical_score = _lexical_overlap_score(query, retrieval_card)
        return SourceCandidate(
            directory=source_dir,
            workbook_path=workbook_path,
            sheet_name=sheet_name,
            image_path=image_path,
            html_path=html_path,
            structure_text=structure_text,
            sheet_text=sheet_text,
            score=lexical_score,
            lexical_score=lexical_score,
            embedding_score=0.0,
            embedding_used=False,
            retrieval_card=retrieval_card,
            table_id=table_id,
            table_name=table_name,
            table_description=table_description,
        )

    def _candidate_from_dir(
        self,
        source_dir: Path,
        allowed_paths: set[str],
        query: str,
    ) -> SourceCandidate | None:
        candidates = self._candidates_from_dir(source_dir, allowed_paths, query)
        return max(candidates, key=lambda candidate: candidate.lexical_score) if candidates else None

    _choose_from_reranker = staticmethod(choose_from_reranker)
