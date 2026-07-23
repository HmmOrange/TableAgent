from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from TableAgent.artifacts import iter_sheet_artifact_dirs
from TableAgent.configs import TableAgentConfig
from TableAgent.llm import BaseLLM, LLMResponse
from TableAgent.pipeline.common import SourceCandidate, is_siflex
from TableAgent.run_logging import Logger
from TableAgent.schema import EvalSample
from TableAgent.structure.layout.parsing import _is_valid_structure, _parse_yaml_mapping
from TableAgent.utils.table_text import _lexical_overlap_score

from .cards import (
    build_metadata_retrieval_card,
    build_sheet_metadata_payload,
    build_source_retrieval_card,
    build_table_retrieval_cards,
)
from .embeddings import MockEmbeddingModel
from .reranking import choose_from_reranker
from .scoring import cosine_similarity, hybrid_score, normalize_scores


logger = Logger(__name__)

_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "of", "on", "or",
    "sheet", "table", "the", "to", "what", "which", "with", "bảng", "bao", "các", "câu",
    "cho", "của", "gì", "hãy", "hỏi", "không", "là", "nào", "nêu", "những", "nhiêu",
    "trong", "và", "với",
}


class SourceRetriever:
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
        data_candidates, metadata_candidates = self._candidate_pools(sample)
        query_type = self._resolve_query_type(sample.question, responses)
        candidates = self._rank_candidates(
            self._candidates_for_query_type(query_type, data_candidates, metadata_candidates),
            sample.question,
        )
        if not candidates:
            return None
        if not self.settings.retrieval_rerank_with_llm or len(candidates) == 1 or self.llm is None:
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

    def _progress(self, stage: str, *, sample: str, candidate: SourceCandidate) -> None:
        progress = getattr(self.templates, "_progress", None)
        if callable(progress):
            progress(stage, sample=sample, workbook=candidate.workbook_path.name, sheet=candidate.sheet_name)

    def load_candidates(self, sample: EvalSample) -> list[SourceCandidate]:
        data_candidates, _metadata_candidates = self._candidate_pools(sample)
        return self._rank_candidates(data_candidates, sample.question)

    def _candidate_pools(self, sample: EvalSample) -> tuple[list[SourceCandidate], list[SourceCandidate]]:
        artifact_dir = self.settings.source_artifact_dir or self.settings.artifact_dir
        source_dirs = artifact_dir / "sources"
        if not source_dirs.is_dir():
            return [], []
        allowed_paths = {
            str(Path(value.strip()).resolve())
            for value in str(sample.table_path).split(";")
            if value.strip()
        }
        selected_values = sample.raw.get("selected_sheets", []) if isinstance(sample.raw, dict) else []
        selected_sheets = {str(value) for value in selected_values if str(value)}
        data_candidates: list[SourceCandidate] = []
        sheet_metadata_candidates: list[SourceCandidate] = []
        for source_dir in iter_sheet_artifact_dirs(source_dirs):
            data, metadata = self._candidates_from_dir(
                source_dir,
                allowed_paths,
                sample.question,
                selected_sheets=selected_sheets,
            )
            data_candidates.extend(data)
            sheet_metadata_candidates.extend(metadata)
        data_candidates = self._deduplicate_candidates(data_candidates)
        sheet_metadata_candidates = self._deduplicate_candidates(sheet_metadata_candidates)
        metadata_candidates = [
            *self._workbook_metadata_candidates(sheet_metadata_candidates, sample.question),
            *sheet_metadata_candidates,
        ]
        return data_candidates, metadata_candidates

    def _rank_candidates(self, candidates: list[SourceCandidate], query: str) -> list[SourceCandidate]:
        if not candidates:
            return []

        embedding_used = False
        if self.embedding_client is not None:
            try:
                vectors = self._encode([query] + [candidate.retrieval_card for candidate in candidates])
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
            score += float(getattr(self.settings, "retrieval_entity_weight", 2.0)) * candidate.entity_score
            scored.append(replace(candidate, score=score, embedding_used=embedding_used))
        ranked = sorted(scored, key=lambda candidate: candidate.score, reverse=True)
        audit_top_k = max(1, int(getattr(self.settings, "retrieval_audit_top_k", 10)))
        audit = tuple(self._audit_row(candidate, rank) for rank, candidate in enumerate(ranked[:audit_top_k], start=1))
        return [
            replace(
                candidate,
                retrieval_rank=rank,
                retrieval_audit=audit,
                retrieval_trace=(
                    {
                        "query_type": candidate.retrieval_type,
                        "top_candidates": [
                            self._audit_row(item, rank)
                            for rank, item in enumerate(ranked[:audit_top_k], start=1)
                        ],
                    },
                ),
            )
            for rank, candidate in enumerate(ranked, start=1)
        ]

    def _resolve_query_type(self, question: str, responses: list[LLMResponse]) -> str:
        configured = getattr(self.settings, "retrieval_query_type", "auto")
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
        response = self.llm.generate(prompt=prompt, system_prompt="You route spreadsheet questions to retrieval indexes.")
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

    def _is_explicit_metadata_query(self, question: str) -> bool:
        lowered = str(question).lower()
        metadata_terms = (
            "workbook", "file", "metadata", "structure", "cấu trúc", "vai trò", "ghi chú",
            "description", "table nào", "sheet nào", "tên sheet", "tên bảng", "danh sách sheet",
            "có những sheet", "chứa sheet", "sheet list", "available sheets", "sheet names", "table names",
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
        max_batches = max(1, int(getattr(self.settings, "retrieval_max_batches", 3)))
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
            prompt = self._selection_prompt(question, visible, fit_context, query_type=query_type)
            response = self.llm.generate(prompt=prompt, system_prompt=self._selection_system_prompt())
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
            status = str(parsed.get("status") or parsed.get("sufficiency") or "enough").strip().lower()
            if status not in {"need_more", "insufficient"} or len(visible) >= len(candidates):
                break
        if selected is None:
            selected = candidates[0]
        fallback_used = selected_index is None
        trace = (
            {
                "query_type": query_type,
                "status": status,
                "visible_count": len(visible),
                "selected_index": selected_index,
                "rationale": rationale,
                "top_candidates": [self._audit_row(item, rank) for rank, item in enumerate(visible, start=1)],
            },
        )
        selected = replace(selected, retrieval_trace=trace)
        object.__setattr__(selected, "reranker_selected_index", selected_index)
        object.__setattr__(selected, "reranker_rationale", rationale)
        object.__setattr__(selected, "fallback_used", fallback_used)
        return selected

    def _selection_prompt(self, question: str, candidates: list[SourceCandidate], fit_context, *, query_type: str) -> str:
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
            "You are a spreadsheet retrieval selection agent. Choose only from the visible candidates. "
            "Return valid YAML and do not answer the user question."
        )

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
    ) -> tuple[list[SourceCandidate], list[SourceCandidate]]:
        if not source_dir.is_dir():
            return [], []
        metadata_path = source_dir / "metadata.json"
        structure_path = source_dir / "structure.yaml"
        sheet_text_path = source_dir / "sheet_text.txt"
        image_path = source_dir / "table.png"
        html_path = source_dir / "table.html"
        if not (metadata_path.is_file() and sheet_text_path.is_file() and image_path.is_file()):
            return [], []

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        workbook_path = Path(metadata.get("workbook_path", ""))
        if allowed_paths and str(workbook_path.resolve()) not in allowed_paths:
            return [], []
        sheet_text = sheet_text_path.read_text(encoding="utf-8")
        sheet_name = str(metadata.get("sheet_name", ""))
        card_workbook_path = Path(str(metadata.get("workbook_name") or workbook_path.name))
        if selected_sheets and sheet_name not in selected_sheets:
            return [], []
        structure_text = structure_path.read_text(encoding="utf-8") if structure_path.is_file() else ""
        if not _is_valid_structure(structure_text):
            structure_text = self._fallback_structure_text(source_dir, sheet_name)

        sheet_metadata_payload = build_sheet_metadata_payload(
            card_workbook_path,
            sheet_name,
            structure_text,
            sheet_text,
            self._read_sheet_metadata(source_dir),
        )
        sheet_metadata_text = yaml.safe_dump({"metadata": sheet_metadata_payload}, allow_unicode=True, sort_keys=False)
        sheet_metadata_card = build_metadata_retrieval_card(sheet_metadata_payload)
        metadata_candidate = self._source_candidate(
            source_dir=source_dir,
            workbook_path=workbook_path,
            sheet_name=sheet_name,
            image_path=image_path,
            html_path=html_path if html_path.is_file() else None,
            structure_text=sheet_metadata_text,
            sheet_text=sheet_metadata_card,
            retrieval_card=sheet_metadata_card,
            query=query,
            retrieval_type="metadata",
            retrieval_level="sheet",
        )
        table_cards = build_table_retrieval_cards(card_workbook_path, sheet_name, structure_text, sheet_text)
        if table_cards:
            data_candidates = [
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
                    retrieval_type="data",
                    retrieval_level="table",
                )
                for table_card in table_cards
            ]
            return data_candidates, [metadata_candidate]

        retrieval_card = build_source_retrieval_card(card_workbook_path, sheet_name, structure_text, sheet_text)
        data_candidate = self._source_candidate(
            source_dir=source_dir,
            workbook_path=workbook_path,
            sheet_name=sheet_name,
            image_path=image_path,
            html_path=html_path if html_path.is_file() else None,
            structure_text=structure_text,
            sheet_text=sheet_text,
            retrieval_card=retrieval_card,
            query=query,
            retrieval_type="data",
            retrieval_level="sheet",
        )
        return [data_candidate], [metadata_candidate]

    def _read_sheet_metadata(self, source_dir: Path) -> dict:
        metadata_yaml = source_dir / "metadata.yaml"
        if not metadata_yaml.is_file():
            return {}
        try:
            payload = yaml.safe_load(metadata_yaml.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _fallback_structure_text(self, source_dir: Path, sheet_name: str) -> str:
        used_range = None
        metadata_yaml = source_dir / "metadata.yaml"
        if metadata_yaml.is_file():
            try:
                payload = yaml.safe_load(metadata_yaml.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    used_range = payload.get("used_range")
            except Exception:
                used_range = None
        table_metadata = source_dir / "table.metadata.json"
        if used_range is None and table_metadata.is_file():
            try:
                payload = json.loads(table_metadata.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    used_range = payload.get("cell_range")
            except Exception:
                used_range = None
        table_id = re.sub(r"[^0-9A-Za-z_]+", "_", sheet_name.lower()).strip("_") or "table1"
        payload = {
            "table1": {
                "id": table_id,
                "name": sheet_name,
                "description": "Fallback structure generated from prepared sheet metadata because layout structure generation failed.",
                "sheet": sheet_name,
                "table_range": used_range,
                "headers": [],
            }
        }
        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)

    def _workbook_metadata_candidates(self, sheet_candidates: list[SourceCandidate], query: str) -> list[SourceCandidate]:
        grouped: dict[Path, list[SourceCandidate]] = {}
        for candidate in sheet_candidates:
            grouped.setdefault(candidate.workbook_path.resolve(), []).append(candidate)
        workbook_candidates: list[SourceCandidate] = []
        for _resolved_path, sheets in grouped.items():
            if not sheets:
                continue
            base = sheets[0]
            sheet_payloads = []
            for sheet in sheets:
                try:
                    payload = yaml.safe_load(sheet.structure_text)
                except Exception:
                    payload = {}
                metadata = payload.get("metadata") if isinstance(payload, dict) else {}
                sheet_payloads.append({
                    "name": sheet.sheet_name,
                    "description": metadata.get("description", "") if isinstance(metadata, dict) else "",
                    "used_range": metadata.get("used_range", "") if isinstance(metadata, dict) else "",
                    "merged_ranges": metadata.get("merged_ranges", []) if isinstance(metadata, dict) else [],
                    "sheet_summary": metadata.get("sheet_summary", "") if isinstance(metadata, dict) else "",
                    "preview": metadata.get("preview", "") if isinstance(metadata, dict) else "",
                    "tables": metadata.get("tables", []) if isinstance(metadata, dict) else [],
                })
            workbook_payload = {
                "type": "workbook",
                "workbook": base.workbook_path.name,
                "description": self._workbook_description(sheet_payloads),
                "sheets": sheet_payloads,
            }
            workbook_text = yaml.safe_dump({"metadata": workbook_payload}, allow_unicode=True, sort_keys=False)
            workbook_card = build_metadata_retrieval_card(workbook_payload)
            workbook_candidates.append(
                self._source_candidate(
                    source_dir=base.directory,
                    workbook_path=base.workbook_path,
                    sheet_name="; ".join(sheet["name"] for sheet in sheet_payloads[:20]),
                    image_path=base.image_path,
                    html_path=base.html_path,
                    structure_text=workbook_text,
                    sheet_text=workbook_card,
                    retrieval_card=workbook_card,
                    query=query,
                    retrieval_type="metadata",
                    retrieval_level="workbook",
                )
            )
        return workbook_candidates

    @staticmethod
    def _workbook_description(sheet_payloads: list[dict[str, Any]]) -> str:
        names = [str(sheet.get("name") or "") for sheet in sheet_payloads if sheet.get("name")]
        table_count = sum(
            len(sheet.get("tables") or [])
            for sheet in sheet_payloads
            if isinstance(sheet.get("tables"), list)
        )
        return (
            f"Workbook with {len(sheet_payloads)} prepared sheets and {table_count} detected tables: "
            + "; ".join(names[:20])
        )

    @staticmethod
    def _deduplicate_candidates(candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        unique: dict[tuple[str, str, str, str, str], SourceCandidate] = {}
        for candidate in candidates:
            key = (
                str(candidate.workbook_path.resolve()),
                candidate.sheet_name,
                candidate.table_id,
                candidate.retrieval_type,
                candidate.retrieval_level,
            )
            unique.setdefault(key, candidate)
        return list(unique.values())

    def _audit_row(self, candidate: SourceCandidate, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "retrieval_type": candidate.retrieval_type,
            "retrieval_level": candidate.retrieval_level,
            "score": candidate.score,
            "lexical_score": candidate.lexical_score,
            "embedding_score": candidate.embedding_score,
            "embedding_used": candidate.embedding_used,
            "entity_score": candidate.entity_score,
            "matched_terms": list(candidate.matched_terms),
            "missing_terms": list(candidate.missing_terms),
            "workbook": candidate.workbook_path.name,
            "sheet": candidate.sheet_name,
            "table_id": candidate.table_id,
            "table_name": candidate.table_name,
            "table_description": candidate.table_description,
            "retrieval_card_preview": candidate.retrieval_card[:600],
        }

    def _legacy_candidates_from_dir(self, source_dir: Path, allowed_paths: set[str], query: str) -> list[SourceCandidate]:
        data_candidates, _metadata_candidates = self._candidates_from_dir(source_dir, allowed_paths, query)
        return data_candidates

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
        retrieval_type: str = "data",
        retrieval_level: str = "table",
    ) -> SourceCandidate:
        lexical_score = _lexical_overlap_score(query, retrieval_card)
        entity_score, matched_terms, missing_terms = self._entity_match(query, retrieval_card)
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
            entity_score=entity_score,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            retrieval_type=retrieval_type,
            retrieval_level=retrieval_level,
        )

    def _entity_match(self, query: str, retrieval_card: str) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
        terms = self._query_terms(query)
        if not terms:
            return 0.0, (), ()
        haystack = retrieval_card.lower()
        matched = tuple(term for term in terms if term in haystack)
        missing = tuple(term for term in terms if term not in haystack)
        return len(matched) / len(terms), matched, missing

    def _query_terms(self, query: str) -> tuple[str, ...]:
        terms: list[str] = []
        seen: set[str] = set()
        for raw_term in re.findall(r"[0-9A-Za-zÀ-ỹ가-힣#./+-]+", str(query).lower()):
            term = raw_term.strip("?.!,;:()[]{}\"'")
            if not term or term in seen or term in _QUERY_STOPWORDS:
                continue
            if len(term) < 2 and not term.isdigit():
                continue
            seen.add(term)
            terms.append(term)
        return tuple(terms)

    def _candidate_from_dir(self, source_dir: Path, allowed_paths: set[str], query: str) -> SourceCandidate | None:
        candidates, _metadata_candidates = self._candidates_from_dir(source_dir, allowed_paths, query)
        return max(candidates, key=lambda candidate: candidate.lexical_score) if candidates else None

    _choose_from_reranker = staticmethod(choose_from_reranker)
