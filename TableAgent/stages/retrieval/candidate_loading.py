from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from TableAgent.artifacts import iter_sheet_artifact_dirs
from TableAgent.pipeline.common import SourceCandidate
from TableAgent.structure.layout.parsing import _is_valid_structure
from TableAgent.utils.table_text import _lexical_overlap_score

from .cards import (
    build_metadata_retrieval_card,
    build_sheet_metadata_payload,
    build_source_retrieval_card,
    build_table_retrieval_cards,
)
from .reranking import choose_from_reranker


_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "of", "on", "or",
    "sheet", "table", "the", "to", "what", "which", "with", "bảng", "bao", "các", "câu",
    "cho", "của", "gì", "hãy", "hỏi", "không", "là", "nào", "nêu", "những", "nhiêu",
    "trong", "và", "với",
}


class RetrievalCandidateLoadingMixin:
    """Load prepared sheet artifacts and build data/metadata candidates."""

    def load_candidates(self, sample) -> list[SourceCandidate]:
        data_candidates, _metadata_candidates = self._candidate_pools(sample)
        return self._rank_candidates(data_candidates, sample.question)

    def _candidate_pools(self, sample) -> tuple[list[SourceCandidate], list[SourceCandidate]]:
        artifact_dir = self.settings.source_artifact_dir or self.settings.artifact_dir
        source_dirs = artifact_dir / "sources"
        if not source_dirs.is_dir():
            return [], []
        allowed_paths = {
            str(Path(value.strip()).resolve())
            for value in str(sample.table_path).split(";")
            if value.strip()
        }
        selected_values = (
            sample.raw.get("selected_sheets", [])
            if isinstance(sample.raw, dict)
            else []
        )
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
        sheet_metadata_candidates = self._deduplicate_candidates(
            sheet_metadata_candidates
        )
        return data_candidates, [
            *self._workbook_metadata_candidates(
                sheet_metadata_candidates, sample.question
            ),
            *sheet_metadata_candidates,
        ]

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
        if not (
            metadata_path.is_file()
            and sheet_text_path.is_file()
            and image_path.is_file()
        ):
            return [], []

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        workbook_text = str(metadata.get("workbook_path", ""))
        workbook_path = self._authorized_workbook_path(workbook_text, allowed_paths)
        if workbook_path is None:
            return [], []
        sheet_text = sheet_text_path.read_text(encoding="utf-8")
        sheet_name = str(metadata.get("sheet_name", ""))
        card_workbook_path = Path(
            str(metadata.get("workbook_name") or workbook_path.name)
        )
        if selected_sheets and sheet_name not in selected_sheets:
            return [], []
        structure_text = (
            structure_path.read_text(encoding="utf-8")
            if structure_path.is_file()
            else ""
        )
        if not _is_valid_structure(structure_text):
            structure_text = self._fallback_structure_text(source_dir, sheet_name)

        sheet_metadata_payload = build_sheet_metadata_payload(
            card_workbook_path,
            sheet_name,
            structure_text,
            sheet_text,
            self._read_sheet_metadata(source_dir),
        )
        sheet_metadata_text = yaml.safe_dump(
            {"metadata": sheet_metadata_payload},
            allow_unicode=True,
            sort_keys=False,
        )
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
        table_cards = build_table_retrieval_cards(
            card_workbook_path, sheet_name, structure_text, sheet_text
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
                    retrieval_type="data",
                    retrieval_level="table",
                )
                for table_card in table_cards
            ], [metadata_candidate]

        retrieval_card = build_source_retrieval_card(
            card_workbook_path, sheet_name, structure_text, sheet_text
        )
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

    @staticmethod
    def _read_sheet_metadata(source_dir: Path) -> dict[str, Any]:
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
                pass
        table_metadata = source_dir / "table.metadata.json"
        if used_range is None and table_metadata.is_file():
            try:
                payload = json.loads(table_metadata.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    used_range = payload.get("cell_range")
            except Exception:
                pass
        table_id = re.sub(r"[^0-9A-Za-z_]+", "_", sheet_name.lower()).strip("_") or "table1"
        payload = {
            "table1": {
                "id": table_id,
                "name": sheet_name,
                "description": (
                    "Fallback structure generated from prepared sheet metadata "
                    "because layout structure generation failed."
                ),
                "sheet": sheet_name,
                "table_range": used_range,
                "headers": [],
            }
        }
        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)

    def _workbook_metadata_candidates(
        self, sheet_candidates: list[SourceCandidate], query: str
    ) -> list[SourceCandidate]:
        grouped: dict[Path, list[SourceCandidate]] = {}
        for candidate in sheet_candidates:
            grouped.setdefault(candidate.workbook_path.resolve(), []).append(candidate)
        workbook_candidates: list[SourceCandidate] = []
        for sheets in grouped.values():
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
                sheet_payloads.append(
                    {
                        "name": sheet.sheet_name,
                        "description": metadata.get("description", "") if isinstance(metadata, dict) else "",
                        "used_range": metadata.get("used_range", "") if isinstance(metadata, dict) else "",
                        "merged_ranges": metadata.get("merged_ranges", []) if isinstance(metadata, dict) else [],
                        "sheet_summary": metadata.get("sheet_summary", "") if isinstance(metadata, dict) else "",
                        "preview": metadata.get("preview", "") if isinstance(metadata, dict) else "",
                        "tables": metadata.get("tables", []) if isinstance(metadata, dict) else [],
                    }
                )
            workbook_payload = {
                "type": "workbook",
                "workbook": base.workbook_path.name,
                "description": self._workbook_description(sheet_payloads),
                "sheets": sheet_payloads,
            }
            workbook_text = yaml.safe_dump(
                {"metadata": workbook_payload}, allow_unicode=True, sort_keys=False
            )
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
            "artifact_id": candidate.artifact_id,
            "rank": rank,
            "retrieval_type": candidate.retrieval_type,
            "retrieval_level": candidate.retrieval_level,
            "score": candidate.score,
            "lexical_score": candidate.lexical_score,
            "bm25_score": candidate.bm25_score,
            "embedding_score": candidate.embedding_score,
            "embedding_used": candidate.embedding_used,
            "embedding_model": candidate.embedding_model,
            "embedding_source": candidate.embedding_source,
            "workbook_reference_score": candidate.workbook_reference_score,
            "entity_score": candidate.entity_score,
            "matched_terms": list(candidate.matched_terms),
            "missing_terms": list(candidate.missing_terms),
            "workbook": candidate.workbook_path.name,
            "sheet": candidate.sheet_name,
            "sheets": list(candidate.sheet_names),
            "table_id": candidate.table_id,
            "table_name": candidate.table_name,
            "table_description": candidate.table_description,
            "retrieval_card_preview": candidate.retrieval_card[:600],
        }

    def _legacy_candidates_from_dir(
        self, source_dir: Path, allowed_paths: set[str], query: str
    ) -> list[SourceCandidate]:
        data_candidates, _metadata_candidates = self._candidates_from_dir(
            source_dir, allowed_paths, query
        )
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
        artifact_id: str = "",
        embedding_vector: tuple[float, ...] = (),
        embedding_model: str = "",
        embedding_score: float = 0.0,
        embedding_used: bool = False,
    ) -> SourceCandidate:
        lexical_score = _lexical_overlap_score(query, retrieval_card)
        entity_score, matched_terms, missing_terms = self._entity_match(
            query, retrieval_card
        )
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
            embedding_score=embedding_score,
            embedding_used=embedding_used,
            retrieval_card=retrieval_card,
            table_id=table_id,
            table_name=table_name,
            table_description=table_description,
            entity_score=entity_score,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            retrieval_type=retrieval_type,
            retrieval_level=retrieval_level,
            artifact_id=artifact_id,
            embedding_vector=embedding_vector,
            embedding_model=embedding_model,
        )

    def _entity_match(
        self, query: str, retrieval_card: str
    ) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
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
        for raw_term in re.findall(
            r"[0-9A-Za-zÀ-ỹ가-힣#./+-]+", str(query).lower()
        ):
            term = raw_term.strip("?.!,;:()[]{}\"'")
            if not term or term in seen or term in _QUERY_STOPWORDS:
                continue
            if len(term) < 2 and not term.isdigit():
                continue
            seen.add(term)
            terms.append(term)
        return tuple(terms)

    def _candidate_from_dir(
        self, source_dir: Path, allowed_paths: set[str], query: str
    ) -> SourceCandidate | None:
        candidates, _metadata_candidates = self._candidates_from_dir(
            source_dir, allowed_paths, query
        )
        return (
            max(candidates, key=lambda candidate: candidate.lexical_score)
            if candidates
            else None
        )

    _choose_from_reranker = staticmethod(choose_from_reranker)
