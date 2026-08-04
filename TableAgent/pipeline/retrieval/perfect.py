from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

from TableAgent.artifacts import iter_sheet_artifact_dirs
from TableAgent.pipeline.common import SourceCandidate
from .cards import build_source_retrieval_card


class PerfectRetrievalMixin:
    """Resolve explicitly provided or deterministic full-sheet sources."""

    def select_perfect(self, sample) -> SourceCandidate:
        configured_candidate = self._configured_perfect_candidate(sample)
        if configured_candidate is not None:
            self._progress("retrieval", sample=sample.sample_id, candidate=configured_candidate)
            return configured_candidate
        explicit_sheet_candidate = self._explicit_sheet_candidate(sample)
        if explicit_sheet_candidate is not None:
            if self.is_perfect_retrieval_excluded(
                explicit_sheet_candidate.workbook_path,
                explicit_sheet_candidate.sheet_name,
                sample.question,
            ):
                raise RuntimeError(
                    f"Perfect retrieval excludes sheet {explicit_sheet_candidate.sheet_name!r} "
                    f"from {explicit_sheet_candidate.workbook_path.name!r}."
                )
            self._progress("retrieval", sample=sample.sample_id, candidate=explicit_sheet_candidate)
            return explicit_sheet_candidate

        question_candidate = self._perfect_question_candidate(sample)
        if question_candidate is not None:
            self._progress("retrieval", sample=sample.sample_id, candidate=question_candidate)
            return question_candidate

        mapping = self._load_perfect_mapping()
        oracle = mapping.get(sample.sample_id) or mapping.get(sample.question)
        if oracle is None:
            raise RuntimeError(
                f"Perfect retrieval has no prior source mapping for sample {sample.sample_id!r}. "
                "Use --table-agent-source-artifacts with a run containing evaluations/report_1.json."
            )
        source_root = self._source_root()
        source_dir = self._find_source_dir(source_root, oracle["source_dir"])
        if not source_dir.is_dir():
            raise RuntimeError(f"Perfect retrieval source directory is missing: {source_dir}")
        candidate = self._full_sheet_candidate_from_dir(source_dir, sample)
        table_id = oracle.get("table_id", "")
        if table_id:
            table_candidates, _metadata_candidates = self._candidates_from_dir(
                source_dir, self._allowed_paths(sample), sample.question
            )
            candidate = next(
                (item for item in table_candidates if item.table_id == table_id),
                candidate,
            )
        if self.is_perfect_retrieval_excluded(
            candidate.workbook_path, candidate.sheet_name, sample.question
        ):
            raise RuntimeError(
                f"Perfect retrieval excludes sheet {candidate.sheet_name!r} "
                f"from {candidate.workbook_path.name!r}."
            )
        self._progress("retrieval", sample=sample.sample_id, candidate=candidate)
        return candidate

    def _configured_perfect_candidate(self, sample) -> SourceCandidate | None:
        raw = sample.raw if isinstance(sample.raw, dict) else {}
        spec = raw.get("perfect_source")
        if not isinstance(spec, dict):
            return None
        expected_workbook = self._normalized_name(spec.get("workbook"))
        expected_sheet = self._normalized_name(spec.get("sheet"))
        if not expected_workbook or not expected_sheet:
            raise RuntimeError(f"Invalid perfect source specification for {sample.sample_id!r}.")

        source_root = self._source_root()
        if not source_root.is_dir():
            raise RuntimeError(f"Prepared source directory is missing: {source_root}")
        for source_dir in iter_sheet_artifact_dirs(source_root):
            metadata_path = source_dir / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            workbook_name = self._normalized_name(
                PureWindowsPath(str(metadata.get("workbook_path", ""))).name
            )
            sheet_name = self._normalized_name(metadata.get("sheet_name"))
            if workbook_name != expected_workbook or sheet_name != expected_sheet:
                continue
            candidate = self._full_sheet_candidate_from_dir(source_dir, sample)
            if self.is_perfect_retrieval_excluded(
                candidate.workbook_path, candidate.sheet_name, sample.question
            ):
                raise RuntimeError(
                    f"Perfect retrieval excludes sheet {candidate.sheet_name!r} "
                    f"from {candidate.workbook_path.name!r}."
                )
            return candidate
        raise RuntimeError(
            f"Perfect retrieval source not found for {sample.sample_id!r}: "
            f"{spec.get('workbook')} / {spec.get('sheet')}"
        )

    def _perfect_question_candidate(self, sample) -> SourceCandidate | None:
        source_root = self._source_root()
        if not source_root.is_dir():
            return None
        allowed_paths = self._allowed_paths(sample)
        scored: list[tuple[float, float, str, str, SourceCandidate]] = []
        for source_dir in iter_sheet_artifact_dirs(source_root):
            data_candidates, _metadata_candidates = self._candidates_from_dir(
                source_dir, allowed_paths, sample.question
            )
            for candidate in data_candidates:
                question_score = self._perfect_question_score(sample.question, candidate)
                scored.append(
                    (
                        question_score,
                        candidate.lexical_score,
                        candidate.sheet_name,
                        candidate.table_id,
                        candidate,
                    )
                )
        if not scored:
            return None
        best = max(scored, key=lambda item: item[:4])
        if best[0] <= 0:
            return None
        best_candidate = self._full_sheet_candidate_from_dir(best[-1].directory, sample)
        if self.is_perfect_retrieval_excluded(
            best_candidate.workbook_path, best_candidate.sheet_name, sample.question
        ):
            raise RuntimeError(
                f"Perfect retrieval excludes sheet {best_candidate.sheet_name!r} "
                f"from {best_candidate.workbook_path.name!r}."
            )
        return best_candidate

    @classmethod
    def _perfect_question_score(cls, question: str, candidate: SourceCandidate) -> float:
        score = float(candidate.lexical_score)
        compact_question = cls._normalized_name(question)
        try:
            structure = yaml.safe_load(candidate.structure_text) or {}
        except yaml.YAMLError:
            return score
        if not isinstance(structure, dict):
            return score

        def reward_exact_label(value: Any, weight: float) -> None:
            nonlocal score
            normalized = cls._normalized_name(str(value or ""))
            if len(normalized) >= 3 and normalized in compact_question:
                score += weight

        def reward_headers(headers: Any) -> None:
            if not isinstance(headers, list):
                return
            for header in headers:
                if isinstance(header, dict):
                    reward_exact_label(header.get("label"), 8.0)
                    reward_headers(header.get("sub_headers"))

        for table in structure.values():
            if isinstance(table, dict):
                reward_exact_label(table.get("name"), 6.0)
                reward_headers(table.get("headers"))
        return score

    def _explicit_sheet_candidate(self, sample) -> SourceCandidate | None:
        source_root = self._source_root()
        if not source_root.is_dir():
            return None
        allowed_paths = self._allowed_paths(sample)
        matches: list[tuple[float, Path]] = []
        for source_dir in iter_sheet_artifact_dirs(source_root):
            metadata_path = source_dir / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            workbook_text = str(metadata.get("workbook_path") or "")
            if self._authorized_workbook_path(workbook_text, allowed_paths) is None:
                continue
            sheet_name = str(metadata.get("sheet_name") or "").strip()
            score = self._sheet_reference_score(sample.question, sheet_name)
            if score > 0:
                matches.append((score, source_dir))
        if not matches:
            return None
        _, source_dir = max(matches, key=lambda item: item[0])
        try:
            return self._full_sheet_candidate_from_dir(source_dir, sample)
        except RuntimeError:
            return None

    @staticmethod
    def _question_explicitly_names_sheet(question: str, sheet_name: str) -> bool:
        parts = [
            re.escape(part)
            for part in re.split(r"\s+", sheet_name.strip())
            if part
        ]
        if not parts:
            return False
        pattern = r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)"
        return re.search(pattern, str(question), flags=re.IGNORECASE) is not None

    @classmethod
    def _sheet_reference_score(cls, question: str, sheet_name: str) -> float:
        if not str(sheet_name).strip():
            return 0.0
        score = float(
            len(sheet_name)
            if cls._question_explicitly_names_sheet(question, sheet_name)
            else 0.0
        )
        sheet_alias = cls._sheet_alias(sheet_name)
        marker_pattern = re.compile(
            r"(?:sheet|worksheet|tab|ph[oò]ng|시트)\s+([^,;:()]+)",
            re.IGNORECASE,
        )
        for marker in marker_pattern.finditer(str(question)):
            words = re.findall(r"\w+", marker.group(1), flags=re.UNICODE)[:4]
            for length in range(1, len(words) + 1):
                candidate = cls._sheet_alias(" ".join(words[:length]))
                if not candidate:
                    continue
                similarity = SequenceMatcher(None, candidate, sheet_alias).ratio()
                if candidate == sheet_alias:
                    score = max(score, 1000.0 + len(sheet_alias))
                elif similarity >= 0.8:
                    score = max(score, 500.0 + similarity)
        return score

    @classmethod
    def _sheet_alias(cls, value: str) -> str:
        tokens = re.findall(
            r"\w+", unicodedata.normalize("NFKD", str(value)).casefold()
        )
        normalized = []
        for token in tokens:
            token = "".join(character for character in token if character.isalnum())
            if re.fullmatch(r"f\d+", token):
                token = token[1:]
            if token:
                normalized.append(token)
        return "".join(normalized)

    def load_perfect_candidates(self, sample) -> list[SourceCandidate]:
        try:
            return [self.select_perfect(sample)]
        except RuntimeError:
            return []

    def _load_perfect_mapping(self) -> dict[str, dict[str, str]]:
        if self._perfect_mapping is not None:
            return self._perfect_mapping
        source_artifact_dir = self.settings.source_artifact_dir
        if source_artifact_dir is None:
            self._perfect_mapping = {}
            return self._perfect_mapping
        run_root = source_artifact_dir
        if run_root.name == "shared" and run_root.parent.name == "artifacts":
            run_root = run_root.parent.parent
        elif run_root.name == "artifacts":
            run_root = run_root.parent
        mapping: dict[str, dict[str, str]] = {}
        for report_path in sorted((run_root / "evaluations").glob("report_*.json")):
            try:
                results = json.loads(report_path.read_text(encoding="utf-8")).get("results", [])
            except (OSError, json.JSONDecodeError):
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
                qa = metadata.get("qa") if isinstance(metadata.get("qa"), dict) else {}
                if result.get("error") is True or qa.get("success") is False:
                    continue
                artifact_text = str(metadata.get("artifact_dir") or "")
                structure_text = str(metadata.get("structure_path") or "")
                source_dir = PureWindowsPath(artifact_text).name if artifact_text else ""
                if not source_dir and structure_text:
                    source_dir = PureWindowsPath(structure_text).parent.name
                if not source_dir:
                    continue
                retrieval_info = metadata.get("retrieval_info") if isinstance(metadata.get("retrieval_info"), dict) else {}
                entry = {"source_dir": source_dir, "table_id": str(retrieval_info.get("table_id") or "")}
                if result.get("sample_id"):
                    mapping[str(result["sample_id"])] = entry
                if result.get("question"):
                    mapping[str(result["question"])] = entry
        self._perfect_mapping = mapping
        return mapping

    def _full_sheet_candidate_from_dir(self, source_dir: Path, sample) -> SourceCandidate:
        paths = [
            source_dir / "metadata.json",
            source_dir / "structure.yaml",
            source_dir / "sheet_text.txt",
            source_dir / "table.png",
        ]
        if not all(path.is_file() for path in paths):
            raise RuntimeError(f"Prepared source is incomplete: {source_dir}")
        metadata = json.loads(paths[0].read_text(encoding="utf-8"))
        workbook_text = str(metadata.get("workbook_path", ""))
        workbook_path = self._authorized_workbook_path(
            workbook_text, self._allowed_paths(sample)
        )
        if workbook_path is None:
            raise RuntimeError(
                f"Perfect retrieval source is not authorized for the sample: {workbook_text}"
            )
        structure_text = paths[1].read_text(encoding="utf-8")
        sheet_text = paths[2].read_text(encoding="utf-8")
        sheet_name = str(metadata.get("sheet_name", ""))
        return self._source_candidate(
            source_dir=source_dir,
            workbook_path=workbook_path,
            sheet_name=sheet_name,
            image_path=paths[3],
            html_path=(source_dir / "table.html") if (source_dir / "table.html").is_file() else None,
            structure_text=structure_text,
            sheet_text=sheet_text,
            retrieval_card=build_source_retrieval_card(
                workbook_path, sheet_name, structure_text, sheet_text
            ),
            query=sample.question,
        )

    def _source_root(self) -> Path:
        artifact_dir = self.settings.source_artifact_dir or self.settings.artifact_dir
        return artifact_dir / "sources"

    @staticmethod
    def _find_source_dir(source_root: Path, source_name: str) -> Path:
        direct = source_root / str(source_name)
        if direct.is_dir():
            return direct
        for candidate in iter_sheet_artifact_dirs(source_root):
            if candidate.name == str(source_name):
                return candidate
        return direct

    @staticmethod
    def _allowed_paths(sample) -> set[str]:
        return {
            str(Path(value.strip()).resolve())
            for value in str(sample.table_path).split(";")
            if value.strip()
        }

    @classmethod
    def _authorized_workbook_path(
        cls, workbook_text: str, allowed_paths: set[str]
    ) -> Path | None:
        configured = Path(workbook_text)
        if not allowed_paths:
            return configured
        if str(configured.resolve()) in allowed_paths:
            return configured
        configured_name = cls._normalized_name(PureWindowsPath(workbook_text).name)
        for allowed_path in allowed_paths:
            candidate = Path(allowed_path)
            if cls._normalized_name(candidate.name) == configured_name:
                return candidate
        return None

    @staticmethod
    def _normalized_name(value: str) -> str:
        text = unicodedata.normalize("NFKD", str(value or "")).casefold()
        return "".join(character for character in text if character.isalnum())

    @classmethod
    def is_perfect_retrieval_excluded(
        cls, workbook_path: Path, sheet_name: str, question: str = ""
    ) -> bool:
        del question
        normalized_book = cls._normalized_name(workbook_path.name)
        target_book = cls._normalized_name(
            "LV01_설비_REPORT 2026년 설비유지보수 계획 VER 1.0_KR_202603.26.xlsx"
        )
        return normalized_book == target_book and cls._normalized_name(sheet_name) == "sheet3"

    def _progress(self, stage: str, *, sample: str, candidate: SourceCandidate) -> None:
        progress = getattr(self.templates, "_progress", None)
        if callable(progress):
            progress(
                stage,
                sample=sample,
                workbook=candidate.workbook_path.name,
                sheet=candidate.sheet_name,
            )
