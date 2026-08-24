from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from TableAgent.rendering.converter import sample_to_xlsx
from TableAgent.pipeline.sample import EvalSample

from TableAgent.configs import TableAgentConfig
from TableAgent.stages.structure.metadata import SheetMetadata
from TableAgent.utils.paths import safe_name
from TableAgent.stages.structure.structure_prompts import LAYOUT_MAS_SYSTEM_PROMPT, LAYOUT_MAS_USER_PROMPT_TEMPLATE
from TableAgent.stages.structure.layout.direction_prompts import (
    DIRECTION_SYSTEM_PROMPT,
    DIRECTION_USER_PROMPT_TEMPLATE,
)
from TableAgent.stages.structure.layout.workflow import TableLayoutWorkflow

CACHE_SCHEMA_VERSION = 6
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class StructureCacheRecord:
    key: str
    directory: Path
    workbook_path: Path
    sheet_name: str
    structure_path: Path
    manifest_path: Path
    status: str
    cache_hit: bool

    @property
    def valid(self) -> bool:
        # Verification quality is diagnostic; a persisted structure remains usable.
        return self.structure_path.is_file()


class StructureCache:
    def __init__(
        self,
        settings: TableAgentConfig,
        workflow: TableLayoutWorkflow | None,
        metadata_for_workbook_sheet: Callable[[Path, str], SheetMetadata],
        progress_callback: Callable[..., None] | None = None,
    ):
        self.settings = settings
        self.workflow = workflow
        self.metadata_for_workbook_sheet = metadata_for_workbook_sheet
        self.progress_callback = progress_callback
        namespace = safe_name(settings.cache_namespace) or "default"
        self.root = settings.structure_cache_dir / f"v{CACHE_SCHEMA_VERSION}" / "datasets" / namespace

    def set_progress_callback(self, callback: Callable[..., None] | None) -> None:
        self.progress_callback = callback

    def prepare(self, sample: EvalSample, *, force: bool) -> StructureCacheRecord:
        if self.workflow is None:
            raise RuntimeError("Verification requires a configured layout VLM client")
        source_path, source_format, source_hash = self._materialize_source(sample)
        sheet_name = self._sheet_name(source_path)
        entry_key = self._entry_name(source_path, sheet_name, source_hash)
        recipe_key = self._recipe_key(source_hash, sheet_name)
        directory = self._preferred_directory(sample, source_path, sheet_name, source_hash)
        key = directory.name or entry_key
        existing = self._resolve_record(
            sample,
            source_path=source_path,
            source_hash=source_hash,
            sheet_name=sheet_name,
            preferred_directory=directory,
            preferred_key=entry_key,
            cache_hit=True,
        )
        if existing is not None and existing.valid and not force:
            self._progress(
                "structure_done",
                sample=sample.sample_id,
                workbook=source_path.name,
                sheet=sheet_name,
                status="cached",
            )
            if existing.directory != directory:
                return self._promote_record(existing, directory, key=key, recipe_key=recipe_key)
            return replace(existing, key=key) if existing.key != key else existing

        with self._lock(entry_key):
            existing = self._resolve_record(
                sample,
                source_path=source_path,
                source_hash=source_hash,
                sheet_name=sheet_name,
                preferred_directory=directory,
                preferred_key=entry_key,
                cache_hit=True,
            )
            if existing is not None and existing.valid and not force:
                self._progress(
                    "structure_done",
                    sample=sample.sample_id,
                    workbook=source_path.name,
                    sheet=sheet_name,
                    status="cached",
                )
                if existing.directory != directory:
                    return self._promote_record(existing, directory, key=key, recipe_key=recipe_key)
                return replace(existing, key=key) if existing.key != key else existing

            staging = directory.with_name(f".{directory.name}.staging-{threading.get_ident()}")
            self._remove_tree_with_retry(staging)
            staging.mkdir(parents=True, exist_ok=True)
            workbook_path = staging / "workbook.xlsx"
            shutil.copy2(source_path, workbook_path)
            metadata = self.metadata_for_workbook_sheet(workbook_path, sheet_name)
            self._progress(
                "prepare_layout",
                sample=sample.sample_id,
                workbook=source_path.name,
                sheet=sheet_name,
                range=getattr(metadata, "used_range", None),
            )
            result = self.workflow.run(
                workbook_path=workbook_path,
                sheet_name=sheet_name,
                metadata=metadata,
                output_dir=staging,
            )
            structure_path = staging / "structure.yaml"
            if result.structure_text.strip():
                structure_path.write_text(result.structure_text, encoding="utf-8")
            manifest = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": key,
                "recipe_key": recipe_key,
                "source_format": source_format,
                "source_path": str(source_path.resolve()),
                "source_sha256": source_hash,
                "source_name": source_path.name,
                "table_id": sample.table_id,
                "sheet_name": sheet_name,
                "status": result.verification.get("status", "not_good"),
                "workflow_version": 6,
                "artifacts": {"structure": "structure.yaml", "workbook": "workbook.xlsx"},
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._remove_tree_with_retry(directory)
            directory.parent.mkdir(parents=True, exist_ok=True)
            self._promote_staging(staging, directory)
            self._progress(
                "structure_done",
                sample=sample.sample_id,
                workbook=source_path.name,
                sheet=sheet_name,
                status=manifest["status"],
            )
            return self._read_record(
                directory,
                key,
                sheet_name,
                source_hash=source_hash,
                cache_hit=False,
            ) or StructureCacheRecord(
                key,
                directory,
                directory / "workbook.xlsx",
                sheet_name,
                directory / "structure.yaml",
                directory / "manifest.json",
                manifest["status"],
                False,
            )

    def load(self, sample: EvalSample) -> StructureCacheRecord | None:
        source_path, _, source_hash = self._materialize_source(sample)
        sheet_name = self._sheet_name(source_path)
        entry_key = self._entry_name(source_path, sheet_name, source_hash)
        directory = self._preferred_directory(sample, source_path, sheet_name, source_hash)
        key = directory.name or entry_key
        record = self._resolve_record(
            sample,
            source_path=source_path,
            source_hash="" if self.settings.trust_structure_cache_path else source_hash,
            sheet_name=sheet_name,
            preferred_directory=directory,
            preferred_key=entry_key,
            cache_hit=True,
        )
        if record is None or not record.valid:
            return record
        # Do not promote on load. QA/reuse must keep the selected structure folder
        # immutable (especially previous run artifacts with metadata.json).
        return replace(record, key=key) if record.key != key else record

    def _materialize_source(self, sample: EvalSample) -> tuple[Path, str, str]:
        values = [Path(value.strip()) for value in str(sample.table_path or "").split(";") if value.strip()]
        if values and values[0].is_file() and values[0].suffix.lower() == ".xlsx":
            source_path = values[0]
            # Compressed structure inputs carry a map back to the original
            # workbook. Keep the original hash in the cache key so QA-only
            # runs can reuse the structure with the original workbook.
            hash_path = source_path
            raw = sample.raw if isinstance(sample.raw, dict) else {}
            original_paths = raw.get("original_workbook_paths")
            if isinstance(original_paths, dict):
                original = original_paths.get(str(source_path.resolve()))
                if original:
                    candidate = Path(str(original))
                    if candidate.is_file():
                        hash_path = candidate
            return source_path, "xlsx", self._sha256(hash_path)
        temporary = self.root / ".inputs" / f"{safe_name(sample.sample_id)}.xlsx"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        sample_to_xlsx(sample, temporary)
        source_payload = json.dumps(
            {
                "table_id": sample.table_id,
                "table_content": sample.table_content,
                "tables": sample.raw.get("tables"),
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return temporary, "converted", hashlib.sha256(source_payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _sheet_name(workbook_path: Path) -> str:
        import openpyxl

        workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
        try:
            return workbook.sheetnames[0]
        finally:
            workbook.close()

    def _preferred_directory(
        self,
        sample: EvalSample,
        source_path: Path,
        sheet_name: str,
        source_hash: str,
    ) -> Path:
        table_name = self._table_name(sample, source_path)
        leaf_name = self._leaf_name(sample, source_path, sheet_name, source_hash)
        return self.root / table_name / leaf_name

    def _table_name(self, sample: EvalSample, source_path: Path) -> str:
        return safe_name(sample.table_id or source_path.stem or sample.sample_id)[:80] or "table"

    def _leaf_name(
        self,
        sample: EvalSample,
        source_path: Path,
        sheet_name: str,
        source_hash: str,
    ) -> str:
        """Human-trackable cache leaf: <table_id>_<recipe_hash>."""
        table_name = self._table_name(sample, source_path)
        recipe_key = self._recipe_key(source_hash, sheet_name)
        return f"{table_name}_{recipe_key}"

    def _entry_name(self, source_path: Path, sheet_name: str, source_hash: str) -> str:
        source_part = safe_name(source_path.name)[:80] or "source"
        sheet_part = safe_name(sheet_name)[:40] or "sheet"
        return f"{source_part}__{sheet_part}__{source_hash[:8]}"

    def _recipe_key(self, source_hash: str, sheet_name: str) -> str:
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "source_sha256": source_hash,
            "sheet_name": sheet_name,
            "workflow_version": 6,
            "viewport_rows": self.settings.viewport_rows,
            "viewport_columns": self.settings.viewport_columns,
            "shift_cells": self.settings.shift_cells,
            "max_retry": self.settings.max_retry,
            "structure_data_only": self.settings.structure_data_only,
            "layout_model": self.settings.layout_model_identity,
            "layout_prompt_sha256": hashlib.sha256(
                (
                    DIRECTION_SYSTEM_PROMPT
                    + DIRECTION_USER_PROMPT_TEMPLATE
                    + LAYOUT_MAS_SYSTEM_PROMPT
                    + LAYOUT_MAS_USER_PROMPT_TEMPLATE
                ).encode("utf-8")
            ).hexdigest(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]

    # Back-compat alias used by older tests/call sites.
    def _key(self, source_hash: str, sheet_name: str) -> str:
        return self._recipe_key(source_hash, sheet_name)

    def _resolve_record(
        self,
        sample: EvalSample,
        *,
        source_path: Path,
        source_hash: str,
        sheet_name: str,
        preferred_directory: Path,
        preferred_key: str,
        cache_hit: bool,
    ) -> StructureCacheRecord | None:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def add(path: Path | None) -> None:
            if path is None:
                return
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                return
            seen.add(resolved)
            candidates.append(path)

        def read_first(paths: list[Path]) -> StructureCacheRecord | None:
            for directory in paths:
                record = self._read_record(
                    directory,
                    preferred_key,
                    sheet_name,
                    source_hash=source_hash,
                    cache_hit=cache_hit,
                )
                if record is not None and record.valid:
                    return record
            return None

        table_name = self._table_name(sample, source_path)
        structure_root = Path(self.settings.structure_cache_dir)
        recipe_key = self._recipe_key(source_hash, sheet_name)
        leaf_name = self._leaf_name(sample, source_path, sheet_name, source_hash)

        add(preferred_directory)
        add(self.root / table_name / leaf_name)
        add(self.root / table_name / preferred_key)
        # Legacy bare recipe/hash leaves from earlier cache layouts.
        add(self.root / table_name / recipe_key)
        add(structure_root / leaf_name)
        add(structure_root / preferred_key)
        add(structure_root / recipe_key)
        add(structure_root / table_name / leaf_name)
        add(structure_root / table_name / preferred_key)
        add(structure_root / table_name / recipe_key)

        hit = read_first(candidates)
        if hit is not None:
            return hit

        # Fallback scan for legacy flat/run-local caches and materialized run folders.
        search_roots = [self.root, structure_root]
        if self.settings.trust_structure_cache_path:
            # An explicitly selected cache path is authoritative, but still
            # restrict discovery to the requested table to avoid loading a
            # same-named sheet from another workbook.
            trusted_roots = [root / table_name for root in search_roots]
            search_roots = [root for root in trusted_roots if root.exists()] or search_roots
        for root in search_roots:
            if not root.exists():
                continue
            marker_paths = []
            try:
                marker_paths.extend(root.rglob("manifest.json"))
                marker_paths.extend(root.rglob("metadata.json"))
            except OSError:
                continue
            for marker_path in marker_paths:
                # Ignore render sidecar metadata files such as table.metadata.json.
                if marker_path.name == "metadata.json" and marker_path.stem != "metadata":
                    continue
                try:
                    payload = json.loads(marker_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                payload_hash = str(
                    payload.get("source_sha256")
                    or payload.get("workbook_sha256")
                    or ""
                )
                payload_sheet = str(payload.get("sheet_name") or "")
                if source_hash and payload_hash and payload_hash != source_hash:
                    continue
                if payload_sheet and payload_sheet != sheet_name:
                    continue
                record = self._read_record(
                    marker_path.parent,
                    preferred_key,
                    sheet_name,
                    source_hash=source_hash,
                    cache_hit=cache_hit,
                )
                if record is not None and record.valid:
                    return record
        return None

    def _promote_record(
        self,
        record: StructureCacheRecord,
        directory: Path,
        *,
        key: str,
        recipe_key: str,
    ) -> StructureCacheRecord:
        if record.directory.resolve() == directory.resolve():
            return record
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            shutil.rmtree(directory)
        shutil.copytree(record.directory, directory)
        manifest_path = directory / "manifest.json"
        metadata_path = directory / "metadata.json"
        manifest: dict[str, Any] = {}
        source_hash = ""
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            source_hash = str(manifest.get("source_sha256") or "")
        elif metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            source_hash = str(
                metadata.get("workbook_sha256")
                or metadata.get("source_sha256")
                or ""
            )
            verification = metadata.get("verification")
            status = (
                str(verification.get("status"))
                if isinstance(verification, dict) and verification.get("status")
                else str(metadata.get("status") or record.status)
            )
            manifest = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": key,
                "recipe_key": recipe_key,
                "source_format": "xlsx",
                "source_path": str(
                    metadata.get("workbook_path")
                    or metadata.get("source_path")
                    or record.workbook_path
                ),
                "source_sha256": source_hash,
                "sheet_name": str(metadata.get("sheet_name") or record.sheet_name),
                "status": status,
                "workflow_version": int(
                    metadata.get("layout_workflow_version")
                    or metadata.get("workflow_version")
                    or CACHE_SCHEMA_VERSION
                ),
                "artifacts": {
                    "structure": "structure.yaml",
                    "workbook": "workbook.xlsx",
                },
            }
            # Ensure preferred cache layout has a workbook.xlsx for future loads.
            target_workbook = directory / "workbook.xlsx"
            source_workbook = Path(
                str(
                    metadata.get("workbook_path")
                    or metadata.get("source_path")
                    or record.workbook_path
                )
            )
            if not target_workbook.is_file() and source_workbook.is_file():
                try:
                    shutil.copy2(source_workbook, target_workbook)
                except OSError:
                    pass
        manifest["cache_key"] = key
        manifest["recipe_key"] = recipe_key
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        promoted = self._read_record(
            directory,
            key,
            record.sheet_name,
            source_hash=source_hash,
            cache_hit=True,
        )
        return promoted or StructureCacheRecord(
            key,
            directory,
            directory / "workbook.xlsx",
            record.sheet_name,
            directory / "structure.yaml",
            manifest_path,
            str(manifest.get("status", record.status)),
            True,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _remove_tree_with_retry(path: Path, attempts: int = 8) -> None:
        if not path.exists():
            return
        last_error: OSError | None = None
        for attempt in range(attempts):
            try:
                shutil.rmtree(path)
                return
            except OSError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(0.25 * (attempt + 1))
        if last_error is not None:
            raise last_error

    @classmethod
    def _promote_staging(cls, staging: Path, directory: Path, attempts: int = 8) -> None:
        last_error: OSError | None = None
        for attempt in range(attempts):
            try:
                staging.replace(directory)
                return
            except OSError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(0.25 * (attempt + 1))
                    cls._remove_tree_with_retry(directory)
        if last_error is not None:
            raise last_error

    @staticmethod
    def _read_record(
        directory: Path,
        key: str,
        sheet_name: str,
        *,
        source_hash: str = "",
        cache_hit: bool,
    ) -> StructureCacheRecord | None:
        structure_path = directory / "structure.yaml"
        if not structure_path.is_file() or structure_path.stat().st_size <= 0:
            return None

        manifest_path = directory / "manifest.json"
        metadata_path = directory / "metadata.json"
        payload: dict[str, Any] = {}
        status = "not_good"
        payload_sheet = ""
        payload_hash = ""
        payload_key = ""

        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            payload_sheet = str(payload.get("sheet_name") or "")
            payload_hash = str(payload.get("source_sha256") or "")
            payload_key = str(payload.get("cache_key") or "")
            status = str(payload.get("status", "not_good"))
        elif metadata_path.is_file():
            # Run-local materialized structure trees keep metadata.json instead of
            # the global-cache manifest.json. Treat them as valid reuse sources.
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            payload_sheet = str(payload.get("sheet_name") or "")
            payload_hash = str(
                payload.get("workbook_sha256")
                or payload.get("source_sha256")
                or ""
            )
            verification = payload.get("verification")
            if isinstance(verification, dict) and verification.get("status"):
                status = str(verification.get("status"))
            else:
                status = str(payload.get("status") or "good")
            manifest_path = metadata_path
            payload_key = directory.name
        else:
            return None

        if payload_sheet and payload_sheet != sheet_name:
            return None
        if source_hash and payload_hash and payload_hash != source_hash:
            return None

        workbook_path = directory / "workbook.xlsx"
        if not workbook_path.is_file():
            # Materialized run artifacts often keep the original workbook path.
            candidate = str(
                payload.get("source_path")
                or payload.get("workbook_path")
                or ""
            ).strip()
            if candidate:
                workbook_path = Path(candidate)
        if not workbook_path.is_file():
            return None

        return StructureCacheRecord(
            key=key or payload_key or directory.name,
            directory=directory,
            workbook_path=workbook_path,
            sheet_name=sheet_name,
            structure_path=structure_path,
            manifest_path=manifest_path,
            status=status,
            cache_hit=cache_hit,
        )

    def _progress(self, stage: str, **fields: Any) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(stage, **fields)
        except TypeError:
            # Some call sites pass a single formatted string callback.
            from TableAgent.pipeline.progress import format_progress

            self.progress_callback(format_progress(stage, fields))

    @staticmethod
    def _lock(key: str):
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.Lock())
        return lock
