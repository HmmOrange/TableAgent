from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets.base import EvalSample
from utils.workbook_converter import sample_to_xlsx

from TableAgent.configs import TableAgentConfig
from TableAgent.perception.metadata import SheetMetadata
from TableAgent.pipeline.common import safe_name
from TableAgent.prompts.structure import LAYOUT_MAS_SYSTEM_PROMPT, LAYOUT_MAS_USER_PROMPT_TEMPLATE
from TableAgent.structure.layout.workflow import TableLayoutWorkflow


CACHE_SCHEMA_VERSION = 1
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
        return self.status == "good" and self.structure_path.is_file()


class StructureCache:
    def __init__(
        self,
        settings: TableAgentConfig,
        workflow: TableLayoutWorkflow | None,
        metadata_for_workbook_sheet: Callable[[Path, str], SheetMetadata],
    ):
        self.settings = settings
        self.workflow = workflow
        self.metadata_for_workbook_sheet = metadata_for_workbook_sheet
        self.root = settings.structure_cache_dir / f"v{CACHE_SCHEMA_VERSION}"

    def prepare(self, sample: EvalSample, *, force: bool) -> StructureCacheRecord:
        if self.workflow is None:
            raise RuntimeError("Verification requires a configured layout VLM client")
        source_path, source_format, source_hash = self._materialize_source(sample)
        sheet_name = self._sheet_name(source_path)
        key = self._key(source_hash, sheet_name)
        directory = self.root / safe_name(sample.table_id or sample.sample_id)[:80] / key
        manifest_path = directory / "manifest.json"
        existing = self._read_record(directory, key, sheet_name, cache_hit=True)
        if existing is not None and existing.valid and not force:
            return existing

        with self._lock(key):
            existing = self._read_record(directory, key, sheet_name, cache_hit=True)
            if existing is not None and existing.valid and not force:
                return existing
            staging = directory.with_name(f".{directory.name}.staging-{threading.get_ident()}")
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True, exist_ok=True)
            workbook_path = staging / "workbook.xlsx"
            shutil.copy2(source_path, workbook_path)
            metadata = self.metadata_for_workbook_sheet(workbook_path, sheet_name)
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
                "source_format": source_format,
                "source_path": str(source_path.resolve()),
                "source_sha256": source_hash,
                "sheet_name": sheet_name,
                "status": result.verification.get("status", "not_good"),
                "workflow_version": 4,
                "artifacts": {"structure": "structure.yaml", "workbook": "workbook.xlsx"},
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if directory.exists():
                shutil.rmtree(directory)
            directory.parent.mkdir(parents=True, exist_ok=True)
            staging.replace(directory)
            return self._read_record(directory, key, sheet_name, cache_hit=False) or StructureCacheRecord(
                key, directory, directory / "workbook.xlsx", sheet_name, directory / "structure.yaml", manifest_path, manifest["status"], False
            )

    def load(self, sample: EvalSample) -> StructureCacheRecord | None:
        source_path, _, source_hash = self._materialize_source(sample)
        sheet_name = self._sheet_name(source_path)
        key = self._key(source_hash, sheet_name)
        directory = self.root / safe_name(sample.table_id or sample.sample_id)[:80] / key
        return self._read_record(directory, key, sheet_name, cache_hit=True)

    def _materialize_source(self, sample: EvalSample) -> tuple[Path, str, str]:
        values = [Path(value.strip()) for value in str(sample.table_path or "").split(";") if value.strip()]
        if values and values[0].is_file() and values[0].suffix.lower() == ".xlsx":
            return values[0], "xlsx", self._sha256(values[0])
        temporary = self.settings.structure_cache_dir / ".inputs" / f"{safe_name(sample.sample_id)}.xlsx"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        sample_to_xlsx(sample, temporary)
        source_payload = json.dumps(
            {"table_id": sample.table_id, "table_content": sample.table_content, "tables": sample.raw.get("tables")},
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

    def _key(self, source_hash: str, sheet_name: str) -> str:
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "source_sha256": source_hash,
            "sheet_name": sheet_name,
            "workflow_version": 4,
            "viewport_rows": self.settings.viewport_rows,
            "viewport_columns": self.settings.viewport_columns,
            "shift_cells": self.settings.shift_cells,
            "max_retry": self.settings.max_retry,
            "layout_model": self.settings.layout_model_identity,
            "layout_prompt_sha256": hashlib.sha256(
                (LAYOUT_MAS_SYSTEM_PROMPT + LAYOUT_MAS_USER_PROMPT_TEMPLATE).encode("utf-8")
            ).hexdigest(),
            "render_backend": self.settings.render_backend,
            "image_scale": self.settings.image_scale,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_record(directory: Path, key: str, sheet_name: str, *, cache_hit: bool) -> StructureCacheRecord | None:
        manifest_path = directory / "manifest.json"
        structure_path = directory / "structure.yaml"
        workbook_path = directory / "workbook.xlsx"
        if not manifest_path.is_file() or not workbook_path.is_file() or not structure_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("cache_key") != key or manifest.get("sheet_name") != sheet_name:
            return None
        return StructureCacheRecord(
            key, directory, workbook_path, sheet_name, structure_path, manifest_path,
            str(manifest.get("status", "not_good")), cache_hit,
        )

    @staticmethod
    def _lock(key: str):
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.Lock())
        return lock
