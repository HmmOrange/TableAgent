from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from datasets.base import EvalSample

from TableAgent.configs import TableAgentConfig
from TableAgent.perception.metadata import ExStructMetadataExtractor, SheetMetadata
from TableAgent.structure.layout.parsing import _is_valid_structure
from TableAgent.pipeline.common import is_siflex, safe_name

LAYOUT_WORKFLOW_VERSION = 4


class SourcePreparer:
    def __init__(
        self,
        settings: TableAgentConfig,
        analyze_sheet: Callable[[Path, str, SheetMetadata, Path], str],
        metadata_extractor: ExStructMetadataExtractor | None = None,
        progress_callback: Callable[..., None] | None = None,
    ):
        self.settings = settings
        self.analyze_sheet = analyze_sheet
        self.metadata_extractor = metadata_extractor or ExStructMetadataExtractor(settings.exstruct_mode)
        self.progress_callback = progress_callback

    def _progress(self, stage: str, **fields: Any) -> None:
        if self.progress_callback:
            self.progress_callback(stage, **fields)

    def prepare(self, samples: list[EvalSample], logger: Any | None = None, *, regenerate_invalid: bool = True) -> None:
        if not samples or not is_siflex(samples[0]):
            return

        for source_path in self._source_paths(samples):
            try:
                self._progress("prepare_extract", workbook=source_path.name)
                workbook_payload = self.metadata_extractor.extract(source_path)
            except Exception as exc:
                if logger:
                    logger.error("ExStruct extraction failed for %s: %s", source_path, exc)
                self._progress("prepare_error", workbook=source_path.name, sheet="<unreadable>")
                continue

            sheets = workbook_payload.get("sheets") or {}
            for sheet_name in sheets:
                self._progress("prepare_metadata", workbook=source_path.name, sheet=sheet_name)
                sheet_dir = self.source_dir(source_path, sheet_name)
                sheet_dir.mkdir(parents=True, exist_ok=True)
                paths = self._paths(sheet_dir)
                structure_exists = paths["structure"].is_file()
                has_current_structure = (
                    self._has_current_valid_structure(paths["structure"], paths["metadata_json"])
                    if regenerate_invalid or not structure_exists
                    else False
                )
                try:
                    metadata = self.metadata_extractor.sheet_metadata(
                        source_path,
                        workbook_payload,
                        sheet_name,
                    )
                    paths["metadata_yaml"].write_text(metadata.to_yaml(), encoding="utf-8")
                    self._write_metadata_json(paths["metadata_json"], source_path, sheet_name, metadata)
                    self._write_sheet_text(source_path, sheet_name, paths["text"])
                except Exception as exc:
                    if logger:
                        logger.error("TableAgent metadata preparation failed for %s:%s: %s", source_path, sheet_name, exc)
                    self._progress("prepare_error", workbook=source_path.name, sheet=sheet_name)
                    continue

                if has_current_structure:
                    self._progress("prepare_cached", workbook=source_path.name, sheet=sheet_name)
                    continue
                if structure_exists and not regenerate_invalid:
                    self._progress("prepare_cached", workbook=source_path.name, sheet=sheet_name)
                    continue
                if paths["error"].is_file():
                    self._progress("prepare_error", workbook=source_path.name, sheet=sheet_name)
                    continue

                try:
                    self._progress("prepare_layout", workbook=source_path.name, sheet=sheet_name, range=metadata.used_range)
                    structure_text = self.analyze_sheet(source_path, sheet_name, metadata, sheet_dir)
                except Exception as exc:
                    structure_text = ""
                    if logger:
                        logger.error("TableAgent layout workflow failed for %s:%s: %s", source_path, sheet_name, exc)
                if _is_valid_structure(structure_text):
                    paths["error"].unlink(missing_ok=True)
                    paths["structure"].write_text(structure_text, encoding="utf-8")
                else:
                    paths["structure"].unlink(missing_ok=True)
                    paths["error"].write_text(
                        f"Failed to generate structure (empty/invalid/token-capped). Raw: {structure_text}",
                        encoding="utf-8",
                    )
                self._progress("prepare_done", workbook=source_path.name, sheet=sheet_name)

    def source_dir(self, source_path: Path, sheet_name: str) -> Path:
        artifact_dir = self.settings.source_artifact_dir or self.settings.artifact_dir
        return artifact_dir / "sources" / f"{safe_name(source_path.name)}_{safe_name(sheet_name)}"

    @staticmethod
    def _source_paths(samples: list[EvalSample]) -> list[Path]:
        sources: set[Path] = set()
        for sample in samples:
            for value in str(sample.table_path or "").split(";"):
                path = Path(value.strip()) if value.strip() else None
                if path and path.is_file() and path.suffix.lower() == ".xlsx":
                    sources.add(path)
        return sorted(sources)

    @staticmethod
    def _paths(sheet_dir: Path) -> dict[str, Path]:
        return {
            "text": sheet_dir / "sheet_text.txt",
            "metadata_yaml": sheet_dir / "metadata.yaml",
            "metadata_json": sheet_dir / "metadata.json",
            "structure": sheet_dir / "structure.yaml",
            "error": sheet_dir / "structure.error",
        }

    @staticmethod
    def _has_current_valid_structure(structure_path: Path, metadata_path: Path) -> bool:
        if not structure_path.is_file() or not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if metadata.get("layout_workflow_version") != LAYOUT_WORKFLOW_VERSION:
            return False
        cached = structure_path.read_text(encoding="utf-8")
        if _is_valid_structure(cached):
            return True
        structure_path.unlink(missing_ok=True)
        return False

    @staticmethod
    def _write_sheet_text(source_path: Path, sheet_name: str, output_path: Path) -> str:
        if output_path.is_file():
            return output_path.read_text(encoding="utf-8")

        import openpyxl

        workbook = openpyxl.load_workbook(source_path, data_only=True, read_only=True)
        try:
            worksheet = workbook[sheet_name]
            values = [
                str(value)
                for row in worksheet.iter_rows(values_only=True)
                for value in row
                if value is not None
            ]
        finally:
            workbook.close()
        sheet_text = " ".join(values)
        output_path.write_text(sheet_text, encoding="utf-8")
        return sheet_text

    @staticmethod
    def _write_metadata_json(
        metadata_path: Path,
        source_path: Path,
        sheet_name: str,
        metadata: SheetMetadata,
    ) -> None:
        payload = {
            "workbook_path": str(source_path.resolve()),
            "sheet_name": sheet_name,
            "safe_filename": safe_name(source_path.name),
            "safe_sheetname": safe_name(sheet_name),
            "layout_workflow_version": LAYOUT_WORKFLOW_VERSION,
            "used_range": metadata.used_range,
            "merged_ranges": metadata.merged_ranges,
        }
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
