from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from datasets.base import EvalSample
from TableAgent.config import TableAgentConfig
from TableAgent.perception.structure import _is_valid_structure
from TableAgent.pipeline.common import is_siflex, safe_name
from TableAgent.rendering.workbook import WorkbookRenderer


class SourcePreparer:
    def __init__(
        self,
        settings: TableAgentConfig,
        renderer: WorkbookRenderer,
        build_structure: Callable[[str, Path, Any | None], str],
    ):
        self.settings = settings
        self.renderer = renderer
        self.build_structure = build_structure

    def prepare(self, samples: list[EvalSample], logger: Any | None = None) -> None:
        if not samples or not is_siflex(samples[0]):
            return

        for source_path, sheet_name in self._source_sheets(samples, logger=logger):
            sheet_dir = self.source_dir(source_path, sheet_name)
            sheet_dir.mkdir(parents=True, exist_ok=True)
            paths = self._paths(sheet_dir)
            try:
                tiles = self.renderer.source_to_image(source_path, sheet_name, paths["image"], paths["html"])
                sheet_text = self._write_sheet_text(source_path, sheet_name, paths["text"])
                self._write_metadata(paths["metadata"], source_path, sheet_name, tiles)
            except Exception as exc:
                if logger:
                    logger.error("TableAgent source prep failed for %s:%s: %s", source_path, sheet_name, exc)
                continue

            if self._has_valid_structure(paths["structure"]):
                continue
            if paths["error"].is_file():
                continue

            structure_text = self.build_structure(sheet_text, paths["image"], logger)
            if _is_valid_structure(structure_text):
                paths["error"].unlink(missing_ok=True)
                paths["structure"].write_text(structure_text, encoding="utf-8")
            else:
                paths["structure"].unlink(missing_ok=True)
                paths["error"].write_text(
                    f"Failed to generate structure (empty/invalid/token-capped). Raw: {structure_text}",
                    encoding="utf-8",
                )

    def source_dir(self, source_path: Path, sheet_name: str) -> Path:
        artifact_dir = self.settings.source_artifact_dir or self.settings.artifact_dir
        return artifact_dir / "sources" / f"{safe_name(source_path.name)}_{safe_name(sheet_name)}"

    def _source_sheets(self, samples: list[EvalSample], logger: Any | None = None) -> list[tuple[Path, str]]:
        sources: set[Path] = set()
        for sample in samples:
            for value in str(sample.table_path or "").split(";"):
                if value.strip():
                    sources.add(Path(value.strip()))

        pairs: list[tuple[Path, str]] = []
        for source_path in sorted(sources):
            if not source_path.is_file() or source_path.suffix.lower() != ".xlsx":
                continue
            try:
                import openpyxl

                workbook = openpyxl.load_workbook(source_path, read_only=True)
                pairs.extend((source_path, sheet_name) for sheet_name in workbook.sheetnames)
                workbook.close()
            except Exception as exc:
                if logger:
                    logger.error("Failed to load workbook %s: %s", source_path, exc)
        return pairs

    @staticmethod
    def _paths(sheet_dir: Path) -> dict[str, Path]:
        return {
            "image": sheet_dir / "table.png",
            "html": sheet_dir / "table.html",
            "text": sheet_dir / "sheet_text.txt",
            "metadata": sheet_dir / "metadata.json",
            "structure": sheet_dir / "structure.yaml",
            "error": sheet_dir / "structure.error",
        }

    @staticmethod
    def _has_valid_structure(structure_path: Path) -> bool:
        if not structure_path.is_file():
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

        workbook = openpyxl.load_workbook(source_path, data_only=True)
        worksheet = workbook[sheet_name]
        values = [str(value) for row in worksheet.iter_rows(values_only=True) for value in row if value is not None]
        workbook.close()
        sheet_text = " ".join(values)
        output_path.write_text(sheet_text, encoding="utf-8")
        return sheet_text

    @staticmethod
    def _write_metadata(
        metadata_path: Path,
        source_path: Path,
        sheet_name: str,
        image_tiles: list[dict[str, Any]],
    ) -> None:
        metadata = {
            "workbook_path": str(source_path.resolve()),
            "sheet_name": sheet_name,
            "safe_filename": safe_name(source_path.name),
            "safe_sheetname": safe_name(sheet_name),
        }
        if image_tiles:
            metadata["image_tiles"] = image_tiles
        if not metadata_path.is_file() or image_tiles:
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
