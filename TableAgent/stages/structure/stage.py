from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from TableAgent.artifacts.layout import workbook_artifact_dir

from .contracts import StructureInput, StructureOutput

if TYPE_CHECKING:
    from .cache import StructureCacheRecord
from .retrieval_artifacts import write_workbook_retrieval_cards


class StructureStage:
    """Compatibility boundary while structure internals move out of the pipeline."""

    def __init__(self, run_structure: Callable[..., list["StructureCacheRecord"]]):
        self._run_structure = run_structure

    def run(self, stage_input: StructureInput) -> StructureOutput:
        records = self._run_structure(
            list(stage_input.samples),
            force=stage_input.force,
        )
        return StructureOutput(records=tuple(records))

    @staticmethod
    def finalize_retrieval_artifacts(
        source_dir: Path,
        workbooks: list[tuple[str, str]],
        *,
        selected_sheets: tuple[str, ...] = (),
        include_embeddings: bool = False,
        embedding_client: Any | None = None,
        embedding_model: str = "",
    ) -> None:
        """Aggregate prepared sheet/table cards into one workbook-level corpus."""
        selected = set(selected_sheets)
        for workbook_name, workbook_sha256 in workbooks:
            workbook_dir = workbook_artifact_dir(
                source_dir,
                workbook_name,
                workbook_sha256,
            )
            records: list[dict[str, Any]] = []
            if workbook_dir.is_dir():
                for jsonl_path in sorted(
                    workbook_dir.glob("*/retrieval_cards.jsonl")
                ):
                    try:
                        payload = [
                            json.loads(line)
                            for line in jsonl_path.read_text(
                                encoding="utf-8"
                            ).splitlines()
                            if line.strip()
                        ]
                    except (OSError, json.JSONDecodeError):
                        continue
                    records.extend(
                        record
                        for record in payload
                        if isinstance(record, dict)
                        and (
                            not selected
                            or str(record.get("sheet") or "") in selected
                        )
                    )
            if records:
                write_workbook_retrieval_cards(
                    workbook_dir,
                    workbook_name,
                    records,
                    include_embeddings=include_embeddings,
                    embedding_client=embedding_client,
                    embedding_model=embedding_model,
                )
