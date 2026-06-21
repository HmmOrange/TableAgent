from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from datasets.base import EvalSample
from table2img.core import RenderResult, document_from_xlsx
from utils.workbook_converter import sample_to_xlsx

from TableAgent.config import TableAgentConfig
from TableAgent.rendering.image_utils import (
    _generate_image_tiles,
    _resize_image_file_to_fit,
    compute_viewport_and_scale,
)


class WorkbookRenderer:
    def __init__(self, settings: TableAgentConfig, renderer: Callable[..., RenderResult], logger: Any):
        self.settings = settings
        self.renderer = renderer
        self.logger = logger

    def sample_to_image(self, sample: EvalSample, sample_dir: Path):
        workbook_path = sample_dir / "table.xlsx"
        workbook = sample_to_xlsx(sample, workbook_path)
        document = document_from_xlsx(workbook_path)
        render_result = self.render_document(document, sample_dir / "table.png")
        tiles = self.postprocess_image(render_result.image_path)
        if tiles:
            (sample_dir / "metadata.json").write_text(json.dumps({"image_tiles": tiles}), encoding="utf-8")
        return workbook, render_result

    def source_to_image(self, source_path: Path, sheet_name: str, image_path: Path, html_path: Path) -> list[dict[str, Any]]:
        if not image_path.is_file() or not html_path.is_file():
            document = document_from_xlsx(source_path, sheet=sheet_name, add_coordinates=True)
            self.render_document(document, image_path)
        return self.postprocess_image(image_path)

    def render_document(self, document: Any, image_path: Path) -> RenderResult:
        _, _, scale = compute_viewport_and_scale(
            estimated_width=document.estimated_width,
            estimated_height=document.estimated_height,
            image_scale=self.settings.image_scale,
            max_viewport_width=self.settings.max_viewport_width,
            max_viewport_height=self.settings.max_viewport_height,
            max_image_dimension=self.settings.max_image_dimension,
            max_image_pixels=self.settings.max_image_pixels,
        )
        return self.renderer(
            document,
            image_path,
            scale=scale,
            backend=self.settings.render_backend,
            keep_html=True,
            timeout_seconds=self.settings.render_timeout_seconds,
            max_viewport_width=self.settings.max_viewport_width,
            max_viewport_height=self.settings.max_viewport_height,
        )

    def postprocess_image(self, image_path: Path) -> list[dict[str, Any]]:
        tiles = []
        if self.settings.image_tile_size is not None:
            tiles = _generate_image_tiles(
                image_path,
                self.settings.image_tile_size,
                self.settings.image_tile_overlap,
                logger=self.logger,
            )
        _resize_image_file_to_fit(
            image_path,
            self.settings.max_image_dimension,
            self.settings.max_image_pixels,
            logger=self.logger,
        )
        return tiles
