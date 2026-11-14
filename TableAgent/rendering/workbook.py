from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from TableAgent.settings import QASettings


_PDFIUM_LOCK = threading.Lock()


@dataclass(frozen=True)
class RenderResult:
    image_path: Path
    html_path: Path | None
    width: int | None
    height: int | None
    browser_path: Path


class WorkbookRenderer:
    """Renders a cell range of a workbook to a PNG with LibreOffice, for the vision model to look at."""

    def __init__(self, settings: QASettings):
        self.settings = settings

    def render_workbook_range(
        self,
        workbook_path: Path,
        sheet_name: str | None,
        cell_range: str | None,
        image_path: Path,
    ) -> RenderResult:
        _render_xlsx_range_with_libreoffice(
            workbook_path,
            sheet_name,
            cell_range,
            image_path,
            libreoffice_path=self.settings.libreoffice_path,
            resolution=self.settings.image_resolution,
            timeout_seconds=self.settings.render_timeout_seconds,
            show_coordinates=self.settings.show_coordinates,
            max_workers=self.settings.max_workers,
            max_cell_width=self.settings.max_cell_width,
        )
        result = _image_render_result(image_path, browser_path=Path("libreoffice"))
        _write_render_metadata(
            workbook_path=workbook_path,
            sheet_name=sheet_name,
            cell_range=cell_range,
            image_path=image_path,
            result=result,
            show_coordinates=self.settings.show_coordinates,
        )
        return result


def _render_xlsx_range_with_libreoffice(
    workbook_path: Path,
    sheet_name: str | None,
    cell_range: str | None,
    image_path: Path,
    *,
    libreoffice_path: Path | str | None = None,
    resolution: int = 192,
    timeout_seconds: float = 60,
    show_coordinates: bool = True,
    max_workers: int = 1,
    max_cell_width: int = 50,
) -> None:
    try:
        import openpyxl
        from openpyxl.worksheet.dimensions import ColumnDimension, RowDimension
    except ImportError as exc:
        raise RuntimeError(
            "LibreOffice workbook rendering requires openpyxl and pypdfium2. "
            "Install it before running TableAgent workbook image rendering."
        ) from exc

    image_path.parent.mkdir(parents=True, exist_ok=True)
    soffice = _resolve_libreoffice_path(libreoffice_path)
    with tempfile.TemporaryDirectory(prefix="tableagent_lo_") as tmp:
        tmpdir = Path(tmp)
        prepared_path = tmpdir / workbook_path.name
        pdf_path = prepared_path.with_suffix(".pdf")
        profile_dir = tmpdir / "lo_profile"
        workbook = openpyxl.load_workbook(workbook_path)
        try:
            worksheet = workbook.worksheets[0] if sheet_name is None else workbook[sheet_name]
            workbook.active = workbook.worksheets.index(worksheet)
            for sheet in workbook.worksheets:
                sheet.sheet_state = "visible" if sheet is worksheet else "hidden"
            if cell_range:
                worksheet.print_area = cell_range
                _clip_merges(worksheet, cell_range)
            worksheet.print_options.headings = bool(show_coordinates)
            worksheet.print_options.gridLines = True
            worksheet.page_margins.left = 0
            worksheet.page_margins.right = 0
            worksheet.page_margins.top = 0
            worksheet.page_margins.bottom = 0
            worksheet.sheet_properties.pageSetUpPr.fitToPage = True
            worksheet.page_setup.fitToWidth = 1
            worksheet.page_setup.fitToHeight = 1
            _ensure_visible_dimensions(worksheet, RowDimension, ColumnDimension)
            _expand_dimensions_to_fit(worksheet, cell_range, max_cell_width=max_cell_width)
            workbook.save(prepared_path)
        finally:
            workbook.close()

        command = [
            str(soffice),
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--nodefault",
            "--nolockcheck",
            f"-env:UserInstallation={_libreoffice_file_url(profile_dir)}",
            "--convert-to",
            "pdf:calc_pdf_Export",
            "--outdir",
            str(tmpdir),
            str(prepared_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(1, float(timeout_seconds)),
        )
        if completed.returncode != 0 or not pdf_path.is_file():
            raise RuntimeError(
                "LibreOffice failed to export workbook to PDF: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        render_resolution = max(384 if cell_range else 96, int(resolution))
        if int(max_workers) > 1:
            _render_pdf_page_in_subprocess(
                pdf_path,
                image_path,
                render_resolution,
                timeout_seconds=timeout_seconds,
            )
        else:
            # PDFium APIs are process-global and not thread-safe.
            with _PDFIUM_LOCK:
                from TableAgent.rendering.pdfium_worker import render_pdf_page

                render_pdf_page(pdf_path, image_path, render_resolution)
        _trim_image_file_to_content(image_path)


def _render_pdf_page_in_subprocess(
    pdf_path: Path,
    image_path: Path,
    resolution: int,
    *,
    timeout_seconds: float,
) -> None:
    # Invoke the worker by file path so it does not depend on the parent process's
    # in-memory sys.path setup when TableAgent is embedded as a submodule.
    worker_path = Path(__file__).with_name("pdfium_worker.py").resolve()
    command = [
        sys.executable,
        str(worker_path),
        str(pdf_path),
        str(image_path),
        str(resolution),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1, float(timeout_seconds)),
    )
    if completed.returncode != 0 or not image_path.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "PDFium worker failed to render the LibreOffice PDF"
            + (f": {detail}" if detail else "")
        )


def _resolve_libreoffice_path(libreoffice_path: Path | str | None) -> Path:
    candidates = [
        str(libreoffice_path or "").strip(),
        shutil.which("soffice") or "",
        shutil.which("libreoffice") or "",
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        "LibreOffice executable not found. Set table_agent.libreoffice_path "
        "or add soffice/libreoffice to PATH."
    )


def _libreoffice_file_url(path: Path) -> str:
    return path.resolve().as_uri()


def _ensure_visible_dimensions(worksheet: Any, row_dimension_cls: Any, column_dimension_cls: Any) -> None:
    from openpyxl.utils import get_column_letter

    for row_index in range(1, worksheet.max_row + 1):
        if row_index not in worksheet.row_dimensions:
            worksheet.row_dimensions[row_index] = row_dimension_cls(worksheet, index=row_index)
    for column_index in range(1, worksheet.max_column + 1):
        column_letter = get_column_letter(column_index)
        if column_letter not in worksheet.column_dimensions:
            worksheet.column_dimensions[column_letter] = column_dimension_cls(worksheet, index=column_letter)


def _clip_merges(worksheet: Any, cell_range: str) -> None:
    """Shrink each merged cell cut by the range edge to its part inside, keeping its text and style, wrapped."""
    from copy import copy

    from openpyxl.utils.cell import range_boundaries

    min_col, min_row, max_col, max_row = range_boundaries(cell_range)
    for merged in list(worksheet.merged_cells.ranges):
        if merged.max_row < min_row or merged.min_row > max_row or merged.max_col < min_col or merged.min_col > max_col:
            continue
        if merged.min_row >= min_row and merged.max_row <= max_row and merged.min_col >= min_col and merged.max_col <= max_col:
            continue
        source = worksheet.cell(merged.min_row, merged.min_col)
        value, style = source.value, copy(source._style)
        worksheet.unmerge_cells(merged.coord)
        top, left = max(merged.min_row, min_row), max(merged.min_col, min_col)
        bottom, right = min(merged.max_row, max_row), min(merged.max_col, max_col)
        target = worksheet.cell(top, left)
        target.value, target._style = value, style
        alignment = copy(target.alignment)
        alignment.wrap_text = True
        target.alignment = alignment
        if (top, left) != (bottom, right):
            worksheet.merge_cells(start_row=top, start_column=left, end_row=bottom, end_column=right)


def _expand_dimensions_to_fit(
    worksheet: Any,
    cell_range: str | None = None,
    *,
    max_cell_width: int = 50,
) -> None:
    """Expand worksheet dimensions so cell text is present in the exported image."""
    from copy import copy

    from openpyxl.utils import get_column_letter
    from openpyxl.utils.cell import range_boundaries

    if max_cell_width < 1:
        raise ValueError("max_cell_width must be a positive integer")

    if cell_range:
        min_column, min_row, max_column, max_row = range_boundaries(cell_range)
    else:
        min_column, min_row = 1, 1
        max_column, max_row = worksheet.max_column, worksheet.max_row

    required_widths: dict[int, float] = {}
    merged_by_cell: dict[tuple[int, int], Any] = {}
    for merged_range in worksheet.merged_cells.ranges:
        for row in range(max(min_row, merged_range.min_row), min(max_row, merged_range.max_row) + 1):
            for column in range(
                max(min_column, merged_range.min_col),
                min(max_column, merged_range.max_col) + 1,
            ):
                merged_by_cell[(row, column)] = merged_range

    viewport_rows = worksheet.iter_rows(
        min_row=min_row,
        max_row=max_row,
        min_col=min_column,
        max_col=max_column,
    )
    for row in viewport_rows:
        for cell in row:
            if cell.value is None or not str(cell.value).strip():
                continue
            lines = str(cell.value).splitlines() or [""]
            required = max(_display_width(line) for line in lines) + 2
            merged_range = merged_by_cell.get((cell.row, cell.column))
            if merged_range is None:
                required_widths[cell.column] = max(required_widths.get(cell.column, 0), required)
                continue
            current = sum(
                float(worksheet.column_dimensions[get_column_letter(column)].width or 13)
                for column in range(merged_range.min_col, merged_range.max_col + 1)
            )
            # The extra width is shared by the merged columns, so they stay balanced.
            extra = max(0, required - current) / (merged_range.max_col - merged_range.min_col + 1)
            for column in range(merged_range.min_col, merged_range.max_col + 1):
                required_widths[column] = max(
                    required_widths.get(column, 0),
                    float(worksheet.column_dimensions[get_column_letter(column)].width or 13) + extra,
                )

    for column, required in required_widths.items():
        letter = get_column_letter(column)
        dimension = worksheet.column_dimensions[letter]
        dimension.width = min(max(float(dimension.width or 13), required), max_cell_width)

    viewport_rows = worksheet.iter_rows(
        min_row=min_row,
        max_row=max_row,
        min_col=min_column,
        max_col=max_column,
    )
    for row in viewport_rows:
        for cell in row:
            if cell.value is None or not str(cell.value).strip():
                continue
            lines = str(cell.value).splitlines() or [""]
            merged_range = merged_by_cell.get((cell.row, cell.column))
            columns = (
                range(merged_range.min_col, merged_range.max_col + 1)
                if merged_range is not None
                else range(cell.column, cell.column + 1)
            )
            available_width = sum(
                float(worksheet.column_dimensions[get_column_letter(column)].width or 13)
                for column in columns
            )
            wrap = bool(cell.alignment.wrap_text) or len(lines) > 1 or any(
                _display_width(line) > available_width for line in lines
            )
            if wrap and not cell.alignment.wrap_text:
                alignment = copy(cell.alignment)
                alignment.wrap_text = True
                cell.alignment = alignment
            wrapped_lines = sum(
                max(1, math.ceil(_display_width(line) / max(1, available_width)))
                for line in lines
            ) if wrap else len(lines)
            if wrapped_lines <= 1:
                continue
            font_size = float(cell.font.sz or 11)
            required_height = wrapped_lines * font_size * 1.25 + 2
            dimension = worksheet.row_dimensions[cell.row]
            dimension.height = max(float(dimension.height or 15), required_height)


def _display_width(value: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in value)


def _image_render_result(image_path: Path, *, browser_path: Path) -> RenderResult:
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    return RenderResult(
        image_path=image_path,
        html_path=None,
        width=width,
        height=height,
        browser_path=browser_path,
    )


def _trim_image_file_to_content(image_path: Path, *, padding: int = 6) -> None:
    from PIL import Image, ImageChops

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        background = Image.new("RGB", rgb.size, "white")
        bbox = ImageChops.difference(rgb, background).getbbox()
        if bbox is None:
            return
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(rgb.width, bbox[2] + padding)
        bottom = min(rgb.height, bbox[3] + padding)
        rgb.crop((left, top, right, bottom)).save(image_path)


def _write_render_metadata(
    *,
    workbook_path: Path,
    sheet_name: str | None,
    cell_range: str | None,
    image_path: Path,
    result: RenderResult,
    show_coordinates: bool,
) -> None:
    from openpyxl.utils.cell import range_boundaries

    payload: dict[str, Any] = {
        "workbook_path": str(workbook_path.resolve()),
        "sheet_name": sheet_name,
        "cell_range": cell_range,
        "image_width": result.width,
        "image_height": result.height,
        "renderer": "libreoffice",
        "show_coordinates": show_coordinates,
    }
    if cell_range:
        min_col, min_row, max_col, max_row = range_boundaries(cell_range)
        payload["bounds"] = {
            "min_row": min_row,
            "max_row": max_row,
            "min_col": min_col,
            "max_col": max_col,
        }
    image_path.with_suffix(".metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
