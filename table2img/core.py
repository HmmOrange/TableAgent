from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup


DEFAULT_SCALE = 2.0
DEFAULT_MAX_VIEWPORT_WIDTH = 16000
DEFAULT_MAX_VIEWPORT_HEIGHT = 12000


@dataclass(frozen=True)
class TableDocument:
    html: str
    title: str = ""
    source_format: str = "unknown"
    estimated_width: int = 1200
    estimated_height: int = 800


@dataclass(frozen=True)
class RenderResult:
    image_path: Path
    html_path: Path | None
    width: int | None
    height: int | None
    browser_path: Path


def table_to_image(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    input_format: str = "auto",
    json_index: int | None = None,
    table_index: int = 0,
    sheet: str | int | None = None,
    scale: float = DEFAULT_SCALE,
    browser_path: str | Path | None = None,
    keep_html: bool = True,
    timeout_seconds: float = 60.0,
) -> RenderResult:
    path = Path(input_path)
    if output_path is None:
        output = path.with_suffix(".png")
    else:
        output = Path(output_path)

    document = document_from_file(
        path,
        input_format=input_format,
        json_index=json_index,
        table_index=table_index,
        sheet=sheet,
    )
    return render_document(
        document,
        output,
        scale=scale,
        browser_path=browser_path,
        keep_html=keep_html,
        timeout_seconds=timeout_seconds,
    )


def document_from_file(
    input_path: str | Path,
    *,
    input_format: str = "auto",
    json_index: int | None = None,
    table_index: int = 0,
    sheet: str | int | None = None,
) -> TableDocument:
    path = Path(input_path)
    fmt = _detect_format(path, input_format)

    if fmt == "html":
        return document_from_html(path.read_text(encoding="utf-8"), source_format="html")
    if fmt == "md":
        rows = rows_from_markdown(path.read_text(encoding="utf-8"))
        return document_from_rows(rows, source_format="md")
    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return document_from_json(data, json_index=json_index, table_index=table_index)
    if fmt == "jsonl":
        data = _read_jsonl(path, json_index=json_index)
        return document_from_json(data, json_index=None, table_index=table_index)
    if fmt == "xlsx":
        return document_from_xlsx(path, sheet=sheet)
    if fmt == "csv":
        return document_from_rows(rows_from_delimited(path, delimiter=","), source_format="csv")
    if fmt == "tsv":
        return document_from_rows(rows_from_delimited(path, delimiter="\t"), source_format="tsv")

    text = path.read_text(encoding="utf-8")
    return document_from_text(text)


def document_from_text(text: str) -> TableDocument:
    stripped = text.lstrip()
    if "<table" in stripped.lower():
        return document_from_html(text, source_format="html")
    if stripped.startswith("{") or stripped.startswith("["):
        return document_from_json(json.loads(text))
    if "|" in text:
        return document_from_rows(rows_from_markdown(text), source_format="md")
    return document_from_rows(list(csv.reader(text.splitlines())), source_format="csv")


def document_from_html(content: str, *, source_format: str = "html") -> TableDocument:
    soup = BeautifulSoup(content, "html.parser")
    table = soup.find("table")
    if table is not None:
        title = _clean_text(table.find("caption").get_text(" ", strip=True)) if table.find("caption") else ""
        fragment = str(table)
        width, height = estimate_html_size(fragment)
        return TableDocument(
            html=wrap_html(fragment, title=title),
            title=title,
            source_format=source_format,
            estimated_width=width,
            estimated_height=height,
        )

    body = str(soup.body) if soup.body else content
    width, height = estimate_html_size(body)
    return TableDocument(
        html=wrap_html(body),
        source_format=source_format,
        estimated_width=width,
        estimated_height=height,
    )


def document_from_json(
    data: Any,
    *,
    json_index: int | None = None,
    table_index: int = 0,
) -> TableDocument:
    data = _select_json_record(data, json_index=json_index)

    if isinstance(data, dict) and "texts" in data:
        return document_from_hitab_json(data)

    if isinstance(data, dict) and isinstance(data.get("tables"), list):
        tables = data["tables"]
        if not tables:
            raise ValueError("JSON field 'tables' is empty.")
        selected = tables[_bounded_index(table_index, len(tables), "table_index")]
        if isinstance(selected, str):
            return document_from_html(selected, source_format="json:html-table")
        if isinstance(selected, list):
            return document_from_rows(selected, source_format="json:rows")
        if isinstance(selected, dict):
            return document_from_json(selected, json_index=None, table_index=table_index)
        raise ValueError(f"Unsupported table entry type in JSON 'tables': {type(selected).__name__}")

    if isinstance(data, dict):
        for key in ("rows", "data", "table", "values"):
            if key in data:
                value = data[key]
                if isinstance(value, str) and "<table" in value.lower():
                    return document_from_html(value, source_format=f"json:{key}")
                if isinstance(value, list):
                    return document_from_rows(value, source_format=f"json:{key}")
                if isinstance(value, dict):
                    return document_from_json(value, json_index=None, table_index=table_index)
        return document_from_rows([[key, value] for key, value in data.items()], source_format="json:object")

    if isinstance(data, list):
        return document_from_rows(data, source_format="json:rows")

    raise ValueError(f"Unsupported JSON table shape: {type(data).__name__}")


def document_from_hitab_json(data: dict[str, Any]) -> TableDocument:
    rows = _rectangularize(data.get("texts") or [])
    title = _clean_text(data.get("title", ""))
    merged_regions = data.get("merged_regions") or []
    fragment = rows_to_html(rows, title=title, merged_regions=merged_regions)
    width, height = estimate_rows_size(rows)
    return TableDocument(
        html=wrap_html(fragment, title=title),
        title=title,
        source_format="hitab-json",
        estimated_width=width,
        estimated_height=height,
    )


def document_from_xlsx(
    input_path: str | Path,
    *,
    sheet: str | int | None = None,
    add_coordinates: bool = True,
) -> TableDocument:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("XLSX input requires openpyxl. Install it or run with an environment that has it.") from exc

    workbook = openpyxl.load_workbook(input_path, data_only=True)
    if sheet is None:
        worksheet = workbook.active
    elif isinstance(sheet, int) or str(sheet).isdigit():
        worksheet = workbook.worksheets[int(sheet)]
    else:
        worksheet = workbook[str(sheet)]

    rows_iter = list(worksheet.iter_rows())
    if not rows_iter:
        rows = []
        merged_regions = []
    elif add_coordinates:
        from openpyxl.utils import get_column_letter

        first_row = rows_iter[0]
        start_row = first_row[0].row
        start_col = first_row[0].column

        col_letters = [get_column_letter(cell.column) for cell in first_row]
        new_rows = [[""] + col_letters]
        for row in rows_iter:
            row_num = str(row[0].row)
            row_values = [cell.value if cell.value is not None else "" for cell in row]
            new_rows.append([row_num] + row_values)
        rows = new_rows

        merged_regions = [
            {
                "first_row": merged.min_row - start_row + 1,
                "last_row": merged.max_row - start_row + 1,
                "first_column": merged.min_col - start_col + 1,
                "last_column": merged.max_col - start_col + 1,
            }
            for merged in worksheet.merged_cells.ranges
        ]
    else:
        rows = [
            [cell.value if cell.value is not None else "" for cell in row]
            for row in rows_iter
        ]
        merged_regions = [
            {
                "first_row": merged.min_row - 1,
                "last_row": merged.max_row - 1,
                "first_column": merged.min_col - 1,
                "last_column": merged.max_col - 1,
            }
            for merged in worksheet.merged_cells.ranges
        ]

    rows = _rectangularize(rows)
    fragment = rows_to_html(
        rows,
        title=worksheet.title,
        merged_regions=merged_regions,
        add_coordinates=add_coordinates,
    )
    width, height = estimate_rows_size(rows)
    return TableDocument(
        html=wrap_html(fragment, title=worksheet.title),
        title=worksheet.title,
        source_format="xlsx",
        estimated_width=width,
        estimated_height=height,
    )


def document_from_rows(rows: Iterable[Iterable[Any] | dict[str, Any]], *, source_format: str) -> TableDocument:
    normalized = _rows_from_any(rows)
    fragment = rows_to_html(normalized)
    width, height = estimate_rows_size(normalized)
    return TableDocument(
        html=wrap_html(fragment),
        source_format=source_format,
        estimated_width=width,
        estimated_height=height,
    )


def rows_from_delimited(input_path: str | Path, *, delimiter: str) -> list[list[str]]:
    with Path(input_path).open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.reader(f, delimiter=delimiter)]


def rows_from_markdown(content: str) -> list[list[str]]:
    table_lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            if table_lines:
                break
            continue
        if "|" in stripped:
            table_lines.append(stripped)
        elif table_lines:
            break

    rows = []
    for line in table_lines:
        cells = _split_markdown_row(line)
        if not cells:
            continue
        if _is_markdown_separator(cells):
            continue
        rows.append(cells)

    if not rows:
        raise ValueError("No markdown table found.")
    return rows


def rows_to_html(
    rows: list[list[Any]],
    *,
    title: str = "",
    merged_regions: list[dict[str, Any]] | None = None,
    add_coordinates: bool = False,
) -> str:
    rows = _rectangularize(rows)
    merge_map: dict[tuple[int, int], tuple[int, int]] = {}
    covered: set[tuple[int, int]] = set()

    for region in merged_regions or []:
        first_row = _safe_int(region.get("first_row"), 0)
        last_row = _safe_int(region.get("last_row"), first_row)
        first_col = _safe_int(region.get("first_column"), 0)
        last_col = _safe_int(region.get("last_column"), first_col)
        if first_row < 0 or first_col < 0 or first_row >= len(rows):
            continue
        if first_col >= len(rows[first_row]):
            continue
        last_row = min(last_row, len(rows) - 1)
        last_col = min(last_col, len(rows[first_row]) - 1)
        rowspan = max(1, last_row - first_row + 1)
        colspan = max(1, last_col - first_col + 1)
        if rowspan == 1 and colspan == 1:
            continue
        merge_map[(first_row, first_col)] = (rowspan, colspan)
        for row_index in range(first_row, first_row + rowspan):
            for col_index in range(first_col, first_col + colspan):
                if (row_index, col_index) != (first_row, first_col):
                    covered.add((row_index, col_index))

    lines = ["<table>"]
    if title:
        lines.append(f"<caption>{html.escape(title)}</caption>")

    for row_index, row in enumerate(rows):
        lines.append("<tr>")
        for col_index, cell in enumerate(row):
            if (row_index, col_index) in covered:
                continue
            rowspan, colspan = merge_map.get((row_index, col_index), (1, 1))
            attrs = []
            if rowspan > 1:
                attrs.append(f'rowspan="{rowspan}"')
            if colspan > 1:
                attrs.append(f'colspan="{colspan}"')
            if add_coordinates and (row_index == 0 or col_index == 0):
                attrs.append('class="excel-coord"')
            attr_text = " " + " ".join(attrs) if attrs else ""
            text = _nbsp_pad(_clean_text(cell))
            lines.append(f"<td{attr_text}>{html.escape(text)}</td>")
        lines.append("</tr>")

    lines.append("</table>")
    return "\n".join(lines)


def wrap_html(fragment: str, *, title: str = "") -> str:
    page_title = html.escape(title or "table2img")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{page_title}</title>
  <style>
    html, body {{
      background: #ffffff;
      margin: 0;
      padding: 0;
    }}
    body {{
      color: #111827;
      display: inline-block;
      font-family: Arial, "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
      font-size: 14px;
      line-height: 1.35;
      padding: 24px;
    }}
    table {{
      border-collapse: collapse;
      border-spacing: 0;
      width: max-content;
    }}
    caption {{
      caption-side: top;
      color: #111827;
      font-size: 16px;
      font-weight: 700;
      padding: 0 0 10px;
      text-align: left;
      white-space: normal;
    }}
    td, th {{
      border: 1px solid #1f2937;
      max-width: 420px;
      min-width: 48px;
      padding: 6px 9px;
      text-align: left;
      vertical-align: middle;
      white-space: pre-wrap;
      word-break: normal;
      overflow-wrap: anywhere;
    }}
    th {{
      font-weight: 700;
    }}
    .excel-coord {{
      background-color: #f3f4f6;
      color: #374151;
      font-weight: 700;
      text-align: center;
      border: 1px solid #9ca3af;
      font-size: 12px;
    }}
  </style>
</head>
<body>
{fragment}
</body>
</html>
"""


def render_document(
    document: TableDocument,
    output_path: str | Path,
    *,
    scale: float = DEFAULT_SCALE,
    browser_path: str | Path | None = None,
    keep_html: bool = True,
    timeout_seconds: float = 60.0,
    max_viewport_width: int = DEFAULT_MAX_VIEWPORT_WIDTH,
    max_viewport_height: int = DEFAULT_MAX_VIEWPORT_HEIGHT,
) -> RenderResult:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if keep_html:
        html_path = output.with_suffix(".html")
        html_path.write_text(document.html, encoding="utf-8")
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="table2img-")
        html_path = Path(temp_dir.name) / "table.html"
        html_path.write_text(document.html, encoding="utf-8")

    browser = find_browser(browser_path)
    width = min(max(320, document.estimated_width), max_viewport_width)
    height = min(max(240, document.estimated_height), max_viewport_height)
    _capture_with_chrome_cli(
        browser,
        html_path,
        output,
        width=width,
        height=height,
        scale=scale,
        timeout_seconds=timeout_seconds,
    )
    image_width, image_height = trim_image(output)

    result = RenderResult(
        image_path=output,
        html_path=html_path if keep_html else None,
        width=image_width,
        height=image_height,
        browser_path=browser,
    )
    if temp_dir is not None:
        temp_dir.cleanup()
    return result


def find_browser(browser_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if browser_path:
        candidates.append(Path(browser_path))

    env_browser = os.environ.get("TABLE2IMG_BROWSER")
    if env_browser:
        candidates.append(Path(env_browser))

    for name in ("chrome", "chrome.exe", "google-chrome", "chromium", "chromium-browser", "msedge", "msedge.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    for root in [Path(p) for p in program_files if p]:
        candidates.extend(
            [
                root / "Google" / "Chrome" / "Application" / "chrome.exe",
                root / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            ]
        )

    candidates.extend(
        [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/chromium"),
            Path("/usr/bin/chromium-browser"),
        ]
    )

    puppeteer_root = Path.home() / ".cache" / "puppeteer" / "chrome"
    if puppeteer_root.exists():
        candidates.extend(puppeteer_root.glob("**/chrome.exe"))
        candidates.extend(puppeteer_root.glob("**/chrome"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "No Chrome/Chromium/Edge browser found. Set TABLE2IMG_BROWSER or pass --browser."
    )


def estimate_rows_size(rows: list[list[Any]]) -> tuple[int, int]:
    rows = _rectangularize(rows)
    if not rows:
        return 640, 320

    col_count = max((len(row) for row in rows), default=1)
    col_widths = [64] * col_count
    row_heights = []

    for row in rows:
        max_lines = 1
        for col_index, cell in enumerate(row):
            text = _clean_text(cell)
            longest = max((len(part) for part in re.split(r"\s+", text) if part), default=0)
            char_count = min(max(len(text), longest), 48)
            col_widths[col_index] = max(col_widths[col_index], min(420, 28 + char_count * 8))
            max_lines = max(max_lines, max(1, (len(text) // 42) + 1))
        row_heights.append(28 + max_lines * 13)

    width = int(sum(col_widths) + 72)
    height = int(sum(row_heights) + 96)
    return width, height


def estimate_html_size(fragment: str) -> tuple[int, int]:
    soup = BeautifulSoup(fragment, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        rows.append([cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"], recursive=False)])
    if rows:
        return estimate_rows_size(rows)
    text_len = len(_clean_text(soup.get_text(" ", strip=True)))
    return min(16000, max(640, 16 * min(text_len, 600))), 800


def trim_image(image_path: str | Path, *, padding: int = 8) -> tuple[int | None, int | None]:
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return None, None

    path = Path(image_path)
    with Image.open(path) as img:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        diff = ImageChops.difference(img, background)
        bbox = diff.getbbox()
        if not bbox:
            return img.size

        left, top, right, bottom = bbox
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(img.width, right + padding)
        bottom = min(img.height, bottom + padding)
        cropped = img.crop((left, top, right, bottom))
        cropped.save(path)
        return cropped.size


def _capture_with_chrome_cli(
    browser: Path,
    html_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    scale: float,
    timeout_seconds: float,
) -> None:
    output_path = output_path.resolve()
    html_uri = html_path.resolve().as_uri()
    base_args = [
        str(browser),
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-sandbox",
        f"--force-device-scale-factor={scale:g}",
        f"--window-size={width},{height}",
        f"--screenshot={output_path}",
        html_uri,
    ]

    last_error = None
    for headless_arg in ("--headless=new", "--headless"):
        args = [base_args[0], headless_arg, *base_args[1:]]
        completed = subprocess.run(
            args,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        if completed.returncode == 0 and output_path.is_file():
            return
        last_error = completed

    stderr = (last_error.stderr if last_error else "").strip()
    stdout = (last_error.stdout if last_error else "").strip()
    raise RuntimeError(f"Chrome screenshot failed. stdout={stdout!r} stderr={stderr!r}")


def _detect_format(path: Path, input_format: str) -> str:
    fmt = input_format.lower().strip()
    if fmt != "auto":
        aliases = {
            "markdown": "md",
            "htm": "html",
            "xls": "xlsx",
            "text": "txt",
        }
        return aliases.get(fmt, fmt)

    suffix = path.suffix.lower().lstrip(".")
    aliases = {
        "htm": "html",
        "markdown": "md",
        "xls": "xlsx",
        "txt": "txt",
    }
    return aliases.get(suffix, suffix or "txt")


def _read_jsonl(path: Path, *, json_index: int | None) -> Any:
    target_index = 0 if json_index is None else json_index
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            if index == target_index:
                return json.loads(line)
    raise IndexError(f"JSONL index out of range: {target_index}")


def _select_json_record(data: Any, *, json_index: int | None) -> Any:
    if json_index is None:
        if isinstance(data, list) and data and not _looks_like_rows(data):
            for item in data:
                if isinstance(item, dict) and ("tables" in item or "texts" in item):
                    return item
        return data

    if not isinstance(data, list):
        if json_index == 0:
            return data
        raise TypeError("--json-index can only select from top-level JSON lists.")
    return data[_bounded_index(json_index, len(data), "json_index")]


def _looks_like_rows(data: list[Any]) -> bool:
    return all(isinstance(row, (list, tuple, dict)) for row in data) and not any(
        isinstance(row, dict) and ("tables" in row or "texts" in row)
        for row in data
    )


def _rows_from_any(rows: Iterable[Iterable[Any] | dict[str, Any]]) -> list[list[Any]]:
    materialized = list(rows)
    if not materialized:
        return []

    if all(isinstance(row, dict) for row in materialized):
        keys = []
        for row in materialized:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        return [keys] + [[row.get(key, "") for key in keys] for row in materialized]

    normalized = []
    for row in materialized:
        if isinstance(row, dict):
            normalized.append([f"{key}: {value}" for key, value in row.items()])
        elif isinstance(row, (list, tuple)):
            normalized.append(list(row))
        else:
            normalized.append([row])
    return _rectangularize(normalized)


def _rectangularize(rows: Iterable[Iterable[Any]]) -> list[list[Any]]:
    normalized = [list(row) for row in rows]
    width = max((len(row) for row in normalized), default=0)
    return [row + [""] * (width - len(row)) for row in normalized]


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_clean_text(cell.replace("\\|", "|")) for cell in stripped.split("|")]


def _is_markdown_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells if cell.strip())


def _clean_text(value: Any) -> str:
    return " ".join(str(value if value is not None else "").replace("\xa0", " ").split())


def _nbsp_pad(value: str) -> str:
    return f"\xa0{value}\xa0" if value else "\xa0"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bounded_index(index: int, length: int, name: str) -> int:
    if index < 0 or index >= length:
        raise IndexError(f"{name} out of range: {index}; available range is 0..{length - 1}")
    return index


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed
