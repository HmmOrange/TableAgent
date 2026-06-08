from __future__ import annotations

import argparse
from pathlib import Path

from .core import DEFAULT_SCALE, positive_float, table_to_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="table2img",
        description="Render HTML, Markdown, JSON, CSV/TSV, or XLSX tables to high-resolution PNG screenshots.",
    )
    parser.add_argument("input", type=Path, help="Input table file.")
    parser.add_argument("-o", "--output", type=Path, help="PNG output path. Defaults to input stem with .png.")
    parser.add_argument(
        "--format",
        default="auto",
        help="Input format override: auto, html, md, json, jsonl, csv, tsv, xlsx.",
    )
    parser.add_argument(
        "--json-index",
        type=int,
        default=None,
        help="Select an item from a top-level JSON/JSONL list. Defaults to the first table-like record.",
    )
    parser.add_argument(
        "--table-index",
        type=int,
        default=0,
        help="Select from JSON fields like tables=[...]. Defaults to 0.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="XLSX sheet name or zero-based sheet index. Defaults to the active sheet.",
    )
    parser.add_argument(
        "--scale",
        type=positive_float,
        default=DEFAULT_SCALE,
        help=f"Browser device scale factor for high-resolution output. Defaults to {DEFAULT_SCALE:g}.",
    )
    parser.add_argument(
        "--browser",
        type=Path,
        default=None,
        help="Chrome, Chromium, or Edge executable. Defaults to auto-detect or TABLE2IMG_BROWSER.",
    )
    parser.add_argument(
        "--no-keep-html",
        action="store_true",
        help="Do not keep the intermediate HTML next to the PNG.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=60.0,
        help="Browser timeout in seconds.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = table_to_image(
        args.input,
        args.output,
        input_format=args.format,
        json_index=args.json_index,
        table_index=args.table_index,
        sheet=args.sheet,
        scale=args.scale,
        browser_path=args.browser,
        keep_html=not args.no_keep_html,
        timeout_seconds=args.timeout,
    )
    size = f"{result.width}x{result.height}" if result.width and result.height else "unknown size"
    print(f"image: {result.image_path.resolve()} ({size})")
    if result.html_path is not None:
        print(f"html:  {result.html_path.resolve()}")
    print(f"browser: {result.browser_path}")
    return 0
