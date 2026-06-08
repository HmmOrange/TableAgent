from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import document_from_file, document_from_json, positive_float, render_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a few SemTab HiTab and MulHi sample tables to PNG.")
    parser.add_argument(
        "--semitab-root",
        type=Path,
        default=Path("semitab"),
        help="Path to the semitab folder. Defaults to ./semitab.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("table2img") / "outputs" / "smoke",
        help="Directory for smoke PNG/HTML files.",
    )
    parser.add_argument("--scale", type=positive_float, default=2.0, help="Browser device scale factor.")
    parser.add_argument("--browser", type=Path, default=None, help="Chrome, Chromium, or Edge executable.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    semitab_root = args.semitab_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for relative in (
        Path("data") / "HiTab" / "tables" / "raw" / "100.json",
        Path("data") / "HiTab" / "tables" / "raw" / "9_totto1028-1.json",
    ):
        source = semitab_root / relative
        document = document_from_file(source)
        jobs.append((f"hitab_{source.stem}", document))

    mulhi_path = semitab_root / "data" / "MultiHiertt" / "dev.json"
    mulhi_data = json.loads(mulhi_path.read_text(encoding="utf-8"))
    for index, item in enumerate(mulhi_data):
        tables = item.get("tables")
        if isinstance(tables, list) and tables:
            document = document_from_json(item, table_index=0)
            uid = str(item.get("uid", index)).replace("/", "_").replace("\\", "_")
            jobs.append((f"mulhi_{uid}", document))
            break

    results = []
    for name, document in jobs:
        result = render_document(
            document,
            output_dir / f"{name}.png",
            scale=args.scale,
            browser_path=args.browser,
            keep_html=True,
        )
        results.append(result)

    for result in results:
        size = f"{result.width}x{result.height}" if result.width and result.height else "unknown size"
        print(f"{result.image_path.resolve()} ({size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
