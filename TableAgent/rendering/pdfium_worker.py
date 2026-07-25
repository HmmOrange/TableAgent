from __future__ import annotations

import argparse
from pathlib import Path


def render_pdf_page(
    pdf_path: str | Path,
    image_path: str | Path,
    resolution: int,
) -> None:
    """Render the first PDF page inside the current process."""
    try:
        import pypdfium2
    except ImportError as exc:
        raise RuntimeError(
            "PDF rendering requires pypdfium2. Install it before running "
            "TableAgent workbook image rendering."
        ) from exc

    destination = Path(image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    try:
        page = pdf[0]
        try:
            bitmap = page.render(scale=max(1, int(resolution)) / 72)
            try:
                bitmap.to_pil().save(destination)
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        pdf.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one PDF page for TableAgent.")
    parser.add_argument("pdf_path")
    parser.add_argument("image_path")
    parser.add_argument("resolution", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_pdf_page(args.pdf_path, args.image_path, args.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
