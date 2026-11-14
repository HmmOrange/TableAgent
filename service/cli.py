"""`table-agent`: answer questions over one workbook from the command line and print the job as JSON."""

from __future__ import annotations

import argparse
import json
import sys

from service.runtime import TableAgentService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Answer questions over an Excel workbook with TableAgent.")
    parser.add_argument("--config", default="config.yaml", help="Path to the private service configuration.")
    parser.add_argument("--workbook", required=True, help="The .xlsx (or .xlsm) workbook to question.")
    parser.add_argument("--question", action="append", required=True, help="A question; repeat for several.")
    parser.add_argument("--llm", help="Model profile to answer with instead of the config's `llm`.")
    parser.add_argument("--vlm", help="Model profile to look at images with instead of the config's `vlm`.")
    parser.add_argument("--output-dir", help="Directory for the job's artifacts (default: service.root_dir).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = TableAgentService.from_config(args.config, llm_profile=args.llm, vlm_profile=args.vlm, root_dir=args.output_dir)
    try:
        result = service.run(args.workbook, args.question)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
