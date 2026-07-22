from __future__ import annotations

import argparse
import json
import sys

from service.runtime import TableAgentService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TableAgent once over one or more workbooks.")
    parser.add_argument("--config", default="config.yaml", help="Path to the private service configuration.")
    parser.add_argument(
        "--stage",
        choices=("structure", "qa", "all"),
        default="all",
        help="Processing stage to run (default: all).",
    )
    parser.add_argument(
        "--workbook",
        action="append",
        required=True,
        metavar="PATH",
        help="Workbook to process. Repeat for multiple workbooks.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        metavar="TEXT",
        help="Question to answer. Repeat for multiple questions; required for qa and all.",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Generate a workbook schema artifact. If neither output flag is set, both are generated.",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Generate a workbook metadata artifact. If neither output flag is set, both are generated.",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="Process only the named worksheet(s). Repeat the flag or separate names with commas.",
    )
    parser.add_argument(
        "--llm",
        help="Configured LLM profile to use instead of the config.yaml default.",
    )
    parser.add_argument(
        "--vlm",
        help="Configured VLM profile to use instead of the config.yaml default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage in {"qa", "all"} and not any(query.strip() for query in args.query):
        parser.error("--query is required when --stage is qa or all")

    try:
        service = TableAgentService.from_config(
            args.config,
            llm_profile=args.llm,
            vlm_profile=args.vlm,
        )
        result = service.run(
            stage=args.stage,
            workbooks=args.workbook,
            queries=args.query,
            schema=args.schema,
            metadata=args.metadata,
            sheets=args.sheet,
        )
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"table-agent: error: {exc}", file=sys.stderr)
        return 1

    stdout_encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if stdout_encoding != "utf8" and callable(reconfigure):
        reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
