from __future__ import annotations

import argparse
import json
import sys

from service.runtime import TableAgentService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TableAgent or delete saved CLI runs.")
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
        default=[],
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
        "--embed",
        action="store_true",
        help="Generate retrieval_cards.pkl with embeddings for ingestion retrieval cards.",
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
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument(
        "--delete-job",
        action="append",
        default=[],
        metavar="ID",
        help="Delete a saved run directory. Repeat for multiple runs.",
    )
    cleanup.add_argument(
        "--delete-all-jobs",
        action="store_true",
        help="Delete every saved TableAgent run under service.root_dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cleanup_requested = bool(args.delete_job or args.delete_all_jobs)
    if cleanup_requested and (args.workbook or args.query or args.embed or args.sheet):
        parser.error("cleanup flags cannot be combined with workbook processing flags")
    if not cleanup_requested and not args.workbook:
        parser.error("--workbook is required unless deleting saved jobs")
    if not cleanup_requested and args.stage in {"qa", "all"} and not any(query.strip() for query in args.query):
        parser.error("--query is required when --stage is qa or all")

    try:
        service = TableAgentService.from_config(
            args.config,
            llm_profile=args.llm,
            vlm_profile=args.vlm,
        )
        if cleanup_requested:
            result = service.delete_runs(args.delete_job, all_runs=args.delete_all_jobs)
        else:
            result = service.run(
                stage=args.stage,
                workbooks=args.workbook,
                queries=args.query,
                embed=args.embed,
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
