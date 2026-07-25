from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

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
        "--artifacts",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Indexed retrieval artifact file (run.json, JSON/JSONL, or retrieval_cards.pkl). "
            "Repeat for multiple files; requires --stage qa."
        ),
    )
    parser.add_argument(
        "--answer-instruction",
        help=(
            "Additional analytical or answer instructions for indexed QA. "
            "This does not affect retrieval ranking."
        ),
    )
    parser.add_argument(
        "--expected-output",
        help="Expected indexed-QA answer shape or acceptance criteria for final review.",
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
    parser.add_argument(
        "--workers",
        "--max-workers",
        dest="max_workers",
        type=_positive_int,
        default=None,
        metavar="N",
        help=(
            "Maximum concurrent structure and QA workers. "
            "Overrides service.max_workers for this run."
        ),
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
    if cleanup_requested and (
        args.workbook
        or args.query
        or args.embed
        or args.sheet
        or args.artifacts
        or args.answer_instruction
        or args.expected_output
        or args.max_workers is not None
    ):
        parser.error("cleanup flags cannot be combined with workbook processing flags")
    if not cleanup_requested and not args.workbook:
        parser.error("--workbook is required unless deleting saved jobs")
    if not cleanup_requested and args.stage in {"qa", "all"} and not any(query.strip() for query in args.query):
        parser.error("--query is required when --stage is qa or all")
    if args.artifacts and args.stage != "qa":
        parser.error("--artifacts requires --stage qa")
    if args.artifacts and args.embed:
        parser.error("--embed cannot be combined with indexed --artifacts")
    if args.artifacts and args.sheet:
        parser.error("--sheet cannot be combined with indexed --artifacts")
    if args.artifacts and len([query for query in args.query if query.strip()]) != 1:
        parser.error("Indexed QA with --artifacts requires exactly one non-empty --query")
    if (args.answer_instruction or args.expected_output) and not args.artifacts:
        parser.error("--answer-instruction and --expected-output require indexed --artifacts")

    try:
        service = TableAgentService.from_config(
            args.config,
            llm_profile=args.llm,
            vlm_profile=args.vlm,
        )
        if cleanup_requested:
            result = service.delete_runs(args.delete_job, all_runs=args.delete_all_jobs)
        elif args.artifacts:
            run_kwargs = {
                "query": next(query for query in args.query if query.strip()),
                "workbooks": args.workbook,
                "artifacts": load_artifacts(args.artifacts),
                "answer_instruction": args.answer_instruction,
                "expected_output": args.expected_output,
            }
            if args.max_workers is not None:
                run_kwargs["max_workers"] = args.max_workers
            result = service.run_indexed_qa(
                **run_kwargs,
            )
        else:
            run_kwargs = {
                "stage": args.stage,
                "workbooks": args.workbook,
                "queries": args.query,
                "embed": args.embed,
                "sheets": args.sheet,
            }
            if args.max_workers is not None:
                run_kwargs["max_workers"] = args.max_workers
            result = service.run(**run_kwargs)
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"table-agent: error: {exc}", file=sys.stderr)
        return 1

    stdout_encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if stdout_encoding != "utf8" and callable(reconfigure):
        reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def load_artifacts(paths: list[str]) -> list[dict[str, Any]]:
    """Load indexed records from ingestion output files for CLI QA."""
    records: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            try:
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
            except (EOFError, pickle.UnpicklingError, AttributeError, ValueError, TypeError) as exc:
                raise ValueError(f"Could not read pickle artifact file: {path}") from exc
        elif suffix == ".jsonl":
            try:
                payload = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not read JSONL artifact file: {path}") from exc
        else:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not read JSON artifact file: {path}") from exc
        records.extend(_artifact_records(payload, path))

    deduplicated: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("id") or ""),
            str(
                record.get("upload_name")
                or record.get("document_name")
                or record.get("workbook")
                or ""
            ),
            str(record.get("sheet") or record.get("sheet_name") or ""),
            str(record.get("table_id") or ""),
        )
        deduplicated.setdefault(key, record)
    if not deduplicated:
        raise ValueError("Artifact files did not contain any retrieval records")
    return list(deduplicated.values())


def _artifact_records(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("retrieval_records", "artifacts", "records"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload] if payload.get("id") else []
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(
            f"Artifact file must contain a list of records or a run.json wrapper: {path}"
        )
    return [dict(item) for item in payload]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
