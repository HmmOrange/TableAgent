"""Run the TableAgent QA package from a JSON request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from .qa_package import TableQAPackage, load_qa_request, qa_response_payload


PackageFactory = Callable[..., TableQAPackage]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--llm-profile", default="table_agent")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    package_factory: PackageFactory = TableQAPackage.from_config,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = load_qa_request(args.request)
        package = package_factory(args.config, llm_profile=args.llm_profile)
        try:
            response = package.run(request)
        finally:
            package.close()
        payload = qa_response_payload(response)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"table-agent-qa: error: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if response.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
