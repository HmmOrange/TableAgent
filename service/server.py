from __future__ import annotations

import argparse

import uvicorn

from service.api import create_app
from service.runtime import TableAgentService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TableAgent HTTP API.")
    parser.add_argument("--config", default="config.yaml", help="Path to the private service configuration.")
    parser.add_argument(
        "--llm",
        help="Configured LLM profile to use instead of the config.yaml default.",
    )
    parser.add_argument(
        "--vlm",
        help="Configured VLM profile to use instead of the config.yaml default.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = TableAgentService.from_config(
        args.config,
        llm_profile=args.llm,
        vlm_profile=args.vlm,
    )
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
