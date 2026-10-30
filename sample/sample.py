"""Answer one question about a workbook with the TableAgent QA stage and print the outcome."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TableAgent.configs import DEFAULT_CONFIG_PATH, load_config
from TableAgent.stages.qa import answer_question
from service import create_model_client

DEFAULT_STRUCTURE_PATH = ROOT / "sample" / "structure.yaml"
DEFAULT_WORKBOOK_PATH = ROOT / "sample" / "QA_sample.xlsx"
DEFAULT_QUESTION = (
    "Find every person whose last name is Pham and whose birth year is 1995. "
    "Return each matching person's full name and score."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TableAgent QA stage on a workbook question.")
    parser.add_argument("--structure", default=str(DEFAULT_STRUCTURE_PATH))
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK_PATH))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--llm-profile", default="table_agent", help="Model profile passed to create_model_client.")
    parser.add_argument("--artifacts", default=str(ROOT / "logs" / "qa_runs"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    llm_client = create_model_client(load_config(args.config), kind="llm", profile=args.llm_profile)
    result = answer_question(
        llm_client, args.question, Path(args.workbook), [Path(args.structure)], Path(args.artifacts), progress=print,
    )
    print(json.dumps({
        "question": result.question,
        "success": result.success,
        "answer": result.answer,
        "error": result.error,
        "seconds": round(result.seconds, 1),
        "trace": result.artifacts.get("trace_md"),
    }, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
