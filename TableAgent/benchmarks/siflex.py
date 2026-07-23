from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from service.clients import create_model_client
from TableAgent.configs import load_config
from TableAgent.llm import BaseLLM
from TableAgent.pipeline import TableAgentPipeline
from TableAgent.pipeline.base import PipelineOutput
from TableAgent.schema import EvalSample


@dataclass(frozen=True)
class PerfectSourceSpec:
    workbook: str
    sheet: str


# This is the source oracle used by the successful 18/18 SiFlex runs.
SIFLEX_PERFECT_SOURCES: dict[str, PerfectSourceSpec] = {
    "Golden_Test_LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522:Q1 – TABLE": PerfectSourceSpec(
        "LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522.xlsx", "Tiêu chuẩn chung AOI."
    ),
    "Golden_Test_LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522:Q2 – TABLE": PerfectSourceSpec(
        "LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522.xlsx", "Tiêu chuẩn chung AOI."
    ),
    "Golden_Test_LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522:Q4 – FORM": PerfectSourceSpec(
        "LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522.xlsx", "Tiêu chuẩn chung AOI."
    ),
    "Golden_Test_LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522:Q6 – TABLE": PerfectSourceSpec(
        "LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522.xlsx", "Tiêu chuẩn chung AOI."
    ),
    "Golden_Test_LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522:Q7 – LIST": PerfectSourceSpec(
        "LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522.xlsx", "Tiêu chuẩn chung AOI."
    ),
    "Golden_Test_LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522:Q8 – TABLE": PerfectSourceSpec(
        "LV01_ENG_REPORT_TIÊU CHUẨN KIỂM TRA AOI ( Áp dụng đường mạch)_VN_20250522.xlsx", "Tiêu chuẩn chung AOI."
    ),
    "Golden_Test_LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26:Q1 – TABLE": PerfectSourceSpec(
        "LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26.xlsx", "2026년 설비유지보수 계획"
    ),
    "Golden_Test_LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26:Q2 – LIST": PerfectSourceSpec(
        "LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26.xlsx", "2026년 설비유지보수 계획"
    ),
    "Golden_Test_LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26:Q9 – LIST": PerfectSourceSpec(
        "LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26.xlsx", "2026년 설비유지보수 계획"
    ),
    "Golden_Test_LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26:Q3 – LIST": PerfectSourceSpec(
        "LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26.xlsx", "Nối khí-Ống nước"
    ),
    "Golden_Test_LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26:Q4 – FORM": PerfectSourceSpec(
        "LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26.xlsx", "OIL"
    ),
    "Golden_Test_LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26:Q6 – TABLE": PerfectSourceSpec(
        "LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26.xlsx", "PRESS"
    ),
    "Golden_Test_LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26:Q7 – LIST": PerfectSourceSpec(
        "LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26.xlsx", "HP 1"
    ),
    "Golden_Test_LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26:Q8 – TABLE": PerfectSourceSpec(
        "LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26.xlsx", "AUTO"
    ),
    "Golden_Test_LV01_설비_REPORT 각공정 고장 설비 수리 진행 현황 20260316_KR,VN_202603.26:Q3 – TABLE": PerfectSourceSpec(
        "LV01_설비_REPORT 각공정 고장 설비 수리 진행 현황 20260316_KR,VN_202603.26.xlsx", "Bao_cao_F2"
    ),
    "Golden_Test_LV01_설비_REPORT 각공정 고장 설비 수리 진행 현황 20260316_KR,VN_202603.26:Q4 – FORM": PerfectSourceSpec(
        "LV01_설비_REPORT 각공정 고장 설비 수리 진행 현황 20260316_KR,VN_202603.26.xlsx", "Bao_cao_F2"
    ),
    "Golden_Test_LV01_설비_REPORT 각공정 고장 설비 수리 진행 현황 20260316_KR,VN_202603.26:Q5 – TABLE": PerfectSourceSpec(
        "LV01_설비_REPORT 각공정 고장 설비 수리 진행 현황 20260316_KR,VN_202603.26.xlsx", "Bao_cao_F2"
    ),
    "Golden_Test_LV01_설비_REPORT 각공정 고장 설비 수리 진행 현황 20260316_KR,VN_202603.26:Q6 – FORM": PerfectSourceSpec(
        "LV01_설비_REPORT 각공정 고장 설비 수리 진행 현황 20260316_KR,VN_202603.26.xlsx", "Bao_cao_F2"
    ),
}

SIFLEX_PERFECT_EXCLUDED_SAMPLE_IDS = frozenset({
    "Golden_Test_LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26:Q3 – TABLE",
    "Golden_Test_LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26:Q4 – TABLE",
    "Golden_Test_LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26:Q5 – TABLE",
    "Golden_Test_LV01_설비_REPORT 2026년  설비유지보수 계획 VER 1.0_KR_202603.26:Q6 – FORM",
    "Golden_Test_LV01_설비_REPORT Quản Lý Thiết bị Tháng 02.2025스케_KR,VN_202603.26:Q5 – LIST",
})

JUDGE_SYSTEM_PROMPT = (
    "You are a strict benchmark judge for document QA. Score only against the "
    "reference answer. Be deterministic. Return JSON only with keys: "
    "factual_correctness, coverage, structure_fidelity, grounding, rationale. "
    "Each score must be a number from 0 to 1. For table/list/form answers, "
    "accept minor wording differences if the facts and row/field relations are "
    "preserved. Penalize missing key fields, wrong values, swapped row "
    "meanings, and hallucinated facts."
)
JUDGE_USER_PROMPT_TEMPLATE = """\
[Question]
{question}

[Expected answer type]
{answer_type}

[Reference answer]
{reference_answer}

[System answer]
{system_answer}

[Scoring rubric]
- factual_correctness: are stated facts/values/relations correct
- coverage: are the important fields/rows/items covered
- structure_fidelity: does the answer preserve useful table/list/form structure
- grounding: does the answer avoid unsupported additions
Return JSON only."""


class SiflexJudge:
    def __init__(self, llm: BaseLLM, pass_threshold: float = 0.8) -> None:
        self.llm = llm
        self.pass_threshold = pass_threshold

    def evaluate(self, sample: EvalSample, output: PipelineOutput) -> dict[str, Any]:
        answer = output.predicted_answer.strip()
        if not answer:
            return self._metrics(0, 0, 0, 0, "System answer is empty.")
        reference = str(sample.answer[0] if sample.answer else "")
        prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            question=sample.question.strip(),
            answer_type=str(sample.raw.get("answer_type", "")).strip(),
            reference_answer=reference.strip(),
            system_answer=answer,
        )
        try:
            response = self.llm.generate(prompt=prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
            data = _normalize_judge_schema(_parse_judge_json(response.content))
            metrics = self._metrics(
                _clamp_score(data.get("factual_correctness")),
                _clamp_score(data.get("coverage")),
                _clamp_score(data.get("structure_fidelity")),
                _clamp_score(data.get("grounding")),
                str(data.get("rationale", "")).strip(),
            )
            metrics["judge_tokens"] = {
                "prompt": response.prompt_tokens,
                "completion": response.completion_tokens,
            }
            metrics["judge_raw_response"] = response.content
            return metrics
        except Exception as exc:
            return self._metrics(0, 0, 0, 0, f"Judge generation failed: {exc}")

    def _metrics(self, factual: float, coverage: float, fidelity: float, grounding: float, rationale: str) -> dict[str, Any]:
        overall = round(0.45 * factual + 0.30 * coverage + 0.15 * fidelity + 0.10 * grounding, 4)
        return {
            "pass": overall >= self.pass_threshold,
            "factual_correctness": factual,
            "coverage": coverage,
            "structure_fidelity": fidelity,
            "grounding": grounding,
            "overall_score": overall,
            "rationale": rationale,
        }


def _clamp_score(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _parse_judge_json(raw: str) -> dict[str, object]:
    text = raw.strip()
    candidates = [text]
    candidates.extend(match.strip() for match in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1].strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Judge LLM did not return valid JSON")


def _normalize_judge_schema(data: dict[str, object]) -> dict[str, object]:
    required = {"factual_correctness", "coverage", "structure_fidelity", "grounding"}
    nested = data.get("sub_query_quality_scores")
    if required.issubset(data) or not isinstance(nested, dict):
        return data
    normalized = dict(data)
    for field in required:
        if field in nested and field not in normalized:
            normalized[field] = nested[field]
    return normalized


def _sample_id(raw: dict[str, Any]) -> str:
    workbook = str(raw.get("golden_workbook", "case")).replace("\\", "/")
    return f"{PurePosixPath(workbook).stem}:{raw.get('sheet_name', '')}"


def _resolve_source(data_dir: Path, value: str) -> Path:
    text = value.replace("\\", "/")
    if text.startswith("golden_test/"):
        text = "golden_tests/" + text[len("golden_test/") :]
    path = data_dir / text
    if not path.is_file():
        raise FileNotFoundError(f"SiFlex source file not found: {path}")
    return path


def load_siflex_samples(data_dir: Path, *, source_specs: dict[str, PerfectSourceSpec] = SIFLEX_PERFECT_SOURCES) -> list[EvalSample]:
    split_path = data_dir / "golden_tests" / "compiled" / "golden_cases.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    samples: list[EvalSample] = []
    for index, raw in enumerate(payload.get("cases", [])):
        sample_id = _sample_id(raw)
        if sample_id in SIFLEX_PERFECT_EXCLUDED_SAMPLE_IDS:
            continue
        spec = source_specs.get(sample_id)
        if spec is None:
            raise KeyError(f"No perfect source mapping for retained sample {sample_id!r}")
        source_paths = [_resolve_source(data_dir, value) for value in raw.get("source_paths", [])]
        question = str(raw.get("question", ""))
        answer = _clean_source_reference(str(raw.get("reference_answer", "")))
        samples.append(EvalSample(
            index=index,
            sample_id=sample_id,
            table_id="|".join(path.name for path in source_paths),
            table_content="",
            question=question,
            answer=[answer],
            sample_path=f"{split_path.resolve()}:cases[{index}]",
            table_path=";".join(str(path.resolve()) for path in source_paths),
            raw={**raw, "answer_type": str(raw.get("answer_type", "")), "perfect_source": {
                "workbook": spec.workbook,
                "sheet": spec.sheet,
            }},
        ))
    if len(samples) != 18:
        raise RuntimeError(f"Expected 18 retained SiFlex samples, found {len(samples)}")
    return samples


def _clean_source_reference(text: str) -> str:
    lines = text.split("\n")
    while lines and (not lines[-1].strip() or lines[-1].lstrip("|- \t").startswith("Nguồn:")):
        lines.pop()
    return "\n".join(lines).strip()


def _preflight(*clients: Any) -> None:
    checked: set[str] = set()
    for client in clients:
        base_url = str(getattr(client, "base_url", "")).rstrip("/")
        if not base_url or base_url in checked:
            continue
        checked.add(base_url)
        response = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {getattr(client, 'api_key', '')}"},
            timeout=5,
        )
        if response.status_code >= 500:
            response.raise_for_status()


def run_benchmark(
    *,
    config_path: str | Path,
    data_dir: str | Path,
    source_artifacts: str | Path,
    output_dir: str | Path,
    answer_profile: str = "table_agent",
    judge_profile: str | None = None,
    skip_preflight: bool = False,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    data_root = Path(data_dir).resolve()
    source_root = Path(source_artifacts).resolve()
    samples = _select_samples(load_siflex_samples(data_root), sample_ids)
    answer_client = create_model_client(config, kind="llm", profile=answer_profile)
    effective_judge_profile = judge_profile or ("siflex_judge" if "siflex_judge" in config else answer_profile)
    judge_client = create_model_client(config, kind="llm", profile=effective_judge_profile)
    if not skip_preflight:
        _preflight(answer_client, judge_client)

    agent_config = dict(config.get("table_agent") or {})
    agent_config.update({
        "phase": "qa",
        "source_artifact_dir": str(source_root),
        "artifact_dir": str(Path(output_dir).resolve() / "artifacts"),
        "perfect_retrieval": True,
        "run_retrieval": True,
        "retrieval_rerank_with_llm": False,
        "cache_namespace": "siflex",
    })
    pipeline = TableAgentPipeline(llm_client=answer_client, layout_vlm_client=None, config=agent_config)
    pipeline.prepare_samples(samples)
    judge = SiflexJudge(judge_client, pass_threshold=float((config.get("siflex_judge") or {}).get("pass_threshold", 0.8)))
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    total = len(samples)
    for number, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        try:
            output = pipeline.run(sample)
            metrics = judge.evaluate(sample, output)
            record = {"sample_id": sample.sample_id, "question": sample.question, "answer": output.predicted_answer, "metrics": metrics, "metadata": output.metadata, "latency": time.perf_counter() - started}
        except Exception as exc:
            record = {"sample_id": sample.sample_id, "question": sample.question, "error": str(exc), "metrics": {"pass": False, "overall_score": 0.0}}
        results.append(record)
        status = "PASS" if record.get("metrics", {}).get("pass") else "FAIL"
        print(f"[{number:02d}/{total}] {status} {sample.sample_id} score={record.get('metrics', {}).get('overall_score', 0.0):.4f}", flush=True)
        (output_root / "report.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    passed = sum(1 for record in results if record.get("metrics", {}).get("pass"))
    summary = {"passed": passed, "total": len(results), "pass_rate": passed / len(results), "perfect_retrieval": True}
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SiFlex perfect retrieval: {passed}/{len(results)} passed ({summary['pass_rate']:.1%})")
    return {"summary": summary, "results": results}


def _select_samples(samples: list[EvalSample], requested_ids: list[str] | None) -> list[EvalSample]:
    if not requested_ids:
        return samples

    selected: list[EvalSample] = []
    for requested_id in requested_ids:
        exact = [sample for sample in samples if sample.sample_id == requested_id]
        matches = exact or [sample for sample in samples if requested_id in sample.sample_id]
        if not matches:
            raise ValueError(f"No retained SiFlex sample matches {requested_id!r}")
        if len(matches) > 1:
            match_ids = ", ".join(sample.sample_id for sample in matches)
            raise ValueError(f"SiFlex sample selector {requested_id!r} is ambiguous: {match_ids}")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone TableAgent SiFlex v5 perfect-retrieval benchmark.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="data/SiFlex")
    parser.add_argument("--source-artifacts", default="outputs/v5/prepared")
    parser.add_argument("--output-dir", default="outputs/siflex-perfect")
    parser.add_argument("--answer-profile", default="table_agent")
    parser.add_argument("--judge-profile", default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Run one retained sample by exact ID or a unique ID substring; repeat to select multiple samples.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_benchmark(
            config_path=args.config,
            data_dir=args.data_dir,
            source_artifacts=args.source_artifacts,
            output_dir=args.output_dir,
            answer_profile=args.answer_profile,
            judge_profile=args.judge_profile,
            skip_preflight=args.skip_preflight,
            sample_ids=args.sample_id,
        )
    except Exception as exc:
        print(f"siflex benchmark: error: {exc}", file=sys.stderr)
        return 1
    return 0 if result["summary"]["passed"] == result["summary"]["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
