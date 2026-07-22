from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Literal

import openpyxl
import pandas as pd

from service.clients import create_model_client
from TableAgent.configs import load_config
from TableAgent.pipeline import TableAgentPipeline
from TableAgent.pipeline.base import PipelineOutput
from TableAgent.pipeline.common import safe_name
from TableAgent.pipeline.source_preparer import LAYOUT_WORKFLOW_VERSION
from TableAgent.schema import EvalSample
from TableAgent.structure.layout.parsing import _is_valid_structure


Stage = Literal["structure", "qa", "all"]
SUPPORTED_WORKBOOK_EXTENSIONS = {".xls", ".xlsm", ".xlsx", ".xltm", ".xltx"}


class TableAgentService:
    """Reusable entry point for running TableAgent over workbook and query batches."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        llm_client: Any | None = None,
        layout_vlm_client: Any | None = None,
        llm_profile: str | None = None,
        vlm_profile: str | None = None,
        root_dir: str | Path | None = None,
        pipeline_factory: Callable[..., TableAgentPipeline] = TableAgentPipeline,
    ):
        self.config = dict(config)
        service_config = self.config.get("service") or {}
        if not isinstance(service_config, dict):
            raise ValueError("service configuration must be a mapping")
        configured_root = root_dir or service_config.get("root_dir", "outputs/table_agent/service")
        self.root_dir = Path(configured_root).expanduser().resolve()
        self.input_dir = self.root_dir / "inputs"
        self.jobs_dir = self.root_dir / "jobs"
        self.structure_dir = self.root_dir / "structure"
        for path in (self.input_dir, self.jobs_dir, self.structure_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.max_workers = max(1, int(service_config.get("max_workers", 1)))
        self.max_upload_bytes = max(1, int(service_config.get("max_upload_mb", 100))) * 1024 * 1024
        self.api_key = str(service_config["api_key"]) if service_config.get("api_key") else None
        self.allow_local_paths = bool(service_config.get("allow_local_paths", False))
        self.allowed_input_roots = tuple(
            Path(value).expanduser().resolve()
            for value in service_config.get("allowed_input_roots", [])
        )
        self._llm_client = llm_client
        self._layout_vlm_client = layout_vlm_client
        self.llm_profile = llm_profile or "table_agent"
        self.vlm_profile = vlm_profile or "table_agent"
        self.pipeline_factory = pipeline_factory

    @classmethod
    def from_config(
        cls,
        path: str | Path = "config.yaml",
        **kwargs: Any,
    ) -> "TableAgentService":
        return cls(load_config(path), **kwargs)

    def run(
        self,
        *,
        stage: Stage = "all",
        queries: Iterable[str] = (),
        workbooks: Iterable[str | Path],
        job_id: str | None = None,
    ) -> dict[str, Any]:
        stage = _validate_stage(stage)
        query_list = _validate_queries(queries, required=stage in {"qa", "all"})
        workbook_list = [Path(value).expanduser().resolve() for value in workbooks]
        if not workbook_list:
            raise ValueError("At least one workbook is required")
        normalized = self._normalize_workbooks(workbook_list)

        run_id = safe_name(job_id or uuid.uuid4().hex)
        if run_id in {".", ".."}:
            raise ValueError("Invalid job id")
        job_dir = (self.jobs_dir / run_id).resolve()
        if not job_dir.is_relative_to(self.jobs_dir):
            raise ValueError("Invalid job id")
        job_dir.mkdir(parents=True, exist_ok=True)

        table_path = ";".join(str(item["path"]) for item in normalized)
        base_sample = self._sample(
            sample_id=f"{run_id}-structure",
            question=query_list[0] if query_list else "Generate workbook structure",
            table_path=table_path,
            workbook_names=[item["name"] for item in normalized],
        )
        structures: list[dict[str, Any]] = []
        answers: list[dict[str, Any]] = []

        if stage in {"structure", "all"}:
            pipeline = self.pipeline_factory(
                llm_client=None,
                layout_vlm_client=self._layout_client(),
                config=self._pipeline_config("structure", job_dir),
            )
            records = pipeline.verify_samples([base_sample], force=False)
            structures = self._structure_results(records, normalized, job_dir)
            structures = self._complete_structure_results(structures, normalized)
            failed = [record for record in structures if record["status"] != "good"]
            if failed:
                raise RuntimeError(f"Structure generation failed for {len(failed)} workbook sheet(s)")

        if stage == "qa":
            structures = self._cached_structure_results(normalized, job_dir)
            failed = [record for record in structures if record["status"] != "good"]
            if failed:
                raise RuntimeError(
                    f"Missing or stale structure cache for {len(failed)} workbook sheet(s); run structure or all first"
                )

        if stage in {"qa", "all"}:
            samples = [
                self._sample(
                    sample_id=f"{run_id}-query-{index}",
                    question=query,
                    table_path=table_path,
                    workbook_names=[item["name"] for item in normalized],
                )
                for index, query in enumerate(query_list, start=1)
            ]
            pipeline = self.pipeline_factory(
                llm_client=self._answer_client(),
                layout_vlm_client=None,
                config=self._pipeline_config("qa", job_dir),
            )
            pipeline.prepare_samples(samples)
            for sample in samples:
                output = pipeline.run(sample)
                answers.append(self._answer_result(sample.question, output, normalized))

        result = {
            "job_id": run_id,
            "stage": stage,
            "workbooks": [item["name"] for item in normalized],
            "structures": structures,
            "answers": answers,
            "artifacts": self._artifact_paths(job_dir),
        }
        (job_dir / "run.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        result["artifacts"] = self._artifact_paths(job_dir)
        return result

    def validate_local_workbook(self, value: str | Path) -> Path:
        if not self.allow_local_paths:
            raise PermissionError("Server-side workbook paths are disabled; upload the workbook instead")
        path = Path(value).expanduser().resolve()
        if self.allowed_input_roots and not any(path.is_relative_to(root) for root in self.allowed_input_roots):
            raise PermissionError(f"Workbook path is outside the configured allowed_input_roots: {path}")
        self._validate_workbook(path)
        return path

    def _answer_client(self) -> Any:
        if self._llm_client is None:
            self._llm_client = create_model_client(
                self.config,
                kind="llm",
                profile=self.llm_profile,
            )
        return self._llm_client

    def _layout_client(self) -> Any:
        if self._layout_vlm_client is None:
            self._layout_vlm_client = create_model_client(
                self.config,
                kind="vlm",
                profile=self.vlm_profile,
            )
        return self._layout_vlm_client

    def _pipeline_config(self, phase: Stage, job_dir: Path) -> dict[str, Any]:
        agent_config = dict(self.config.get("table_agent") or {})
        agent_config.update(
            {
                "phase": phase,
                "artifact_dir": str(job_dir / "artifacts"),
                "source_artifact_dir": str(self.structure_dir),
                "structure_cache_dir": str(self.structure_dir / "cache"),
                "cache_namespace": "service",
            }
        )
        return agent_config

    @staticmethod
    def _sample(
        *,
        sample_id: str,
        question: str,
        table_path: str,
        workbook_names: list[str],
    ) -> EvalSample:
        return EvalSample(
            index=0,
            sample_id=sample_id,
            table_id="workbook_set",
            table_content="",
            question=question,
            answer=[],
            sample_path="service/siflex/request.json",
            table_path=table_path,
            raw={"source": "table-agent-service", "workbooks": workbook_names},
        )

    def _normalize_workbook(self, source: Path) -> dict[str, Any]:
        self._validate_workbook(source)
        digest = _sha256(source)
        destination = self.input_dir / f"{digest[:24]}.xlsx"
        if not destination.is_file():
            staging = self.input_dir / f".{digest[:24]}-{uuid.uuid4().hex}.xlsx"
            try:
                if source.suffix.lower() == ".xlsx":
                    shutil.copy2(source, staging)
                elif source.suffix.lower() == ".xls":
                    sheets = pd.read_excel(source, sheet_name=None)
                    with pd.ExcelWriter(staging, engine="openpyxl") as writer:
                        for sheet_name, frame in sheets.items():
                            frame.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)
                else:
                    workbook = openpyxl.load_workbook(source, data_only=False, keep_vba=False)
                    try:
                        workbook.save(staging)
                    finally:
                        workbook.close()
                staging.replace(destination)
            finally:
                staging.unlink(missing_ok=True)
        return {"name": source.name, "path": destination, "sha256": digest}

    def _normalize_workbooks(self, sources: list[Path]) -> list[dict[str, Any]]:
        normalized = []
        seen_paths: set[Path] = set()
        for source in sources:
            item = self._normalize_workbook(source)
            if item["path"] in seen_paths:
                continue
            seen_paths.add(item["path"])
            normalized.append(item)
        name_counts: dict[str, int] = {}
        for item in normalized:
            name_counts[item["name"]] = name_counts.get(item["name"], 0) + 1
        for item in normalized:
            if name_counts[item["name"]] > 1:
                item["name"] = f"{item['name']} ({item['sha256'][:8]})"
        return normalized

    @staticmethod
    def _validate_workbook(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Workbook not found: {path}")
        if path.suffix.lower() not in SUPPORTED_WORKBOOK_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_WORKBOOK_EXTENSIONS))
            raise ValueError(f"Unsupported workbook extension '{path.suffix}'; expected one of: {supported}")

    def _structure_results(
        self,
        records: Iterable[Any],
        normalized: list[dict[str, Any]],
        job_dir: Path,
    ) -> list[dict[str, Any]]:
        results = []
        for index, record in enumerate(records, start=1):
            workbook_name = _workbook_name(record.workbook_path, normalized)
            artifact = None
            structure_text = None
            if record.structure_path.is_file():
                structure_text = record.structure_path.read_text(encoding="utf-8")
                filename = f"{index:03d}_{safe_name(workbook_name)}_{safe_name(record.sheet_name)}.yaml"
                target = job_dir / "structures" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record.structure_path, target)
                artifact = target.relative_to(job_dir).as_posix()
            results.append(
                {
                    "workbook": workbook_name,
                    "sheet": record.sheet_name,
                    "status": record.status,
                    "cache_hit": record.cache_hit,
                    "structure": structure_text,
                    "artifact": artifact,
                }
            )
        return results

    def _cached_structure_results(
        self,
        normalized: list[dict[str, Any]],
        job_dir: Path,
    ) -> list[dict[str, Any]]:
        allowed = {str(item["path"].resolve()) for item in normalized}
        records = []
        source_root = self.structure_dir / "sources"
        if source_root.is_dir():
            for source_dir in source_root.iterdir():
                metadata_path = source_dir / "metadata.json"
                structure_path = source_dir / "structure.yaml"
                if not metadata_path.is_file() or not structure_path.is_file():
                    continue
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    workbook_path = Path(str(metadata.get("workbook_path", ""))).resolve()
                    structure_text = structure_path.read_text(encoding="utf-8")
                except (OSError, json.JSONDecodeError):
                    continue
                if str(workbook_path) not in allowed:
                    continue
                records.append(
                    SimpleNamespace(
                        workbook_path=workbook_path,
                        sheet_name=str(metadata.get("sheet_name", "")),
                        structure_path=structure_path,
                        status=(
                            "good"
                            if metadata.get("layout_workflow_version") == LAYOUT_WORKFLOW_VERSION
                            and _is_valid_structure(structure_text)
                            else "not_good"
                        ),
                        cache_hit=True,
                    )
                )
        results = self._structure_results(records, normalized, job_dir)
        return self._complete_structure_results(results, normalized)

    @staticmethod
    def _complete_structure_results(
        results: list[dict[str, Any]],
        normalized: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_identity = {(item["workbook"], item["sheet"]): item for item in results}
        completed = []
        for item in normalized:
            workbook = openpyxl.load_workbook(item["path"], read_only=True, data_only=True)
            try:
                sheet_names = list(workbook.sheetnames)
            finally:
                workbook.close()
            for sheet_name in sheet_names:
                identity = (item["name"], sheet_name)
                completed.append(
                    by_identity.get(identity)
                    or {
                        "workbook": item["name"],
                        "sheet": sheet_name,
                        "status": "not_good",
                        "cache_hit": False,
                        "structure": None,
                        "artifact": None,
                    }
                )
        return completed

    @staticmethod
    def _answer_result(
        query: str,
        output: PipelineOutput,
        normalized: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metadata = output.metadata or {}
        qa = dict(metadata.get("qa") or {})
        qa.pop("artifacts", None)
        return {
            "query": query,
            "answer": output.predicted_answer,
            "latency": output.latency,
            "token_usage": output.token_usage,
            "workbook": _workbook_name(Path(str(metadata.get("workbook_path", ""))), normalized),
            "sheets": metadata.get("workbook_sheets") or [],
            "verification": metadata.get("verification") or {},
            "retrieval": metadata.get("retrieval_info") or {},
            "qa": qa,
        }

    @staticmethod
    def _artifact_paths(job_dir: Path) -> list[str]:
        return sorted(path.relative_to(job_dir).as_posix() for path in job_dir.rglob("*") if path.is_file())


def _validate_stage(stage: str) -> Stage:
    value = str(stage).strip().lower()
    if value not in {"structure", "qa", "all"}:
        raise ValueError("stage must be one of: structure, qa, all")
    return value  # type: ignore[return-value]


def _validate_queries(queries: Iterable[str], *, required: bool) -> list[str]:
    result = [str(query).strip() for query in queries if str(query).strip()]
    if required and not result:
        raise ValueError("At least one non-empty query is required for qa and all stages")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workbook_name(path: Path, normalized: list[dict[str, Any]]) -> str:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for item in normalized:
        if resolved == item["path"].resolve():
            return str(item["name"])
    return path.name


__all__ = ["SUPPORTED_WORKBOOK_EXTENSIONS", "Stage", "TableAgentService"]
