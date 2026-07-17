from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TableAgentConfig:
    phase: str
    structure_cache_dir: Path
    cache_namespace: str
    layout_model_identity: str | None
    artifact_dir: Path
    run_artifact_dir: Path | None
    source_artifact_dir: Path | None
    repeat_dir_template: str
    max_refinement_rounds: int
    max_context_chars: int
    image_scale: float
    render_backend: str
    render_timeout_seconds: float
    libreoffice_path: Path | None
    libreoffice_image_resolution: int
    workbook_show_coordinates: bool
    generation_max_tokens: int | None
    max_image_dimension: int | None
    max_viewport_width: int
    max_viewport_height: int
    max_image_pixels: int | None
    image_tile_size: int | None
    image_tile_overlap: int
    run_retrieval: bool
    retrieval_rerank_with_llm: bool
    retrieval_top_k: int
    retrieval_candidate_max_chars: int
    exstruct_mode: str
    viewport_rows: int
    viewport_columns: int
    shift_cells: int
    max_retry: int
    qa_max_retries: int
    qa_max_experience_records: int
    qa_log_path: Path | None
    qa_max_observation_chars: int
    qa_max_error_chars: int
    qa_max_value_repr_chars: int
    retrieval_embedding_provider: str | None
    retrieval_lexical_weight: float
    retrieval_embedding_weight: float
    retrieval_entity_weight: float
    retrieval_audit_top_k: int


    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "TableAgentConfig":
        merged = table_agent_config_dict(config)
        return cls(
            phase=_phase(merged.get("phase", "all")),
            structure_cache_dir=Path(str(merged.get("structure_cache_dir", "cache/table_agent/structure"))),
            cache_namespace=str(merged.get("cache_namespace", "default")),
            layout_model_identity=(str(merged["layout_model_identity"]) if merged.get("layout_model_identity") else None),
            artifact_dir=Path(str(_required(merged, "artifact_dir"))),
            run_artifact_dir=_optional_path(merged.get("run_artifact_dir")),
            source_artifact_dir=_optional_path(merged.get("source_artifact_dir")),
            repeat_dir_template=str(merged.get("repeat_dir_template", "repeat_{run_id}")),
            max_refinement_rounds=int(_required(merged, "max_refinement_rounds")),
            max_context_chars=int(_required(merged, "max_context_chars")),
            image_scale=float(_required(merged, "image_scale")),
            render_backend=str(merged.get("render_backend", "auto")).strip().lower(),
            render_timeout_seconds=float(_required(merged, "render_timeout_seconds")),
            libreoffice_path=_optional_path(merged.get("libreoffice_path")),
            libreoffice_image_resolution=int(
                merged.get("libreoffice_image_resolution", merged.get("aspose_image_resolution", 192))
            ),
            workbook_show_coordinates=_bool(merged.get("workbook_show_coordinates", True)),
            generation_max_tokens=_optional_int(merged.get("generation_max_tokens")),
            max_image_dimension=_optional_int(merged.get("max_image_dimension")),
            max_viewport_width=int(_required(merged, "max_viewport_width")),
            max_viewport_height=int(_required(merged, "max_viewport_height")),
            max_image_pixels=_optional_int(merged.get("max_image_pixels")),
            image_tile_size=_optional_int(merged.get("image_tile_size")),
            image_tile_overlap=int(_required(merged, "image_tile_overlap")),
            run_retrieval=_bool(merged.get("run_retrieval", True)),
            retrieval_rerank_with_llm=_bool(_required(merged, "retrieval_rerank_with_llm")),
            retrieval_top_k=int(_required(merged, "retrieval_top_k")),
            retrieval_candidate_max_chars=int(_required(merged, "retrieval_candidate_max_chars")),
            exstruct_mode=str(merged.get("exstruct_mode", "light")),
            viewport_rows=int(merged.get("viewport_rows", 20)),
            viewport_columns=int(merged.get("viewport_columns", 20)),
            shift_cells=int(merged.get("shift_cells", 15)),
            max_retry=int(merged.get("max_retry", 3)),
            qa_max_retries=int(merged.get("qa_max_retries", 3)),
            qa_max_experience_records=int(merged.get("qa_max_experience_records", 5)),
            qa_log_path=_optional_path(merged.get("qa_log_path")),
            qa_max_observation_chars=int(merged.get("qa_max_observation_chars", 2000)),
            qa_max_error_chars=int(merged.get("qa_max_error_chars", 2000)),
            qa_max_value_repr_chars=int(merged.get("qa_max_value_repr_chars", 800)),
            retrieval_embedding_provider=merged.get("retrieval_embedding_provider"),
            retrieval_lexical_weight=float(merged.get("retrieval_lexical_weight", 0.5)),
            retrieval_embedding_weight=float(merged.get("retrieval_embedding_weight", 0.5)),
            retrieval_entity_weight=float(merged.get("retrieval_entity_weight", 2.0)),
            retrieval_audit_top_k=int(merged.get("retrieval_audit_top_k", 10)),
        )


TableAgentSettings = TableAgentConfig


def table_agent_config_dict(override: dict[str, Any] | None = None) -> dict[str, Any]:
    explicit = override or {}
    if "table_agent" in explicit and isinstance(explicit["table_agent"], dict):
        explicit = explicit["table_agent"]
    return dict(explicit)


def run_scoped_table_agent_config(config: dict[str, Any], run_name: str) -> dict[str, Any]:
    agent_config = dict(config.get("table_agent", {}))
    vlm_config = config.get("vlm", {}) if isinstance(config.get("vlm"), dict) else {}
    agent_config.setdefault(
        "layout_model_identity",
        agent_config.get("layout_vlm_provider") or agent_config.get("vlm_provider") or vlm_config.get("provider"),
    )
    artifact_root = Path(str(agent_config.get("artifact_root", "TableAgent")))
    run_dir_template = str(agent_config.get("run_dir_template", "{run_name}"))
    repeat_dir_template = str(agent_config.get("repeat_dir_template", "repeat_{run_id}"))
    run_artifact_dir = artifact_root / run_dir_template.format(run_name=run_name)
    structure_cache_dir = Path(str(agent_config.get("structure_cache_dir", "cache/table_agent/structure")))
    agent_config.update({
        "artifact_dir": str(run_artifact_dir / repeat_dir_template.format(run_id=1)),
        "run_artifact_dir": str(run_artifact_dir),
        "source_artifact_dir": str(structure_cache_dir / "v1" / "prepared"),
        "repeat_dir_template": repeat_dir_template,
    })
    return agent_config


def resolve_table_agent_run_roots(
    pipeline_name: str,
    output_dir: str | Path,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    requested_output_dir = Path(output_dir)
    if pipeline_name != "table_agent":
        return requested_output_dir, Path("logs")
    if requested_output_dir == Path("outputs"):
        requested_output_dir = Path("TableAgent") / "outputs"
    return requested_output_dir, requested_output_dir


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing table_agent.{key}; configure it in ./config.yaml")
    return config[key]


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_path(value: Any) -> Path | None:
    return None if value is None else Path(str(value))


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _phase(value: Any) -> str:
    phase = str(value).strip().lower()
    if phase not in {"structure", "qa", "all"}:
        raise ValueError("table_agent.phase must be one of: structure, qa, all")
    return phase
