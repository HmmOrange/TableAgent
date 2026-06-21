from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs import DEFAULT_CONFIG_PATH, load_config
from configs.config import deep_merge


@dataclass(frozen=True)
class TableAgentConfig:
    artifact_dir: Path
    run_artifact_dir: Path | None
    source_artifact_dir: Path | None
    repeat_dir_template: str
    max_refinement_rounds: int
    max_context_chars: int
    image_scale: float
    render_backend: str
    render_timeout_seconds: float
    generation_max_tokens: int | None
    max_image_dimension: int | None
    max_viewport_width: int
    max_viewport_height: int
    max_image_pixels: int | None
    image_tile_size: int | None
    image_tile_overlap: int
    retrieval_rerank_with_llm: bool
    retrieval_top_k: int
    retrieval_candidate_max_chars: int

    @classmethod
    def from_config(cls, override: dict[str, Any] | None = None) -> "TableAgentConfig":
        merged = table_agent_config_dict(override)
        return cls(
            artifact_dir=Path(str(_required(merged, "artifact_dir"))),
            run_artifact_dir=_optional_path(merged.get("run_artifact_dir")),
            source_artifact_dir=_optional_path(merged.get("source_artifact_dir")),
            repeat_dir_template=str(merged.get("repeat_dir_template", "repeat_{run_id}")),
            max_refinement_rounds=int(_required(merged, "max_refinement_rounds")),
            max_context_chars=int(_required(merged, "max_context_chars")),
            image_scale=float(_required(merged, "image_scale")),
            render_backend=str(merged.get("render_backend", "auto")).strip().lower(),
            render_timeout_seconds=float(_required(merged, "render_timeout_seconds")),
            generation_max_tokens=_optional_int(merged.get("generation_max_tokens")),
            max_image_dimension=_optional_int(merged.get("max_image_dimension")),
            max_viewport_width=int(_required(merged, "max_viewport_width")),
            max_viewport_height=int(_required(merged, "max_viewport_height")),
            max_image_pixels=_optional_int(merged.get("max_image_pixels")),
            image_tile_size=_optional_int(merged.get("image_tile_size")),
            image_tile_overlap=int(_required(merged, "image_tile_overlap")),
            retrieval_rerank_with_llm=_bool(_required(merged, "retrieval_rerank_with_llm")),
            retrieval_top_k=int(_required(merged, "retrieval_top_k")),
            retrieval_candidate_max_chars=int(_required(merged, "retrieval_candidate_max_chars")),
        )


TableAgentSettings = TableAgentConfig


def table_agent_config_dict(override: dict[str, Any] | None = None) -> dict[str, Any]:
    root_config = load_config(DEFAULT_CONFIG_PATH)
    base = root_config.get("table_agent", {})
    if not isinstance(base, dict):
        base = {}

    explicit = override or {}
    if "table_agent" in explicit and isinstance(explicit["table_agent"], dict):
        explicit = explicit["table_agent"]
    return deep_merge(base, explicit)


def run_scoped_table_agent_config(config: dict[str, Any], run_name: str) -> dict[str, Any]:
    agent_config = dict(config.get("table_agent", {}))
    artifact_root = Path(str(agent_config.get("artifact_root", "TableAgent")))
    run_dir_template = str(agent_config.get("run_dir_template", "{run_name}"))
    repeat_dir_template = str(agent_config.get("repeat_dir_template", "repeat_{run_id}"))
    shared_dir_name = str(agent_config.get("shared_dir_name", "shared"))

    run_artifact_dir = artifact_root / run_dir_template.format(run_name=run_name)
    agent_config.update({
        "artifact_dir": str(run_artifact_dir / repeat_dir_template.format(run_id=1)),
        "run_artifact_dir": str(run_artifact_dir),
        "source_artifact_dir": str(run_artifact_dir / shared_dir_name),
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

    agent_config = config.get("table_agent", {})
    if not isinstance(agent_config, dict):
        return requested_output_dir, Path("logs")
    if requested_output_dir == Path("outputs"):
        requested_output_dir = Path(str(agent_config.get("evaluation_output_dir", output_dir)))
    return requested_output_dir, Path(str(agent_config.get("log_dir", "logs")))


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
