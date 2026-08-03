from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


RetrievalMode = Literal["off", "auto", "lexical", "hybrid", "indexed", "perfect"]
QARoutingMode = Literal["auto", "normal", "common_info"]


@dataclass(frozen=True)
class RetrievalRoutingConfig:
    mode: RetrievalMode = "auto"
    rerank_with_llm: bool = True
    top_k: int = 8
    candidate_max_chars: int = 4000
    embedding_provider: str | None = None
    bm25_weight: float = 0.3
    embedding_weight: float = 0.7
    explicit_workbook_guard: bool = True
    explicit_sheet_guard: bool = True
    audit_top_k: int = 10
    query_type: str = "auto"
    max_batches: int = 3

    @property
    def enabled(self) -> bool:
        return self.mode not in {"off", "indexed"}

    @property
    def perfect(self) -> bool:
        return self.mode == "perfect"

    @property
    def use_embeddings(self) -> bool:
        return self.mode in {"auto", "hybrid", "indexed"}


@dataclass(frozen=True)
class QARoutingConfig:
    mode: QARoutingMode = "auto"
    common_info_enabled: bool = True
    common_info_fallback: Literal["normal", "error"] = "normal"


@dataclass(frozen=True)
class RoutingConfig:
    retrieval: RetrievalRoutingConfig
    qa: QARoutingConfig

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RoutingConfig":
        routing = _mapping(config.get("routing"), "table_agent.routing")
        retrieval = _mapping(routing.get("retrieval"), "table_agent.routing.retrieval")
        qa = _mapping(routing.get("qa"), "table_agent.routing.qa")

        retrieval_mode = _retrieval_mode(
            _legacy_retrieval_mode(config)
            if "run_retrieval" in config or "perfect_retrieval" in config
            else retrieval.get("mode", "auto")
        )
        qa_mode = _qa_mode(qa.get("mode", config.get("qa_routing_mode", "auto")))
        common_info_fallback = str(
            qa.get("common_info_fallback", config.get("qa_common_info_fallback", "normal"))
        ).strip().lower()
        if common_info_fallback not in {"normal", "error"}:
            raise ValueError(
                "table_agent.routing.qa.common_info_fallback must be normal or error"
            )

        return cls(
            retrieval=RetrievalRoutingConfig(
                mode=retrieval_mode,
                rerank_with_llm=_bool(
                    _prefer_legacy(
                        config,
                        "retrieval_rerank_with_llm",
                        retrieval.get("rerank_with_llm"),
                        True,
                    )
                ),
                top_k=_positive_int(
                    _prefer_legacy(config, "retrieval_top_k", retrieval.get("top_k"), 8),
                    "table_agent.routing.retrieval.top_k",
                ),
                candidate_max_chars=_positive_int(
                    _prefer_legacy(
                        config,
                        "retrieval_candidate_max_chars",
                        retrieval.get("candidate_max_chars"),
                        4000,
                    ),
                    "table_agent.routing.retrieval.candidate_max_chars",
                ),
                embedding_provider=_optional_text(
                    config.get(
                        "retrieval_embedding_provider",
                        retrieval.get("embedding_provider"),
                    )
                ),
                bm25_weight=_weight(
                    _prefer_legacy(
                        config,
                        "retrieval_bm25_weight",
                        retrieval.get("bm25_weight"),
                        config.get("retrieval_lexical_weight", 0.3),
                    ),
                    "table_agent.routing.retrieval.bm25_weight",
                ),
                embedding_weight=_weight(
                    _prefer_legacy(
                        config,
                        "retrieval_embedding_weight",
                        retrieval.get("embedding_weight"),
                        0.7,
                    ),
                    "table_agent.routing.retrieval.embedding_weight",
                ),
                explicit_workbook_guard=_bool(
                    retrieval.get("explicit_workbook_guard", True)
                ),
                explicit_sheet_guard=_bool(
                    retrieval.get("explicit_sheet_guard", True)
                ),
                audit_top_k=_positive_int(
                    _prefer_legacy(
                        config,
                        "retrieval_audit_top_k",
                        retrieval.get("audit_top_k"),
                        10,
                    ),
                    "table_agent.routing.retrieval.audit_top_k",
                ),
                query_type=_query_type(
                    _prefer_legacy(
                        config,
                        "retrieval_query_type",
                        retrieval.get("query_type"),
                        "auto",
                    )
                ),
                max_batches=_positive_int(
                    _prefer_legacy(
                        config,
                        "retrieval_max_batches",
                        retrieval.get("max_batches"),
                        3,
                    ),
                    "table_agent.routing.retrieval.max_batches",
                ),
            ),
            qa=QARoutingConfig(
                mode=qa_mode,
                common_info_enabled=_bool(
                    qa.get(
                        "common_info_enabled",
                        config.get("qa_common_info_enabled", True),
                    )
                ),
                common_info_fallback=common_info_fallback,  # type: ignore[arg-type]
            ),
        )


def _legacy_retrieval_mode(config: dict[str, Any]) -> str:
    if _bool(config.get("perfect_retrieval", False)):
        return "perfect"
    return "auto" if _bool(config.get("run_retrieval", True)) else "off"


def _prefer_legacy(
    config: dict[str, Any],
    legacy_key: str,
    nested_value: Any,
    default: Any,
) -> Any:
    if legacy_key in config:
        return config[legacy_key]
    return default if nested_value is None else nested_value


def _retrieval_mode(value: Any) -> RetrievalMode:
    mode = str(value).strip().lower()
    allowed = {"off", "auto", "lexical", "hybrid", "indexed", "perfect"}
    if mode not in allowed:
        raise ValueError(
            "table_agent.routing.retrieval.mode must be one of: "
            + ", ".join(sorted(allowed))
        )
    return mode  # type: ignore[return-value]


def _qa_mode(value: Any) -> QARoutingMode:
    mode = str(value).strip().lower()
    if mode not in {"auto", "normal", "common_info"}:
        raise ValueError(
            "table_agent.routing.qa.mode must be one of: auto, normal, common_info"
        )
    return mode  # type: ignore[return-value]


def _query_type(value: Any) -> str:
    query_type = str(value).strip().lower()
    if query_type not in {"auto", "data", "metadata", "both"}:
        raise ValueError(
            "table_agent.routing.retrieval.query_type must be auto, data, metadata, or both"
        )
    return query_type


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _positive_int(value: Any, path: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{path} must be a positive integer")
    return parsed


def _weight(value: Any, path: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{path} must be between 0.0 and 1.0")
    return parsed


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
