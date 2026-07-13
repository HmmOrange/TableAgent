from __future__ import annotations

from typing import Any, TypeVar

from TableAgent.perception.structure import _parse_yaml_mapping


CandidateT = TypeVar("CandidateT")


def choose_from_reranker(content: str, candidates: list[CandidateT]) -> CandidateT:
    if not candidates:
        raise ValueError("Reranking requires at least one candidate")
    parsed = _parse_yaml_mapping(content)
    try:
        selected_index = int(parsed.get("selected_index"))
    except (TypeError, ValueError):
        selected_index = -1

    if 0 <= selected_index < len(candidates):
        chosen = candidates[selected_index]
        _set_metadata(chosen, "reranker_selected_index", selected_index)
        _set_metadata(chosen, "reranker_rationale", str(parsed.get("rationale", "")))
        _set_metadata(chosen, "fallback_used", False)
        return chosen

    chosen = candidates[0]
    _set_metadata(chosen, "reranker_selected_index", None)
    _set_metadata(chosen, "reranker_rationale", "")
    _set_metadata(chosen, "fallback_used", True)
    return chosen


def _set_metadata(candidate: Any, name: str, value: Any) -> None:
    object.__setattr__(candidate, name, value)
