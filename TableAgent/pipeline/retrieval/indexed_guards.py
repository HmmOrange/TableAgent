from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Any

import yaml

from TableAgent.pipeline.common import SourceCandidate


_CLOSE_SHEET_SCORE = 500.0
_EXACT_SHEET_SCORE = 1000.0


def apply_indexed_guards(
    question: str,
    data_candidates: list[SourceCandidate],
    metadata_candidates: list[SourceCandidate],
    *,
    workbook_guard_enabled: bool,
    sheet_guard_enabled: bool,
) -> tuple[list[SourceCandidate], list[SourceCandidate], dict[str, Any]]:
    trace: dict[str, Any] = {}
    if workbook_guard_enabled:
        data_candidates, metadata_candidates, workbook_trace = _guard_workbook(
            question,
            data_candidates,
            metadata_candidates,
        )
        trace["explicit_workbook_guard"] = workbook_trace
    if sheet_guard_enabled:
        data_candidates, metadata_candidates, sheet_trace = _guard_sheet(
            question,
            data_candidates,
            metadata_candidates,
        )
        trace["explicit_sheet_guard"] = sheet_trace
    return data_candidates, metadata_candidates, trace


def attach_guard_trace(candidate: SourceCandidate, guard_trace: dict[str, Any]) -> SourceCandidate:
    if not guard_trace:
        return candidate
    trace = dict(candidate.retrieval_trace[-1]) if candidate.retrieval_trace else {}
    trace.update(guard_trace)
    previous = candidate.retrieval_trace[:-1] if candidate.retrieval_trace else ()
    return replace(candidate, retrieval_trace=(*previous, trace))


def expand_metadata_selection(
    selected: SourceCandidate,
    ranked: list[SourceCandidate],
    *,
    limit: int = 3,
) -> SourceCandidate:
    if selected.retrieval_type != "metadata" or selected.retrieval_level != "sheet":
        return selected
    workbook = str(selected.workbook_path.resolve()).casefold()
    sheets: list[SourceCandidate] = []
    seen: set[str] = set()
    for candidate in ranked:
        if candidate.retrieval_type != "metadata" or candidate.retrieval_level != "sheet":
            continue
        if str(candidate.workbook_path.resolve()).casefold() != workbook:
            continue
        identity = candidate.sheet_name.casefold()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        sheets.append(candidate)
        if len(sheets) >= limit:
            break
    if len(sheets) < 2:
        return selected

    sheet_names = tuple(candidate.sheet_name for candidate in sheets)
    payload = {
        "metadata": {
            "type": "workbook",
            "workbook": selected.workbook_path.name,
            "selected_sheet_count": len(sheet_names),
            "sheets": [
                {"name": candidate.sheet_name, "summary": candidate.retrieval_card}
                for candidate in sheets
            ],
        }
    }
    combined_card = "\n\n".join(
        f"Sheet: {candidate.sheet_name}\n{candidate.retrieval_card}"
        for candidate in sheets
    )
    return replace(
        selected,
        sheet_name="; ".join(sheet_names),
        sheet_names=sheet_names,
        structure_text=yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).strip(),
        sheet_text=combined_card,
        retrieval_card=combined_card,
        retrieval_level="workbook",
        artifact_id="multi-sheet:" + ":".join(item.artifact_id for item in sheets),
    )


def _guard_workbook(
    question: str,
    data: list[SourceCandidate],
    metadata: list[SourceCandidate],
) -> tuple[list[SourceCandidate], list[SourceCandidate], dict[str, Any]]:
    candidates = [*data, *metadata]
    groups: dict[str, list[SourceCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(str(candidate.workbook_path.resolve()).casefold(), []).append(candidate)
    base = {
        "applied": False,
        "reason": "single_workbook" if len(groups) <= 1 else "not_detected",
        "candidate_count_before": len(candidates),
        "candidate_count_after": len(candidates),
    }
    if len(groups) <= 1:
        return data, metadata, base

    compact_question = _compact(question)
    exact = [
        (identity, group[0])
        for identity, group in groups.items()
        if _compact(group[0].workbook_path.stem) in compact_question
    ]
    if len(exact) == 1:
        identity, representative = exact[0]
        selected = lambda item: str(item.workbook_path.resolve()).casefold() == identity
        filtered_data = [item for item in data if selected(item)]
        filtered_metadata = [item for item in metadata if selected(item)]
        return filtered_data, filtered_metadata, {
            **base,
            "applied": True,
            "match_type": "exact_name",
            "workbook": representative.workbook_path.name,
            "reason": None,
            "candidate_count_after": len(filtered_data) + len(filtered_metadata),
        }
    if len(exact) > 1:
        return data, metadata, {**base, "reason": "ambiguous"}

    query_tokens = set(_workbook_tokens(question))
    workbook_tokens = {
        identity: set(_workbook_tokens(group[0].workbook_path.stem))
        for identity, group in groups.items()
    }
    frequencies = Counter(token for tokens in workbook_tokens.values() for token in tokens)
    matches = []
    for identity, tokens in workbook_tokens.items():
        unique = sorted(
            token
            for token in query_tokens & tokens
            if frequencies[token] == 1 and _strong_token(token)
        )
        if len(unique) >= 2 and any(character.isdigit() for token in unique for character in token):
            matches.append((len(unique), sum(map(len, unique)), identity, unique))
    matches.sort(reverse=True)
    if not matches or (len(matches) > 1 and matches[0][:2] == matches[1][:2]):
        return data, metadata, base

    _, _, identity, terms = matches[0]
    representative = groups[identity][0]

    def boost(candidate: SourceCandidate) -> SourceCandidate:
        if str(candidate.workbook_path.resolve()).casefold() != identity:
            return candidate
        return replace(candidate, workbook_reference_score=1.0)

    return [boost(item) for item in data], [boost(item) for item in metadata], {
        **base,
        "match_type": "unique_identifier_terms",
        "matched_terms": terms,
        "workbook": representative.workbook_path.name,
        "reason": "ranking_boost",
    }


def _guard_sheet(
    question: str,
    data: list[SourceCandidate],
    metadata: list[SourceCandidate],
) -> tuple[list[SourceCandidate], list[SourceCandidate], dict[str, Any]]:
    candidates = [*data, *metadata]
    grouped: dict[tuple[str, str], list[SourceCandidate]] = {}
    for candidate in candidates:
        identity = (
            str(candidate.workbook_path.resolve()).casefold(),
            _sheet_alias(candidate.sheet_name),
        )
        if identity[1]:
            grouped.setdefault(identity, []).append(candidate)
    scored = [(_sheet_score(question, group[0].sheet_name), identity, group[0]) for identity, group in grouped.items()]
    exact = [item for item in scored if item[0] >= _EXACT_SHEET_SCORE]
    close = [item for item in scored if _CLOSE_SHEET_SCORE <= item[0] < _EXACT_SHEET_SCORE]
    matches = exact or close
    base = {
        "applied": False,
        "reason": "not_detected" if not matches else "ambiguous",
        "candidate_count_before": len(candidates),
        "candidate_count_after": len(candidates),
    }
    if len(matches) != 1:
        return data, metadata, base
    score, identity, representative = matches[0]
    selected = lambda item: (
        str(item.workbook_path.resolve()).casefold(),
        _sheet_alias(item.sheet_name),
    ) == identity
    filtered_data = [item for item in data if selected(item)]
    filtered_metadata = [item for item in metadata if selected(item)]
    return filtered_data, filtered_metadata, {
        **base,
        "applied": True,
        "match_type": "exact" if exact else "close_alias",
        "score": score,
        "workbook": representative.workbook_path.name,
        "sheet": representative.sheet_name,
        "reason": None,
        "candidate_count_after": len(filtered_data) + len(filtered_metadata),
    }


def _sheet_score(question: str, sheet_name: str) -> float:
    alias = _sheet_alias(sheet_name)
    if not alias:
        return 0.0
    marker = r"(?:sheet|worksheet|tab|ph[oò]ng|시트)"
    phrases = []
    phrases.extend(match.group(1) for match in re.finditer(rf"{marker}\s+([^,;:()]+)", question, re.I))
    phrases.extend(match.group(1) for match in re.finditer(rf"([^,;:()]+?)\s+{marker}(?!\w)", question, re.I))
    score = 0.0
    for phrase in phrases:
        words = re.findall(r"\w+", phrase, flags=re.UNICODE)
        for length in range(1, min(4, len(words)) + 1):
            for selected in (words[:length], words[-length:]):
                candidate = _sheet_alias(" ".join(selected))
                similarity = SequenceMatcher(None, candidate, alias).ratio()
                if candidate == alias:
                    score = max(score, _EXACT_SHEET_SCORE + len(alias))
                elif similarity >= 0.8:
                    score = max(score, _CLOSE_SHEET_SCORE + similarity)
    return score


def _sheet_alias(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _compact(value: str) -> str:
    return "".join(character for character in _sheet_alias(value) if character.isalnum())


def _workbook_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return [
        token
        for token in re.findall(r"[^\W_]+(?:\.[^\W_]+)*", normalized, flags=re.UNICODE)
        if token not in {"xlsx", "xls", "xlsm", "csv"}
    ]


def _strong_token(token: str) -> bool:
    return len(token) >= 3 or "." in token or any(ord(character) > 127 for character in token)
