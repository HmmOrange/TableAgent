from __future__ import annotations

import re
import unicodedata
from typing import Any

from TableAgent.utils import range_to_a1


def question_header_hints(env: Any, question: str, table_ids: list[str]) -> str:
    """Return exact label matches so agents do not drift to neighboring columns."""
    normalized_question = _normalize(question)
    matches = []
    for table_id in table_ids:
        structure = env.get_table_structure(table_id) or {}
        for header in _walk_headers(structure.get("headers") or []):
            label = str(getattr(header, "label", "") or "").strip()
            header_id = str(getattr(header, "id", "") or "").strip()
            if not label or not header_id:
                continue
            normalized_label = _normalize(label)
            if normalized_label and _contains_phrase(normalized_question, normalized_label):
                header_range = getattr(header, "header_range", None)
                data_range = getattr(header, "data_range", None)
                # Carry the ranges the way group hints do, so a matched header can be read
                # without a second get_header round trip.
                matches.append(
                    f"- table_id={table_id}; header_id={header_id}; label={label}; "
                    f"orientation={getattr(header, 'orientation', '')}; "
                    f"header_range={range_to_a1(header_range) if header_range else None}; "
                    f"data_range={range_to_a1(data_range) if data_range else None}"
                )
    return "\n".join(dict.fromkeys(matches)) or "No exact header-label match."


def _walk_headers(headers: list[Any]):
    for header in headers:
        yield header
        yield from _walk_headers(list(getattr(header, "sub_headers", []) or []))


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def _contains_phrase(question: str, label: str) -> bool:
    return f" {label} " in f" {question} "
