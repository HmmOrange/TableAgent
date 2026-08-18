from __future__ import annotations

from typing import Any

from TableAgent.stages.qa.header_hints import _contains_phrase, _normalize
from TableAgent.utils import range_to_a1


def question_group_hints(
    env: Any,
    question: str,
    table_ids: list[str],
    *,
    limit: int = 8,
) -> str:
    normalized_question = _normalize(question)
    matches: list[str] = []
    seen: set[tuple[str, str]] = set()
    for table_id in table_ids:
        structure = env.get_table_structure(table_id) or {}
        for group in structure.get("groups") or []:
            label = str(getattr(group, "label", "") or "").strip()
            group_id = str(getattr(group, "id", "") or "").strip()
            key = (table_id, group_id)
            if not label or not group_id or key in seen:
                continue
            if _contains_phrase(normalized_question, _normalize(label)):
                seen.add(key)
                group_range = getattr(group, "group_range", None)
                data_range = getattr(group, "data_range", None)
                matches.append(
                    f"- table_id={table_id}; group_id={group_id}; label={label}; "
                    f"axis={getattr(group, 'axis', '')}; "
                    f"group_range={range_to_a1(group_range) if group_range else None}; "
                    f"data_range={range_to_a1(data_range) if data_range else None}"
                )
                if len(matches) >= limit:
                    return "\n".join(matches)
    return "\n".join(matches) or "No exact group-label match."
