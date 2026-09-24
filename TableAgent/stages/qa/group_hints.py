from __future__ import annotations

from typing import Any

from TableAgent.stages.qa.header_hints import _contains_phrase, _normalize
from TableAgent.utils import range_to_a1


#: Tokens that carry no topical signal. Length alone does not exclude them -- "and" is
#: three characters, and on its own it was enough to surface an unrelated section as a
#: lead while the section the question actually named scored zero.
_STOPWORDS = frozenset({
    "all", "also", "and", "any", "are", "both", "but", "did", "does", "each", "for",
    "from", "had", "has", "have", "how", "into", "its", "many", "more", "most", "much",
    "not", "one", "only", "out", "over", "per", "some", "than", "that", "the", "their",
    "then", "there", "these", "they", "this", "those", "two", "use", "was", "were",
    "what", "when", "where", "which", "who", "whose", "with", "within", "would",
})


def _content_tokens(text: str) -> set[str]:
    """Tokens worth matching on; short and common words make unrelated sections look confident."""
    return {
        token
        for token in text.split()
        if len(token) >= 3 and token not in _STOPWORDS
    }


def question_group_hints(
    env: Any,
    question: str,
    table_ids: list[str],
    *,
    limit: int = 8,
) -> str:
    """Rank the structure groups a question plausibly refers to.

    An exact label match is reported as authoritative. A partial match is reported
    separately and explicitly marked, because a group label often carries footnote
    markers or wording the question never repeats verbatim -- requiring the exact
    phrase left roughly four out of five questions with no hint at all.
    """
    normalized_question = _normalize(question)
    question_tokens = _content_tokens(normalized_question)
    exact: list[str] = []
    partial: list[tuple[int, str]] = []
    seen: set[tuple[str, str]] = set()

    for table_id in table_ids:
        structure = env.get_table_structure(table_id) or {}
        for group in structure.get("groups") or []:
            label = str(getattr(group, "label", "") or "").strip()
            group_id = str(getattr(group, "id", "") or "").strip()
            key = (table_id, group_id)
            if not label or not group_id or key in seen:
                continue
            normalized_label = _normalize(label)
            description = _normalize(str(getattr(group, "description", "") or ""))
            overlap = max(
                len(question_tokens & _content_tokens(normalized_label)),
                len(question_tokens & _content_tokens(description)),
            )
            is_exact = _contains_phrase(normalized_question, normalized_label)
            if not is_exact and overlap < 1:
                continue
            seen.add(key)
            group_range = getattr(group, "group_range", None)
            data_range = getattr(group, "data_range", None)
            line = (
                f"- table_id={table_id}; group_id={group_id}; label={label}; "
                f"axis={getattr(group, 'axis', '')}; "
                f"group_range={range_to_a1(group_range) if group_range else None}; "
                f"data_range={range_to_a1(data_range) if data_range else None}"
            )
            if is_exact:
                exact.append(line)
            else:
                partial.append((overlap, line))

    sections: list[str] = []
    if exact:
        sections.append("Exact label matches (authoritative):")
        sections.extend(exact[:limit])
    remaining = limit - len(exact)
    if partial and remaining > 0:
        partial.sort(key=lambda item: item[0], reverse=True)
        sections.append("Partial label matches (verify before relying on them):")
        sections.extend(line for _, line in partial[:remaining])
    return "\n".join(sections) or "No group matched this question."
