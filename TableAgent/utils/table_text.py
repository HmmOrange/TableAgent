from __future__ import annotations

import re


def _lexical_overlap_score(query: str, text: str) -> float:
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    if not query_words or not text_words:
        return 0.0
    return float(len(query_words & text_words))


def _rows_to_markdown_simple(rows: list[list[str]]) -> str:
    trimmed_rows = []
    for row in rows:
        end = len(row)
        while end > 0 and not str(row[end - 1]).strip():
            end -= 1
        clean_row = [str(cell).strip() for cell in row[:end]]
        if any(clean_row):
            trimmed_rows.append(clean_row)
    if not trimmed_rows:
        return ""

    width = max(len(row) for row in trimmed_rows)
    normalized = [row + [""] * (width - len(row)) for row in trimmed_rows]
    header = [_escape(cell) or f"Column {index}" for index, cell in enumerate(normalized[0], start=1)]
    body = [[_escape(cell) for cell in row] for row in normalized[1:]]
    rows_out = [header, ["---"] * width, *body]
    return "\n".join("| " + " | ".join(row) + " |" for row in rows_out)


def _escape(value: str) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|").strip()
