import json

from .contracts import HeaderUnderstanding

_FIELDS = (
    "row_headers",
    "column_headers",
    "row_group_headers",
    "column_group_headers",
)


def parse_understanding(text: str) -> HeaderUnderstanding:
    value = str(text).strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
        if value.lower().startswith("json\n"):
            value = value[5:].lstrip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Understanding response must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != set(_FIELDS):
        raise ValueError("Understanding response must contain exactly the four header lists")

    lists: dict[str, tuple[str, ...]] = {}
    for field in _FIELDS:
        items = payload[field]
        if not isinstance(items, list):
            raise ValueError(f"{field} must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field} items must be non-empty strings")
            label = item.strip()
            if label not in seen:
                normalized.append(label)
                seen.add(label)
        lists[field] = tuple(normalized)
    return HeaderUnderstanding(**lists)
