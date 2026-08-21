import re

import yaml

from .contracts import StructureUnderstanding

_FIELDS = ("headers", "groups")
_FENCE = re.compile(r"^```(?:yaml)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def parse_understanding(text: str) -> StructureUnderstanding:
    value = str(text).strip()
    fenced = _FENCE.match(value)
    if fenced:
        value = fenced.group(1).strip()
    try:
        payload = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ValueError("Understanding response must be valid YAML") from exc
    if not isinstance(payload, dict) or set(payload) != set(_FIELDS):
        raise ValueError("Understanding response must contain exactly headers and groups")

    return StructureUnderstanding(
        headers=_parse_labels(payload["headers"], "headers"),
        groups=_parse_labels(payload["groups"], "groups"),
    )


def _parse_labels(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list")
    labels: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{item_path} must be a non-empty string")
        normalized_label = " ".join(item.split())
        key = normalized_label.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(normalized_label)
    return tuple(labels)
