from __future__ import annotations

import re
from typing import Any

import yaml


_YAML_FENCE = re.compile(r"```(?:yaml|yml)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_UNCERTAIN_RANGE_VALUES = {"unknown", "uncertain", "n/a", "none", "null", "?"}
_YAML_BOOLEAN_TAG = "tag:yaml.org,2002:bool"


class _Yaml12SafeLoader(yaml.SafeLoader):
    pass


_Yaml12SafeLoader.yaml_implicit_resolvers = {
    key: [(tag, pattern) for tag, pattern in resolvers if tag != _YAML_BOOLEAN_TAG]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12SafeLoader.add_implicit_resolver(
    _YAML_BOOLEAN_TAG,
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _load_yaml(content: str) -> Any:
    return yaml.load(content, Loader=_Yaml12SafeLoader)


def _extract_yaml_text(content: str) -> str:
    text = content.strip()
    fenced = _YAML_FENCE.search(text)
    return fenced.group(1).strip() if fenced else text


def extract_strict_structure(content: str) -> tuple[str, str]:
    """Return schema-only structure YAML and any discarded model text."""
    text = content.strip()
    candidates = [(match.group(1).strip(), match.span()) for match in _YAML_FENCE.finditer(text)]
    if not candidates:
        header_match = re.search(r"(?m)^headers:\s*", text)
        if header_match:
            candidates = [(text[header_match.start():].strip(), (header_match.start(), len(text)))]
        else:
            candidates = [(text, (0, len(text)))]

    for candidate, span in candidates:
        try:
            parsed = _load_yaml(candidate)
        except yaml.YAMLError:
            continue
        normalized = _normalize_structure(parsed)
        if normalized is None:
            continue
        discarded_parts = [(text[:span[0]] + "\n" + text[span[1]:]).strip()]
        extras = _structure_extras(parsed)
        if extras:
            discarded_parts.append(yaml.safe_dump(extras, sort_keys=False, allow_unicode=True).strip())
        discarded = "\n".join(part for part in discarded_parts if part)
        return yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True).strip(), discarded

    return "", text


def _parse_yaml_mapping(content: str) -> dict[str, Any]:
    try:
        parsed = _load_yaml(_extract_yaml_text(content))
    except yaml.YAMLError:
        return {"status": "not_good", "feedback": content}
    return parsed if isinstance(parsed, dict) else {"status": "not_good", "feedback": content}


def _is_valid_structure(structure_text: str | None) -> bool:
    if not structure_text or not structure_text.strip():
        return False
    text = structure_text.strip()
    if text.startswith("ERROR:"):
        return False

    lowered = text.lower()
    if any(term in lowered for term in (
        "connection error",
        "rate limit",
        "quota exceeded",
        "unauthorized",
        "api key",
        "authentication",
        "timeout",
        "endpoint offline",
    )):
        return False

    try:
        parsed = _load_yaml(_extract_yaml_text(text))
    except Exception:
        return False
    if not isinstance(parsed, dict) or "error" in parsed:
        return False
    headers = parsed.get("headers")
    if isinstance(headers, list) and headers:
        return any(not _is_placeholder_header(header) for header in headers)

    tables = _table_mappings(parsed)
    return bool(tables) and all(
        isinstance(table.get("headers"), list)
        and table["headers"]
        and any(not _is_placeholder_header(header) for header in table["headers"])
        for table in tables.values()
    )


def _is_placeholder_header(header: Any) -> bool:
    if isinstance(header, dict):
        value = str(header.get("label") or header.get("name") or "").strip()
        if not value:
            value = " ".join(str(v) for v in header.values() if isinstance(v, (str, int, float)))
    elif isinstance(header, (list, tuple)):
        value = " ".join(str(v) for v in header if v is not None)
    else:
        value = str(header).strip()

    value = value.strip().lower()
    if not value:
        return True
    return any(re.fullmatch(pattern, value) for pattern in (
        r"column\s*\d+",
        r"col\s*\d+",
        r"placeholder\s*\d*",
        r"untitled\s*\d*",
        r"header\s*\d*",
        r"field\s*\d*",
        r"attr(?:ibute)?\s*\d*",
        r"var(?:iable)?\s*\d*",
        r"val(?:ue)?\s*\d*",
        r"empty",
        r"none",
        r"null",
        r"n/a",
        r"-",
    ))


def _normalize_structure(parsed: Any) -> dict[str, Any] | None:
    if not isinstance(parsed, dict) or set(parsed) == {"error"}:
        return None
    headers = parsed.get("headers")
    if not isinstance(headers, list) or not headers:
        return None

    normalized_headers = []
    for header in headers:
        normalized = _normalize_header(header, include_sub_headers=True)
        if normalized is None:
            return None
        normalized_headers.append(normalized)
    return {"headers": normalized_headers}


def _normalize_header(header: Any, *, include_sub_headers: bool) -> dict[str, Any] | None:
    if not isinstance(header, dict):
        return None
    label = str(header.get("label") or "").strip()
    if not label:
        return None

    orientation = str(header.get("orientation") or "column").strip().lower()
    if orientation not in {"row", "column"}:
        orientation = "column"
    cell_range = header.get("range")
    if cell_range is not None:
        cell_range = str(cell_range).strip() or None
        if cell_range and cell_range.lower() in _UNCERTAIN_RANGE_VALUES:
            cell_range = None

    normalized = {
        "label": label,
        "description": str(header.get("description") or "").strip(),
        "orientation": orientation,
        "range": cell_range,
    }
    if include_sub_headers:
        sub_headers = header.get("sub_headers") or []
        if not isinstance(sub_headers, list):
            return None
        normalized_sub_headers = []
        for sub_header in sub_headers:
            normalized_sub_header = _normalize_header(sub_header, include_sub_headers=False)
            if normalized_sub_header is None:
                return None
            normalized_sub_headers.append(normalized_sub_header)
        normalized["sub_headers"] = normalized_sub_headers
    return normalized


def _structure_extras(parsed: dict[str, Any]) -> dict[str, Any]:
    extras = {key: value for key, value in parsed.items() if key != "headers"}
    header_extras = []
    for header in parsed.get("headers", []):
        if not isinstance(header, dict):
            continue
        item = {
            key: value
            for key, value in header.items()
            if key not in {"label", "description", "orientation", "range", "sub_headers"}
        }
        sub_extras = []
        for sub_header in header.get("sub_headers") or []:
            if isinstance(sub_header, dict):
                sub_extras.append({
                    key: value
                    for key, value in sub_header.items()
                    if key not in {"label", "description", "orientation", "range"}
                })
        if any(sub_extras):
            item["sub_headers"] = sub_extras
        header_extras.append(item)
    if any(header_extras):
        extras["headers"] = header_extras
    return extras


def extract_layout_structure(content: str) -> tuple[str, str, list[str], str]:
    """Parse a LayoutAgent response without persisting its control envelope."""
    text = content.strip()
    candidates = [(match.group(1).strip(), match.span()) for match in _YAML_FENCE.finditer(text)]
    if not candidates:
        candidates = [(text, (0, len(text)))]

    for candidate, span in candidates:
        try:
            parsed = _load_yaml(candidate)
        except yaml.YAMLError:
            continue
        if not isinstance(parsed, dict):
            continue

        source = parsed.get("structure") or parsed.get("updated_structure") or parsed
        normalized = _normalize_layout_structure(source)
        if normalized is None:
            continue

        directions = parsed.get("remaining_directions") or parsed.get("directions") or []
        if not isinstance(directions, list):
            directions = []
        changelog = str(parsed.get("changelog") or "").strip()
        discarded = (text[:span[0]] + "\n" + text[span[1]:]).strip()
        return (
            yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True).strip(),
            discarded,
            [str(direction).strip().lower() for direction in directions],
            changelog,
        )

    legacy, discarded = extract_strict_structure(content)
    return legacy, discarded, [], ""


def nullify_structure_ranges(structure_text: str, field_paths: list[str] | None = None) -> str:
    try:
        parsed = _load_yaml(structure_text)
    except yaml.YAMLError:
        return structure_text
    if not isinstance(parsed, dict):
        return structure_text

    paths_applied = 0
    for field_path in field_paths or []:
        if _set_range_path_to_null(parsed, field_path):
            paths_applied += 1
    return yaml.safe_dump(parsed, sort_keys=False, allow_unicode=True).strip()


def _set_range_path_to_null(structure: dict[str, Any], field_path: str) -> bool:
    tokens = []
    for name, index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", field_path):
        tokens.append(int(index) if index else name)
    if not tokens or tokens[-1] not in {"range", "header_range", "data_range"}:
        return False
    current: Any = structure
    try:
        for token in tokens[:-1]:
            current = current[token]
        if isinstance(current, dict) and tokens[-1] in current:
            current[tokens[-1]] = None
            return True
    except (KeyError, IndexError, TypeError):
        return False
    return False


def _normalize_layout_structure(parsed: Any) -> dict[str, Any] | None:
    legacy = _normalize_structure(parsed)
    if legacy is not None:
        return legacy
    if not isinstance(parsed, dict):
        return None

    tables = _table_mappings(parsed)
    if not tables:
        return None
    normalized_tables: dict[str, Any] = {}
    for key, table in tables.items():
        headers = table.get("headers")
        if not isinstance(headers, list) or not headers:
            return None
        normalized_headers = []
        used_ids: set[str] = set()
        for header in headers:
            normalized = _normalize_layout_header(header, include_sub_headers=True, used_ids=used_ids)
            if normalized is None:
                return None
            normalized_headers.append(normalized)
        normalized_tables[key] = {
            "id": str(table.get("id") or key).strip(),
            "name": str(table.get("name") or "").strip() or None,
            "description": str(table.get("description") or "").strip(),
            "sheet": str(table.get("sheet") or "").strip() or None,
            "headers": normalized_headers,
        }
    return normalized_tables


def _table_mappings(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for key, value in parsed.items():
        if not re.fullmatch(r"table\d+", str(key), flags=re.IGNORECASE):
            continue
        if isinstance(value, dict):
            tables[str(key)] = value
            continue
        if isinstance(value, list):
            merged: dict[str, Any] = {}
            for item in value:
                if isinstance(item, dict):
                    merged.update(item)
            if merged:
                tables[str(key)] = merged
    return tables


def _normalize_layout_header(
    header: Any,
    *,
    include_sub_headers: bool,
    used_ids: set[str],
) -> dict[str, Any] | None:
    if not isinstance(header, dict):
        return None
    label = str(header.get("label") or "").strip()
    if not label:
        return None
    orientation = str(header.get("orientation") or "column").strip().lower()
    if orientation not in {"row", "column"}:
        orientation = "column"

    base_id = re.sub(r"[^a-z0-9]+", "_", str(header.get("id") or label).strip().lower()).strip("_") or "header"
    header_id = base_id
    suffix = 2
    while header_id in used_ids:
        header_id = f"{base_id}_{suffix}"
        suffix += 1
    used_ids.add(header_id)

    normalized = {
        "id": header_id,
        "label": label,
        "description": str(header.get("description") or "").strip(),
        "orientation": orientation,
        "header_range": _normalize_range_value(header.get("header_range", header.get("range"))),
        "data_range": _normalize_range_value(header.get("data_range")),
    }
    if include_sub_headers:
        sub_headers = header.get("sub_headers") or []
        if not isinstance(sub_headers, list):
            return None
        normalized_sub_headers = []
        for sub_header in sub_headers:
            child = _normalize_layout_header(sub_header, include_sub_headers=False, used_ids=used_ids)
            if child is None:
                return None
            normalized_sub_headers.append(child)
        normalized["sub_headers"] = normalized_sub_headers
    return normalized


def _normalize_range_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _UNCERTAIN_RANGE_VALUES:
        return None
    return text
