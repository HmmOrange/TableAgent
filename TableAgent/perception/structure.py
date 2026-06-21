from __future__ import annotations

import re
from typing import Any

import yaml


_YAML_FENCE = re.compile(r"```(?:yaml|yml)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_UNCERTAIN_RANGE_VALUES = {"unknown", "uncertain", "n/a", "none", "null", "?"}


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
            parsed = yaml.safe_load(candidate)
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
        parsed = yaml.safe_load(_extract_yaml_text(content))
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
        parsed = yaml.safe_load(_extract_yaml_text(text))
    except Exception:
        return False
    if not isinstance(parsed, dict) or "error" in parsed:
        return False
    headers = parsed.get("headers")
    if not isinstance(headers, list) or not headers:
        return False

    return any(not _is_placeholder_header(header) for header in headers)


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

    orientation = str(header.get("orientation") or "mixed").strip().lower()
    if orientation not in {"row", "column", "mixed"}:
        orientation = "mixed"
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
