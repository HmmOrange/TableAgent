from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import yaml


_YAML_FENCE = re.compile(r"```(?:yaml|yml)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)
_UNCERTAIN_RANGE_VALUES = {"unknown", "uncertain", "n/a", "none", "null", "?"}
_YAML_BOOLEAN_TAG = "tag:yaml.org,2002:bool"
_LAYOUT_STRUCTURE_KEYS = {"structure", "updated_structure"}
_FREE_TEXT_SCHEMA_FIELDS = {"name", "label", "description", "sheet", "changelog"}
_FREE_TEXT_SCALAR_LINE = re.compile(
    r"^(?P<prefix>[ \t]*(?:-[ \t]+)?(?P<key>[A-Za-z_][A-Za-z0-9_-]*):[ \t]*)"
    r"(?P<value>.*?)(?P<newline>\r?\n)?$"
)


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


def _load_yaml_with_free_text_retry(content: str) -> Any:
    try:
        return _load_yaml(content)
    except yaml.YAMLError:
        repaired = _quote_known_free_text_scalars(content)
        if repaired == content:
            raise
        return _load_yaml(repaired)


def _quote_known_free_text_scalars(content: str) -> str:
    repaired_lines = []
    for line in content.splitlines(keepends=True):
        match = _FREE_TEXT_SCALAR_LINE.match(line)
        if match is None or match.group("key") not in _FREE_TEXT_SCHEMA_FIELDS:
            repaired_lines.append(line)
            continue

        value = match.group("value").strip()
        if not value or value.startswith(('"', "'", "|", ">")) or value.lower() in {"null", "~"}:
            repaired_lines.append(line)
            continue

        repaired_lines.append(
            f'{match.group("prefix")}{json.dumps(value, ensure_ascii=False)}{match.group("newline") or ""}'
        )
    return "".join(repaired_lines)


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
            parsed = _load_yaml_with_free_text_retry(candidate)
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
            normalized_sub_header = _normalize_header(
                sub_header,
                include_sub_headers=isinstance(sub_header, dict) and "sub_headers" in sub_header,
            )
            if normalized_sub_header is None:
                return None
            normalized_sub_headers.append(normalized_sub_header)
        normalized["sub_headers"] = normalized_sub_headers
    return normalized


def _structure_extras(parsed: dict[str, Any]) -> dict[str, Any]:
    extras = {key: value for key, value in parsed.items() if key != "headers"}
    header_extras = [
        _header_extras(header)
        for header in parsed.get("headers", [])
        if isinstance(header, dict)
    ]
    if any(header_extras):
        extras["headers"] = header_extras
    return extras


def _header_extras(header: dict[str, Any]) -> dict[str, Any]:
    item = {
        key: value
        for key, value in header.items()
        if key not in {"label", "description", "orientation", "range", "sub_headers"}
    }
    sub_extras = [
        _header_extras(sub_header)
        for sub_header in header.get("sub_headers") or []
        if isinstance(sub_header, dict)
    ]
    if any(sub_extras):
        item["sub_headers"] = sub_extras
    return item


@dataclass(frozen=True)
class LayoutParseResult:
    structure_text: str
    discarded: str
    changelog: str
    preflight_errors: list[str]


def extract_layout_structure(content: str) -> tuple[str, str, str]:
    """Parse a LayoutAgent response without persisting its control envelope."""
    result = extract_layout_structure_result(content)
    return result.structure_text, result.discarded, result.changelog


def extract_layout_structure_result(content: str) -> LayoutParseResult:
    """Parse a response and report schema entries rejected during normalization."""
    text = content.strip()
    candidates = [(match.group(1).strip(), match.span()) for match in _YAML_FENCE.finditer(text)]
    if not candidates:
        candidates = [(text, (0, len(text)))]

    for candidate, span in candidates:
        try:
            parsed = _load_yaml_with_free_text_retry(candidate)
        except yaml.YAMLError:
            recovered = _recover_layout_envelope(candidate)
            if recovered is None:
                continue
            normalized, candidate_discarded, changelog, preflight_errors = recovered
            discarded = "\n".join(part for part in (
                (text[:span[0]] + "\n" + text[span[1]:]).strip(),
                candidate_discarded,
            ) if part)
            return LayoutParseResult(
                yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True).strip(),
                discarded,
                changelog,
                preflight_errors,
            )
        if not isinstance(parsed, dict):
            continue

        source = parsed.get("structure") or parsed.get("updated_structure") or parsed
        preflight_errors: list[str] = []
        normalized = _normalize_layout_structure(source, preflight_errors=preflight_errors)
        if normalized is None:
            continue

        changelog = str(parsed.get("changelog") or "").strip()
        discarded = (text[:span[0]] + "\n" + text[span[1]:]).strip()
        return LayoutParseResult(
            yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True).strip(),
            discarded,
            changelog,
            preflight_errors,
        )

    legacy, discarded = extract_strict_structure(content)
    return LayoutParseResult(legacy, discarded, "", [])


def _recover_layout_envelope(
    candidate: str,
) -> tuple[dict[str, Any], str, str, list[str]] | None:
    structure_block = _extract_top_level_block(candidate, _LAYOUT_STRUCTURE_KEYS)
    if structure_block is None:
        return None
    block, discarded = structure_block
    try:
        parsed = _load_yaml_with_free_text_retry(block)
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None

    source = parsed.get("structure") or parsed.get("updated_structure")
    preflight_errors: list[str] = []
    normalized = _normalize_layout_structure(source, preflight_errors=preflight_errors)
    if normalized is None:
        return None

    changelog = ""
    changelog_block = _extract_top_level_block(candidate, {"changelog"})
    if changelog_block is not None:
        try:
            changelog_payload = _load_yaml_with_free_text_retry(changelog_block[0])
        except yaml.YAMLError:
            changelog_payload = None
        if isinstance(changelog_payload, dict):
            changelog = str(changelog_payload.get("changelog") or "").strip()
        else:
            changelog = changelog_block[0].splitlines()[0].partition(":")[2].strip()

    return normalized, discarded, changelog, preflight_errors


def _extract_top_level_block(
    content: str,
    keys: set[str],
) -> tuple[str, str] | None:
    lines = content.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line and not line[0].isspace():
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):(?:\s|$)", line)
            if match and match.group(1) in keys:
                start = index
                break
    if start is None:
        return None

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line[0].isspace() and not line.lstrip().startswith("#"):
            end = index
            break

    block = "\n".join(lines[start:end]).strip()
    discarded = "\n".join(lines[:start] + lines[end:]).strip()
    return block, discarded


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
    if not tokens or tokens[-1] not in {"range", "header_range", "group_range", "data_range"}:
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


def _normalize_layout_structure(
    parsed: Any,
    *,
    preflight_errors: list[str] | None = None,
) -> dict[str, Any] | None:
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
            continue
        normalized_headers = []
        used_ids: set[str] = set()
        for index, header in enumerate(headers):
            normalized = _normalize_layout_header(
                header,
                include_sub_headers=True,
                used_ids=used_ids,
                path=f"{key}.headers[{index}]",
                rejected_headers=preflight_errors,
            )
            if normalized is None:
                continue
            normalized_headers.append(normalized)
        if not normalized_headers:
            continue
        groups = table.get("groups") or []
        if not isinstance(groups, list):
            _reject_layout_item(preflight_errors, f"{key}.groups", "value is not a list")
            groups = []
        normalized_groups = []
        used_group_ids: set[str] = set()
        for index, group in enumerate(groups):
            normalized_group = _normalize_layout_group(
                group,
                used_ids=used_group_ids,
                path=f"{key}.groups[{index}]",
                preflight_errors=preflight_errors,
            )
            if normalized_group is not None:
                normalized_groups.append(normalized_group)
        normalized_tables[key] = {
            "id": str(table.get("id") or key).strip(),
            "name": str(table.get("name") or "").strip() or None,
            "description": str(table.get("description") or "").strip(),
            "sheet": str(table.get("sheet") or "").strip() or None,
            "headers": normalized_headers,
            "groups": normalized_groups,
        }
    return normalized_tables or None


def _table_mappings(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for key, value in parsed.items():
        table: dict[str, Any] | None = None
        if isinstance(value, dict):
            table = value
        elif isinstance(value, list):
            merged: dict[str, Any] = {}
            for item in value:
                if isinstance(item, dict):
                    merged.update(item)
            if merged:
                table = merged
        if table is not None and isinstance(table.get("headers"), list):
            tables[str(key)] = table
    return tables


def _normalize_layout_header(
    header: Any,
    *,
    include_sub_headers: bool,
    used_ids: set[str],
    path: str,
    rejected_headers: list[str] | None,
) -> dict[str, Any] | None:
    if not isinstance(header, dict):
        _reject_layout_header(rejected_headers, path, "header is not a mapping")
        return None
    label = str(header.get("label") or "").strip()
    if not label:
        identity = ", ".join(
            f"{key}={header[key]!r}"
            for key in ("id", "header_range", "data_range")
            if header.get(key) not in (None, "")
        )
        detail = f" ({identity})" if identity else ""
        _reject_layout_header(rejected_headers, path, f"label is empty{detail}")
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
            _reject_layout_header(rejected_headers, f"{path}.sub_headers", "value is not a list")
            sub_headers = []
        normalized_sub_headers = []
        for index, sub_header in enumerate(sub_headers):
            child = _normalize_layout_header(
                sub_header,
                include_sub_headers=isinstance(sub_header, dict) and "sub_headers" in sub_header,
                used_ids=used_ids,
                path=f"{path}.sub_headers[{index}]",
                rejected_headers=rejected_headers,
            )
            if child is None:
                continue
            normalized_sub_headers.append(child)
        normalized["sub_headers"] = normalized_sub_headers
    return normalized


def _normalize_layout_group(
    group: Any,
    *,
    used_ids: set[str],
    path: str,
    preflight_errors: list[str] | None,
) -> dict[str, Any] | None:
    if not isinstance(group, dict):
        _reject_layout_item(preflight_errors, path, "group is not a mapping")
        return None
    label = str(group.get("label") or "").strip()
    if not label:
        _reject_layout_item(preflight_errors, path, "label is empty")
        return None
    axis = str(group.get("axis") or "").strip().lower()
    if axis not in {"row", "column", "region"}:
        _reject_layout_item(preflight_errors, path, f"axis must be row, column, or region: {axis or '<empty>'}")
        return None
    base_id = re.sub(r"[^a-z0-9]+", "_", str(group.get("id") or label).strip().lower()).strip("_") or "group"
    group_id = base_id
    suffix = 2
    while group_id in used_ids:
        group_id = f"{base_id}_{suffix}"
        suffix += 1
    used_ids.add(group_id)
    return {
        "id": group_id,
        "label": label,
        "group_range": _normalize_range_value(group.get("group_range")),
        "data_range": _normalize_range_value(group.get("data_range")),
        "axis": axis,
        "description": str(group.get("description") or "").strip(),
    }


def _reject_layout_item(errors: list[str] | None, path: str, reason: str) -> None:
    if errors is not None:
        errors.append(f"{path}: {reason}")


def _reject_layout_header(rejected_headers: list[str] | None, path: str, reason: str) -> None:
    if rejected_headers is None:
        return
    rejected_headers.append(
        f"{path} was rejected because {reason}. The candidate was removed; add it back only if "
        "the workbook shows a meaningful label, using that exact visible text. Otherwise leave it "
        "omitted. Preserve all accepted headers."
    )


def _normalize_range_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _UNCERTAIN_RANGE_VALUES:
        return None
    return text
