from __future__ import annotations

from pathlib import Path

import yaml


def _compact_text(value: object, *, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_headers_text(headers_list: list, *, limit: int = 40) -> list[str]:
    parts = []
    for header in headers_list:
        if len(parts) >= limit:
            break
        if not isinstance(header, dict):
            continue
        header_text = str(header.get("label", ""))
        if header.get("id"):
            header_text += f" ({header['id']})"
        if header.get("description"):
            header_text += f": {header['description']}"
        if header.get("orientation"):
            header_text += f" [{header['orientation']}]"
        parts.append(header_text)
        if isinstance(header.get("sub_headers"), list):
            parts.extend(extract_headers_text(header["sub_headers"], limit=limit - len(parts)))
    return parts


def extract_columns(headers_list: list, *, limit: int = 80, parent: str = "") -> list[dict[str, str]]:
    columns: list[dict[str, str]] = []
    for header in headers_list:
        if len(columns) >= limit:
            break
        if not isinstance(header, dict):
            continue
        label = str(header.get("label") or "")
        header_id = str(header.get("id") or "")
        path = " > ".join(part for part in (parent, label or header_id) if part)
        column = {
            "id": header_id,
            "label": label,
            "path": path,
            "description": _compact_text(header.get("description"), limit=180),
            "orientation": str(header.get("orientation") or ""),
        }
        columns.append({key: value for key, value in column.items() if value})
        sub_headers = header.get("sub_headers")
        if isinstance(sub_headers, list):
            columns.extend(extract_columns(sub_headers, limit=limit - len(columns), parent=path))
    return columns[:limit]


def _columns_text(columns: list[dict[str, str]], *, limit: int = 24) -> str:
    parts: list[str] = []
    for column in columns[:limit]:
        label = column.get("path") or column.get("label") or column.get("id") or ""
        if column.get("id") and column["id"] not in label:
            label += f" ({column['id']})"
        if column.get("description"):
            label += f": {column['description']}"
        if column.get("orientation"):
            label += f" [{column['orientation']}]"
        if label:
            parts.append(label)
    return "; ".join(parts)


def _table_summary(table_id: str, table_name: str, description: str, columns: list[dict[str, str]]) -> str:
    fragments = [table_name or table_id]
    if description:
        fragments.append(description)
    column_names = [
        column.get("path") or column.get("label") or column.get("id") or ""
        for column in columns[:16]
    ]
    column_names = [name for name in column_names if name]
    if column_names:
        fragments.append("Columns: " + "; ".join(column_names))
    return _compact_text(". ".join(fragments), limit=900)


def build_source_retrieval_card(
    workbook_path: Path,
    sheet_name: str,
    structure_text: str,
    sheet_text: str,
) -> str:
    table_parts = []
    try:
        structure_data = yaml.safe_load(structure_text)
    except Exception:
        structure_data = {}

    if isinstance(structure_data, dict):
        top_headers = structure_data.get("headers")
        if isinstance(top_headers, list):
            headers_text = "; ".join(extract_headers_text(top_headers))
            if headers_text:
                table_parts.append(f"Headers: {headers_text}")

        for table_key, table_value in structure_data.items():
            if table_key == "relations" or not isinstance(table_value, dict):
                continue
            table_id = table_value.get("id", table_key)
            table_name = table_value.get("name", "")
            table_parts.append(f"Table: {table_name or table_id} ({table_id})")
            if table_value.get("description"):
                table_parts.append(f"Description: {table_value['description']}")
            headers = table_value.get("headers", [])
            if isinstance(headers, list):
                headers_text = "; ".join(extract_headers_text(headers))
                if headers_text:
                    table_parts.append(f"Headers: {headers_text}")

    parts = [f"Workbook: {workbook_path.name}", f"Sheet: {sheet_name}", *table_parts]
    if sheet_text:
        parts.append(f"Sheet preview: {sheet_text[:500]}")
    return "\n".join(parts)


def build_table_retrieval_cards(
    workbook_path: Path,
    sheet_name: str,
    structure_text: str,
    sheet_text: str,
) -> list[dict[str, str]]:
    """Build one retrieval card and isolated structure payload per detected table."""
    try:
        structure_data = yaml.safe_load(structure_text)
    except Exception:
        structure_data = {}
    if not isinstance(structure_data, dict):
        return []

    cards: list[dict[str, str]] = []
    for table_key, table_value in structure_data.items():
        if table_key == "relations" or not isinstance(table_value, dict):
            continue
        table_id = str(table_value.get("id") or table_key)
        table_name = str(table_value.get("name") or table_id)
        description = str(table_value.get("description") or "")
        table_sheet = str(table_value.get("sheet") or sheet_name)
        parts = [
            f"Workbook: {workbook_path.name}",
            f"Sheet: {table_sheet}",
            f"Table: {table_name} ({table_id})",
        ]
        if description:
            parts.append(f"Description: {description}")
        headers = table_value.get("headers", [])
        if isinstance(headers, list):
            headers_text = "; ".join(extract_headers_text(headers))
            if headers_text:
                parts.append(f"Headers: {headers_text}")
        relations_text = _table_relations_text(structure_data.get("relations"), table_id)
        if relations_text:
            parts.append(f"Relations: {relations_text}")
        if sheet_text:
            parts.append(f"Sheet preview: {sheet_text[:500]}")
        cards.append(
            {
                "table_key": str(table_key),
                "table_id": table_id,
                "table_name": table_name,
                "description": description,
                "retrieval_card": "\n".join(parts),
                "structure_text": _single_table_structure_text(
                    structure_data,
                    str(table_key),
                    table_id,
                ),
            }
        )
    return cards


def build_sheet_metadata_payload(
    workbook_path: Path,
    sheet_name: str,
    structure_text: str,
    sheet_text: str,
    sheet_metadata: dict | None = None,
) -> dict:
    try:
        structure_data = yaml.safe_load(structure_text)
    except Exception:
        structure_data = {}
    sheet_metadata = sheet_metadata if isinstance(sheet_metadata, dict) else {}
    payload = {
        "type": "sheet",
        "workbook": workbook_path.name,
        "sheet": sheet_name,
        "description": "",
        "used_range": sheet_metadata.get("used_range", ""),
        "merged_ranges": sheet_metadata.get("merged_ranges", [])
        if isinstance(sheet_metadata.get("merged_ranges"), list)
        else [],
        "sheet_summary": "",
        "preview": _compact_text(sheet_text, limit=900),
        "tables": [],
    }
    if not isinstance(structure_data, dict):
        payload["description"] = _compact_text(sheet_text, limit=500)
        payload["sheet_summary"] = payload["description"]
        return payload

    top_headers = structure_data.get("headers")
    if isinstance(top_headers, list):
        payload["headers"] = extract_headers_text(top_headers)
        payload["columns"] = extract_columns(top_headers)

    for table_key, table_value in structure_data.items():
        if table_key == "relations" or not isinstance(table_value, dict):
            continue
        table_id = str(table_value.get("id") or table_key)
        table_name = str(table_value.get("name") or table_id)
        description = _compact_text(table_value.get("description"), limit=500)
        headers = table_value.get("headers", [])
        columns = extract_columns(headers) if isinstance(headers, list) else []
        table_payload = {
            "key": str(table_key),
            "id": table_id,
            "name": table_name,
            "description": description,
            "summary": "",
        }
        table_range = table_value.get("table_range") or table_value.get("range")
        if table_range:
            table_payload["range"] = str(table_range)
        if isinstance(headers, list):
            table_payload["headers"] = extract_headers_text(headers)
            table_payload["columns"] = columns
            table_payload["column_count"] = len(columns)
        table_payload["summary"] = _table_summary(table_id, table_name, description, columns)
        payload["tables"].append(table_payload)
    if not payload["description"]:
        names = [table.get("name", "") for table in payload["tables"] if table.get("name")]
        payload["description"] = "; ".join(names[:10]) or _compact_text(sheet_text, limit=500)
    table_summaries = [
        table.get("summary", "")
        for table in payload["tables"]
        if isinstance(table, dict) and table.get("summary")
    ]
    payload["sheet_summary"] = _compact_text(
        f"Sheet {sheet_name}. " + " ".join(table_summaries[:8]),
        limit=1600,
    )
    return payload


def build_metadata_retrieval_card(payload: dict) -> str:
    parts = [
        f"Metadata type: {payload.get('type', '')}",
        f"Workbook: {payload.get('workbook', '')}",
    ]
    if payload.get("sheet"):
        parts.append(f"Sheet: {payload['sheet']}")
    if payload.get("description"):
        parts.append(f"Description: {payload['description']}")
    if payload.get("sheet_summary"):
        parts.append(f"Sheet summary: {payload['sheet_summary']}")
    if payload.get("preview"):
        parts.append(f"Preview: {payload['preview']}")
    sheets = payload.get("sheets")
    if isinstance(sheets, list):
        sheet_names = [str(sheet.get("name") or "") for sheet in sheets if isinstance(sheet, dict)]
        parts.append(f"Sheets: {'; '.join(sheet_names)}")
        for sheet in sheets[:20]:
            if not isinstance(sheet, dict):
                continue
            tables = sheet.get("tables") if isinstance(sheet.get("tables"), list) else []
            table_names = [
                str(table.get("name") or table.get("id") or "")
                for table in tables
                if isinstance(table, dict)
            ]
            detail = [
                f"name={sheet.get('name', '')}",
                f"description={sheet.get('description', '')}",
                f"summary={sheet.get('sheet_summary', '')}",
                f"tables={'; '.join(table_names[:20])}",
            ]
            parts.append("Sheet detail: " + " | ".join(item for item in detail if not item.endswith("=")))
            for table in tables[:10]:
                if not isinstance(table, dict):
                    continue
                fields = [
                    str(table.get("id") or ""),
                    str(table.get("name") or ""),
                    str(table.get("description") or ""),
                    str(table.get("summary") or ""),
                ]
                columns = table.get("columns") if isinstance(table.get("columns"), list) else []
                columns_text = _columns_text(columns, limit=12)
                if columns_text:
                    fields.append("Columns: " + columns_text)
                parts.append("Workbook table detail: " + " | ".join(field for field in fields if field))
    tables = payload.get("tables")
    if isinstance(tables, list):
        table_names = [
            str(table.get("name") or table.get("id") or "")
            for table in tables
            if isinstance(table, dict)
        ]
        parts.append(f"Tables: {'; '.join(table_names[:20])}")
        for table in tables[:20]:
            if not isinstance(table, dict):
                continue
            fields = [
                str(table.get("id") or ""),
                str(table.get("name") or ""),
                str(table.get("description") or ""),
                str(table.get("summary") or ""),
            ]
            columns = table.get("columns") if isinstance(table.get("columns"), list) else []
            columns_text = _columns_text(columns, limit=24)
            if columns_text:
                fields.append("Columns: " + columns_text)
            else:
                headers = table.get("headers") if isinstance(table.get("headers"), list) else []
                fields.extend(str(header) for header in headers[:12])
            parts.append("Table detail: " + " | ".join(field for field in fields if field))
    return "\n".join(parts)


def _single_table_structure_text(structure_data: dict, table_key: str, table_id: str) -> str:
    filtered = {table_key: structure_data[table_key]}
    relations = _table_relations_payload(structure_data.get("relations"), table_id)
    if relations:
        filtered["relations"] = relations
    return yaml.safe_dump(filtered, allow_unicode=True, sort_keys=False)


def _table_relations_payload(relations: object, table_id: str) -> object:
    if not isinstance(relations, dict):
        return None
    if table_id in relations:
        return {table_id: relations[table_id]}
    return relations


def _table_relations_text(relations: object, table_id: str, *, limit: int = 20) -> str:
    payload = _table_relations_payload(relations, table_id)
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    for category_value in payload.values():
        relation_groups = (
            category_value.values()
            if isinstance(category_value, dict)
            else [category_value]
        )
        for records in relation_groups:
            if not isinstance(records, list):
                continue
            for record in records:
                if len(parts) >= limit:
                    return "; ".join(parts)
                if not isinstance(record, dict):
                    continue
                relation_id = str(record.get("id") or record.get("relation_id") or "")
                description = str(record.get("description") or record.get("formula") or "")
                text = " ".join(value for value in (relation_id, description) if value)
                if text:
                    parts.append(text)
    return "; ".join(parts)


_extract_headers_text = extract_headers_text
_build_retrieval_card = build_source_retrieval_card
