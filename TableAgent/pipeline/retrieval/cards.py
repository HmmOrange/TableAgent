from __future__ import annotations

from pathlib import Path

import yaml


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
        table_range = table_value.get("table_range") or table_value.get("range")
        if table_range:
            parts.append(f"Range: {table_range}")
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
