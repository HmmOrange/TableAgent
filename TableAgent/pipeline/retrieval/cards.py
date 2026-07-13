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


_extract_headers_text = extract_headers_text
_build_retrieval_card = build_source_retrieval_card
