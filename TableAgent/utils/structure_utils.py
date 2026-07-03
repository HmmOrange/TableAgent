from __future__ import annotations
import yaml
from typing import Any, Dict, List
from TableAgent.schema.header import Header
from TableAgent.utils.excel_utils import parse_a1_range


def _parse_optional_a1_range(value: Any, sheet_name: str = ""):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return parse_a1_range(text, sheet_name)


def parse_header_dict(d: Dict[str, Any], sheet_name: str = "") -> Header:
    header_range = _parse_optional_a1_range(d.get("header_range"), sheet_name)
    data_range = _parse_optional_a1_range(d.get("data_range"), sheet_name)
    sub_headers = [parse_header_dict(sub, sheet_name) for sub in d.get("sub_headers", [])]
    return Header(
        id=str(d["id"]),
        label=str(d["label"]),
        description=str(d["description"]),
        orientation=d["orientation"],
        header_range=header_range,
        data_range=data_range,
        sub_headers=sub_headers
    )

def load_table_structures(yaml_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load table configurations from structure.yaml and parse into Header and CellRange objects.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    parsed = {}
    for table_key, table_data in data.items():
        # Prefer the exact worksheet name emitted by the layout phase.
        sheet_name = table_data.get("sheet") or table_data.get("name", table_key)
        headers = []
        for h_dict in table_data.get("headers", []):
            headers.append(parse_header_dict(h_dict, sheet_name))
        
        table_id = str(table_data.get("id") or table_key)
        parsed[table_id] = {
            "id": table_id,
            "name": table_data.get("name", table_key),
            "description": table_data.get("description", ""),
            "sheet": sheet_name,
            "headers": headers
        }
    return parsed

def flatten_headers(headers: List[Header]) -> List[Header]:
    """Recursively flatten headers to get all headers in the hierarchy."""
    flat = []
    for h in headers:
        flat.append(h)
        if h.sub_headers:
            flat.extend(flatten_headers(h.sub_headers))
    return flat

def get_leaf_headers(headers: List[Header]) -> List[Header]:
    """Recursively find all leaf headers (headers with no sub_headers)."""
    leaf = []
    for h in headers:
        if not h.sub_headers:
            leaf.append(h)
        else:
            leaf.extend(get_leaf_headers(h.sub_headers))
    return leaf
