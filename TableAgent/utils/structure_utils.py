from __future__ import annotations
import yaml
from typing import Any, Dict, List
from TableAgent.schema.header import Header
from TableAgent.utils.excel_utils import parse_a1_range


RELATION_CATEGORIES = (
    "normal_formulas",
    "aggregate_formulas",
    "cell_formulas",
    "invalid_formulas",
)


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
        if table_key == "relations" or not isinstance(table_data, dict):
            continue
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


def load_formula_relations(yaml_path: str) -> List[Dict[str, Any]]:
    """Load formula relations embedded in a structure file or emitted per table."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    relation_root = data.get("relations") if isinstance(data, dict) else None
    if relation_root is None:
        relation_root = data
    if not isinstance(relation_root, dict):
        return []

    if any(category in relation_root for category in RELATION_CATEGORIES):
        sources = [(None, relation_root)]
    else:
        sources = [
            (str(table_id), payload)
            for table_id, payload in relation_root.items()
            if isinstance(payload, dict)
            and any(category in payload for category in RELATION_CATEGORIES)
        ]

    relations: List[Dict[str, Any]] = []
    for table_id, payload in sources:
        for category in RELATION_CATEGORIES:
            records = payload.get(category, []) or []
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                normalized = dict(record)
                normalized["category"] = category
                if table_id is not None:
                    normalized.setdefault("table_id", table_id)
                relations.append(normalized)
    return relations

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
