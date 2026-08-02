from __future__ import annotations

import hashlib
import json
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from TableAgent.artifacts.retrieval_cards import write_workbook_retrieval_cards
from TableAgent.pipeline.retrieval.cards import (
    build_metadata_retrieval_card,
    build_sheet_metadata_payload,
    build_table_retrieval_cards,
)
from TableAgent.structure.verification.checks import verify_structure


RELATION_CATEGORIES = {
    "normal_formulas",
    "aggregate_formulas",
    "cell_formulas",
    "invalid_formulas",
}


def rebuild_artifacts(
    *,
    workbook_path: Path,
    workbook_name: str,
    workbook_sha256: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Validate patched structures and deterministically rebuild retrieval artifacts."""
    structures = result.get("structures")
    if not isinstance(structures, list) or not structures:
        raise ValueError("TableAgent result has no structures to rebuild")

    existing_records = [
        record
        for record in result.get("retrieval_records", [])
        if isinstance(record, dict)
    ]
    existing_by_id = {
        str(record.get("id") or ""): record
        for record in existing_records
        if str(record.get("id") or "")
    }
    workbook_metadata = _workbook_metadata(result, existing_records)
    schema = _existing_schema(result, existing_records)

    rebuilt_structures: list[dict[str, Any]] = []
    sheet_records: list[dict[str, Any]] = []
    seen_sheets: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="table-agent-rebuild-") as temp_text:
        temp_dir = Path(temp_text)
        for index, item in enumerate(structures, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Structure entry {index} must be an object")
            sheet_name = str(item.get("sheet") or "").strip()
            if not sheet_name:
                raise ValueError(f"Structure entry {index} is missing its sheet name")
            if sheet_name in seen_sheets:
                raise ValueError(f"Duplicate structure for sheet '{sheet_name}'")
            seen_sheets.add(sheet_name)

            structure = _parse_structure(item.get("structure"), sheet_name)
            _validate_structure_shape(structure, sheet_name)
            structure_path = temp_dir / f"{index:03d}-{_safe_name(sheet_name)}.yaml"
            structure_path.write_text(
                yaml.safe_dump(structure, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            report = verify_structure(workbook_path, sheet_name, structure_path)
            repaired_text = str(report.get("repaired_structure_yaml") or "").strip()
            if str(report.get("status") or "").lower() != "good" or not repaired_text:
                feedback = str(report.get("feedback") or "Structure verification failed")
                raise ValueError(f"Sheet '{sheet_name}' is invalid: {feedback}")
            repaired = yaml.safe_load(repaired_text)
            if not isinstance(repaired, dict):
                raise ValueError(f"Sheet '{sheet_name}' produced an invalid repaired structure")
            _validate_structure_shape(repaired, sheet_name)
            canonical_text = yaml.safe_dump(
                repaired,
                allow_unicode=True,
                sort_keys=False,
            ).strip()

            rebuilt_item = dict(item)
            rebuilt_item.update(
                {
                    "workbook": str(item.get("workbook") or workbook_name),
                    "sheet": sheet_name,
                    "status": "good",
                    "verification_status": "good",
                    "structure": canonical_text,
                    "verification": {
                        key: value
                        for key, value in report.items()
                        if key != "repaired_structure_yaml"
                    },
                }
            )
            rebuilt_structures.append(rebuilt_item)

            existing_metadata_record = existing_by_id.get(
                f"{workbook_name}:{sheet_name}:metadata",
                {},
            )
            existing_metadata = existing_metadata_record.get("metadata")
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}
            sheet_text = str(existing_metadata.get("preview") or "")
            metadata = build_sheet_metadata_payload(
                Path(workbook_name),
                sheet_name,
                canonical_text,
                sheet_text,
                existing_metadata,
            )
            sheet_records.append(
                {
                    "id": f"{workbook_name}:{sheet_name}:metadata",
                    "retrieval_type": "metadata",
                    "retrieval_level": "sheet",
                    "workbook": workbook_name,
                    "sheet": sheet_name,
                    "table_id": "",
                    "table_name": "",
                    "retrieval_card": build_metadata_retrieval_card(metadata),
                    "metadata": metadata,
                    "structure_yaml": canonical_text,
                }
            )
            for card in build_table_retrieval_cards(
                Path(workbook_name),
                sheet_name,
                canonical_text,
                sheet_text,
            ):
                table_id = str(card.get("table_id") or card.get("table_key") or "")
                sheet_records.append(
                    {
                        "id": f"{workbook_name}:{sheet_name}:{table_id}",
                        "retrieval_type": "data",
                        "retrieval_level": "table",
                        "workbook": workbook_name,
                        "sheet": sheet_name,
                        "table_id": table_id,
                        "table_name": str(card.get("table_name") or ""),
                        "retrieval_card": str(card.get("retrieval_card") or ""),
                        "metadata": {
                            "table_key": str(card.get("table_key") or ""),
                            "description": str(card.get("description") or ""),
                        },
                        "structure_yaml": str(card.get("structure_text") or canonical_text),
                    }
                )

            prior_sheet = schema.get(sheet_name)
            if not isinstance(prior_sheet, dict):
                prior_sheet = {}
            schema[sheet_name] = {
                **prior_sheet,
                "id": str(prior_sheet.get("id") or _sheet_id(sheet_name)),
                "description": str(
                    prior_sheet.get("description")
                    or f"Verified structure for worksheet {sheet_name}."
                ),
                "structure": repaired,
            }

        schema = {
            sheet_name: schema[sheet_name]
            for sheet_name in seen_sheets
            if sheet_name in schema
        }
        schema_text = yaml.safe_dump(schema, allow_unicode=True, sort_keys=False).strip()
        all_records = write_workbook_retrieval_cards(
            temp_dir / "workbook",
            workbook_name,
            sheet_records,
            include_embeddings=False,
        )

    common = _common_record_fields(existing_records)
    common.update(
        {
            "artifact_version": int(common.get("artifact_version") or 1),
            "document_name": str(common.get("document_name") or workbook_name),
            "workbook_sha256": str(common.get("workbook_sha256") or workbook_sha256),
            "schema_yaml": schema_text,
            "workbook_metadata": workbook_metadata,
        }
    )
    rebuilt_records = []
    for record in all_records:
        record_id = str(record.get("id") or "")
        merged = {
            **existing_by_id.get(record_id, {}),
            **record,
            **common,
        }
        merged.pop("embedding", None)
        rebuilt_records.append(merged)

    rebuilt = deepcopy(result)
    rebuilt["structures"] = rebuilt_structures
    rebuilt["schema_artifacts"] = [{"workbook": workbook_name, "schema": schema_text}]
    rebuilt["retrieval_records"] = rebuilt_records
    rebuilt["workbooks"] = [workbook_name]
    return rebuilt


def _parse_structure(value: Any, sheet_name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        structure = deepcopy(value)
    elif isinstance(value, str) and value.strip():
        try:
            structure = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(f"Sheet '{sheet_name}' contains invalid structure YAML") from exc
    else:
        raise ValueError(f"Sheet '{sheet_name}' has no structure")
    if not isinstance(structure, dict) or not structure:
        raise ValueError(f"Sheet '{sheet_name}' structure must be a non-empty object")
    return structure


def _validate_structure_shape(structure: dict[str, Any], sheet_name: str) -> None:
    table_ids: set[str] = set()
    table_count = 0
    for table_key, table in structure.items():
        if table_key == "relations":
            continue
        if not isinstance(table, dict):
            raise ValueError(f"Sheet '{sheet_name}' table '{table_key}' must be an object")
        table_count += 1
        table_id = str(table.get("id") or table_key).strip()
        if not table_id:
            raise ValueError(f"Sheet '{sheet_name}' table '{table_key}' has no id")
        if table_id in table_ids:
            raise ValueError(f"Sheet '{sheet_name}' has duplicate table id '{table_id}'")
        table_ids.add(table_id)
        table.setdefault("id", table_id)
        configured_sheet = str(table.get("sheet") or sheet_name).strip()
        if configured_sheet != sheet_name:
            raise ValueError(
                f"Table '{table_id}' belongs to sheet '{configured_sheet}', not '{sheet_name}'"
            )
        table["sheet"] = sheet_name
        headers = table.get("headers")
        if not isinstance(headers, list):
            raise ValueError(f"Table '{table_id}' headers must be a list")
        _validate_header_ids(headers, table_id)
    if table_count == 0:
        raise ValueError(f"Sheet '{sheet_name}' structure has no tables")
    _validate_relations(structure.get("relations"), table_ids)


def _validate_header_ids(headers: list[Any], table_id: str) -> None:
    seen: set[str] = set()

    def visit(items: list[Any]) -> None:
        for header in items:
            if not isinstance(header, dict):
                raise ValueError(f"Table '{table_id}' contains a non-object header")
            header_id = str(header.get("id") or "").strip()
            if not header_id:
                raise ValueError(f"Table '{table_id}' contains a header without an id")
            if header_id in seen:
                raise ValueError(f"Table '{table_id}' has duplicate header id '{header_id}'")
            seen.add(header_id)
            children = header.get("sub_headers", [])
            if not isinstance(children, list):
                raise ValueError(f"Header '{header_id}' sub_headers must be a list")
            visit(children)

    visit(headers)


def _validate_relations(relations: Any, table_ids: set[str]) -> None:
    if relations is None:
        return
    if not isinstance(relations, dict):
        raise ValueError("relations must be an object")
    if any(category in relations for category in RELATION_CATEGORIES):
        roots = [(None, relations)]
    else:
        roots: list[tuple[str, dict[str, Any]]] = []
        for table_id, payload in relations.items():
            if table_id not in table_ids:
                raise ValueError(f"Relations reference unknown table id '{table_id}'")
            if not isinstance(payload, dict):
                raise ValueError(f"Relations for table '{table_id}' must be an object")
            roots.append((str(table_id), payload))
    for scoped_table_id, root in roots:
        relation_ids: set[str] = set()
        for category, records in root.items():
            if category not in RELATION_CATEGORIES:
                continue
            if not isinstance(records, list):
                raise ValueError(f"Relation category '{category}' must be a list")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(f"Relation category '{category}' contains a non-object record")
                relation_id = str(record.get("id") or record.get("relation_id") or "").strip()
                if not relation_id:
                    raise ValueError(f"Relation category '{category}' contains a record without an id")
                if relation_id in relation_ids:
                    raise ValueError(f"Duplicate relation id '{relation_id}'")
                relation_ids.add(relation_id)
                target_table = str(record.get("table_id") or "").strip()
                if target_table and target_table not in table_ids:
                    raise ValueError(
                        f"Relation '{relation_id}' references unknown table id '{target_table}'"
                    )
                if scoped_table_id and target_table and target_table != scoped_table_id:
                    raise ValueError(
                        f"Relation '{relation_id}' in table '{scoped_table_id}' "
                        f"references table '{target_table}'"
                    )


def _existing_schema(result: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    schema_text = ""
    for item in result.get("schema_artifacts", []):
        if isinstance(item, dict) and str(item.get("schema") or "").strip():
            schema_text = str(item["schema"])
            break
    if not schema_text:
        schema_text = next(
            (
                str(record.get("schema_yaml") or "")
                for record in records
                if str(record.get("schema_yaml") or "").strip()
            ),
            "",
        )
    try:
        schema = yaml.safe_load(schema_text) if schema_text else {}
    except yaml.YAMLError:
        schema = {}
    return schema if isinstance(schema, dict) else {}


def _workbook_metadata(
    result: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    for record in records:
        metadata = record.get("workbook_metadata")
        if isinstance(metadata, dict):
            return deepcopy(metadata)
    for item in result.get("metadata_artifacts", []):
        if isinstance(item, dict) and isinstance(item.get("metadata"), dict):
            return deepcopy(item["metadata"])
    return {}


def _common_record_fields(records: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = {
        "id",
        "retrieval_type",
        "retrieval_level",
        "workbook",
        "sheet",
        "table_id",
        "table_name",
        "retrieval_card",
        "metadata",
        "structure_yaml",
        "text",
        "content_hash",
        "embedding",
    }
    for record in records:
        if record:
            return {key: value for key, value in record.items() if key not in excluded}
    return {}


def _sheet_id(sheet_name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", sheet_name).strip("_").lower() or "sheet"
    return f"sheet_{value}" if value[0].isdigit() else value


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return cleaned or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


__all__ = ["rebuild_artifacts"]
