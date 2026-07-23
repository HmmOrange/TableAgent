from __future__ import annotations

import csv
import asyncio
import json
import pickle
from pathlib import Path
from typing import Any, Iterable
from concurrent.futures import ThreadPoolExecutor

import yaml

from TableAgent.pipeline.retrieval.cards import (
    build_metadata_retrieval_card,
    build_sheet_metadata_payload,
    build_table_retrieval_cards,
)
from TableAgent.pipeline.retrieval.embeddings import MockEmbeddingModel

DEFAULT_RETRIEVAL_CARD_EMBEDDING_MODEL = "mock-hash-embedding"


def write_sheet_retrieval_cards(
    sheet_dir: Path,
    workbook_path: Path,
    sheet_name: str,
    *,
    embedding_client: Any | None = None,
    embedding_model: str = DEFAULT_RETRIEVAL_CARD_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """Persist sheet metadata and table data retrieval cards for ingestion artifacts."""
    metadata_path = sheet_dir / "metadata.yaml"
    structure_path = sheet_dir / "structure.yaml"
    sheet_text_path = sheet_dir / "sheet_text.txt"

    sheet_metadata = _read_yaml_mapping(metadata_path)
    structure_text = structure_path.read_text(encoding="utf-8") if structure_path.is_file() else ""
    sheet_text = sheet_text_path.read_text(encoding="utf-8") if sheet_text_path.is_file() else ""

    records: list[dict[str, Any]] = []
    metadata_payload = _strip_layout_noise(
        build_sheet_metadata_payload(
            workbook_path,
            sheet_name,
            structure_text,
            sheet_text,
            sheet_metadata,
        )
    )
    metadata_card = build_metadata_retrieval_card(metadata_payload)
    records.append(
        {
            "id": f"{workbook_path.name}:{sheet_name}:metadata",
            "retrieval_type": "metadata",
            "retrieval_level": "sheet",
            "workbook": workbook_path.name,
            "sheet": sheet_name,
            "table_id": "",
            "table_name": "",
            "retrieval_card": metadata_card,
            "metadata": metadata_payload,
        }
    )

    for table_card in build_table_retrieval_cards(workbook_path, sheet_name, structure_text, sheet_text):
        records.append(
            {
                "id": f"{workbook_path.name}:{sheet_name}:{table_card.get('table_id') or table_card.get('table_key')}",
                "retrieval_type": "data",
                "retrieval_level": "table",
                "workbook": workbook_path.name,
                "sheet": sheet_name,
                "table_id": table_card.get("table_id", ""),
                "table_name": table_card.get("table_name", ""),
                "retrieval_card": table_card.get("retrieval_card", ""),
                "metadata": {
                    "table_key": table_card.get("table_key", ""),
                    "description": table_card.get("description", ""),
                },
            }
        )

    _write_records(
        sheet_dir,
        records,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
    )
    return records


def write_workbook_retrieval_cards(
    workbook_dir: Path,
    workbook_name: str,
    sheet_records: Iterable[dict[str, Any]],
    *,
    embedding_client: Any | None = None,
    embedding_model: str = DEFAULT_RETRIEVAL_CARD_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """Persist a workbook metadata card plus all sheet/table cards."""
    records = list(sheet_records)
    sheet_metadata_records = [
        record for record in records
        if record.get("retrieval_type") == "metadata" and record.get("retrieval_level") == "sheet"
    ]
    workbook_payload = {
        "type": "workbook",
        "workbook": workbook_name,
        "description": _workbook_description(sheet_metadata_records),
        "sheets": [
            {
                "name": record.get("sheet", ""),
                **(
                    record.get("metadata", {})
                    if isinstance(record.get("metadata"), dict)
                    else {}
                ),
            }
            for record in sheet_metadata_records
        ],
    }
    workbook_payload = _strip_layout_noise(workbook_payload)
    workbook_card = build_metadata_retrieval_card(workbook_payload)
    workbook_record = {
        "id": f"{workbook_name}:metadata",
        "retrieval_type": "metadata",
        "retrieval_level": "workbook",
        "workbook": workbook_name,
        "sheet": "",
        "table_id": "",
        "table_name": "",
        "retrieval_card": workbook_card,
        "metadata": workbook_payload,
    }
    all_records = [workbook_record, *records]
    _write_records(
        workbook_dir,
        all_records,
        embedding_client=embedding_client,
        embedding_model=embedding_model,
    )
    return all_records


def _write_records(
    directory: Path,
    records: list[dict[str, Any]],
    *,
    embedding_client: Any | None,
    embedding_model: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    jsonl_path = directory / "retrieval_cards.jsonl"
    csv_path = directory / "retrieval_cards.csv"
    pickle_path = directory / "retrieval_cards.pkl"
    jsonl_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "retrieval_type",
                "retrieval_level",
                "workbook",
                "sheet",
                "table_id",
                "table_name",
                "retrieval_card",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in writer.fieldnames})
    pickle_path.write_bytes(
        pickle.dumps(
            _records_with_embeddings(
                records,
                embedding_client=embedding_client,
                embedding_model=embedding_model,
            )
        )
    )


def _records_with_embeddings(
    records: list[dict[str, Any]],
    *,
    embedding_client: Any | None,
    embedding_model: str,
) -> list[dict[str, Any]]:
    client = embedding_client or MockEmbeddingModel()
    vectors = _encode_cards(client, [str(record.get("retrieval_card") or "") for record in records])
    result: list[dict[str, Any]] = []
    for record, vector in zip(records, vectors):
        values = [float(value) for value in vector]
        result.append(
            {
                **record,
                "embedding": {
                    "model": embedding_model,
                    "dimension": len(values),
                    "values": values,
                },
            }
        )
    return result


def _encode_cards(embedding_client: Any, texts: list[str]) -> list[list[float]]:
    async def get_embeddings():
        encoder = getattr(embedding_client, "encode", None)
        if callable(encoder):
            return await encoder(texts)
        batch_encoder = getattr(embedding_client, "batch_encode", None)
        if callable(batch_encoder):
            return await batch_encoder(texts)
        raise TypeError("embedding_client must implement async encode() or batch_encode()")

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        vectors = asyncio.run(get_embeddings())
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            vectors = executor.submit(asyncio.run, get_embeddings()).result()
    return [list(vector) for vector in vectors]


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _strip_layout_noise(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_layout_noise(child)
            for key, child in value.items()
            if str(key) not in {"range", "table_range", "used_range", "merged_ranges"}
        }
    if isinstance(value, list):
        return [_strip_layout_noise(item) for item in value]
    return value


def _workbook_description(sheet_records: list[dict[str, Any]]) -> str:
    names = [str(record.get("sheet") or "") for record in sheet_records if record.get("sheet")]
    if not names:
        return ""
    return "Workbook contains sheets: " + "; ".join(names[:50])


__all__ = [
    "DEFAULT_RETRIEVAL_CARD_EMBEDDING_MODEL",
    "write_sheet_retrieval_cards",
    "write_workbook_retrieval_cards",
]
