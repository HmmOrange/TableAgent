from __future__ import annotations

from pathlib import Path

import openpyxl
import yaml

from service.artifact_rebuild import rebuild_artifacts
from service.artifact_rebuild import _validate_structure_shape


def _workbook(path: Path) -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(["Product", "Amount"])
    sheet.append(["A", 10])
    sheet.append(["B", 20])
    workbook.save(path)
    workbook.close()
    return path


def test_rebuild_artifacts_validates_structure_and_rebuilds_cards_without_models(
    tmp_path: Path,
) -> None:
    workbook = _workbook(tmp_path / "sales.xlsx")
    structure = {
        "sales_table": {
            "id": "sales_table",
            "name": "Sales",
            "description": "Corrected sales records",
            "sheet": "Sales",
            "headers": [
                {
                    "id": "product",
                    "label": "Product",
                    "description": "Product name",
                    "orientation": "column",
                    "header_range": "A1",
                    "data_range": "A2:A3",
                    "sub_headers": [],
                },
                {
                    "id": "amount",
                    "label": "Amount",
                    "description": "Corrected sales amount",
                    "orientation": "column",
                    "header_range": "B1",
                    "data_range": "B2:B3",
                    "sub_headers": [],
                },
            ],
        },
        "relations": {
            "normal_formulas": [],
            "aggregate_formulas": [
                {
                    "id": "total_sales",
                    "table_id": "sales_table",
                    "description": "Total sales amount",
                    "range": "B4",
                    "expression": "SUM(B2:B3)",
                    "formula_example": "=SUM(B2:B3)",
                }
            ],
            "cell_formulas": [],
            "invalid_formulas": [],
        },
    }
    schema = {
        "Sales": {
            "id": "sales",
            "description": "Existing model-generated description",
            "structure": structure,
        }
    }
    result = {
        "job_id": "job-1",
        "stage": "structure",
        "workbooks": ["sales.xlsx"],
        "structures": [
            {
                "workbook": "sales.xlsx",
                "sheet": "Sales",
                "status": "good",
                "structure": yaml.safe_dump(structure, sort_keys=False),
            }
        ],
        "schema_artifacts": [
            {"workbook": "sales.xlsx", "schema": yaml.safe_dump(schema, sort_keys=False)}
        ],
        "metadata_artifacts": [
            {"workbook": "sales.xlsx", "metadata": {"name": "sales.xlsx"}}
        ],
        "retrieval_records": [
            {
                "id": "sales.xlsx:Sales:metadata",
                "retrieval_type": "metadata",
                "retrieval_level": "sheet",
                "workbook": "sales.xlsx",
                "sheet": "Sales",
                "retrieval_card": "old card",
                "metadata": {"preview": "Product Amount A 10 B 20"},
                "schema_yaml": yaml.safe_dump(schema, sort_keys=False),
                "workbook_metadata": {"name": "sales.xlsx"},
                "workbook_sha256": "source-hash",
            }
        ],
    }

    rebuilt = rebuild_artifacts(
        workbook_path=workbook,
        workbook_name="sales.xlsx",
        workbook_sha256="source-hash",
        result=result,
    )

    records = {record["id"]: record for record in rebuilt["retrieval_records"]}
    assert records["sales.xlsx:Sales:sales_table"]["structure_yaml"]
    assert "Corrected sales amount" in records["sales.xlsx:Sales:sales_table"]["retrieval_card"]
    assert "total_sales" in records["sales.xlsx:Sales:sales_table"]["retrieval_card"]
    assert records["sales.xlsx:metadata"]["retrieval_level"] == "workbook"
    rebuilt_schema = yaml.safe_load(records["sales.xlsx:metadata"]["schema_yaml"])
    assert rebuilt_schema["Sales"]["description"] == "Existing model-generated description"
    assert rebuilt["structures"][0]["verification_status"] == "good"


def test_multi_table_relations_allow_same_generated_id_per_table() -> None:
    structure = {
        "first": {
            "id": "first",
            "sheet": "Sales",
            "headers": [{"id": "amount", "sub_headers": []}],
        },
        "second": {
            "id": "second",
            "sheet": "Sales",
            "headers": [{"id": "amount_2", "sub_headers": []}],
        },
        "relations": {
            "first": {
                "normal_formulas": [{"id": "rel_repeat_row_formula_a_0"}],
            },
            "second": {
                "normal_formulas": [{"id": "rel_repeat_row_formula_a_0"}],
            },
        },
    }

    _validate_structure_shape(structure, "Sales")
