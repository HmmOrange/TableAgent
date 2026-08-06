from pathlib import Path

import openpyxl
import yaml

from TableAgent.configs import TableAgentConfig, load_config
from TableAgent.structure.verification.checks import verify_structure
from TableAgent.structure.verification.runner import DeterministicVerifier


def _formula_workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Sheet1"
    worksheet["A1"] = "Total"
    worksheet["A2"] = "=1+1"
    workbook.save(path)
    workbook.close()


def _structure(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "table1": {
                    "id": "formula_table",
                    "name": "Formula table",
                    "sheet": "Sheet1",
                    "headers": [
                        {
                            "id": "total",
                            "label": "Total",
                            "orientation": "column",
                            "header_range": "A1",
                            "data_range": "A2",
                            "sub_headers": [],
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_structure_data_only_defaults_to_false() -> None:
    config = dict(load_config("config.example.yaml")["table_agent"])
    config.pop("structure_data_only", None)
    settings = TableAgentConfig.from_config(config)

    assert settings.structure_data_only is False


def test_formula_cells_are_visible_by_default(tmp_path: Path) -> None:
    workbook_path = tmp_path / "formula.xlsx"
    structure_path = tmp_path / "structure.yaml"
    _formula_workbook(workbook_path)
    _structure(structure_path)

    formula_result = verify_structure(workbook_path, "Sheet1", structure_path)
    cached_value_result = verify_structure(
        workbook_path,
        "Sheet1",
        structure_path,
        data_only=True,
    )

    assert formula_result["status"] == "good"
    assert cached_value_result["status"] == "not_good"
    assert "contains no visible data" in cached_value_result["feedback"]


def test_deterministic_verifier_passes_data_only_to_worker(tmp_path: Path) -> None:
    workbook_path = tmp_path / "formula.xlsx"
    _formula_workbook(workbook_path)

    formula_dir = tmp_path / "formula-view"
    formula_dir.mkdir()
    _structure(formula_dir / "structure_after.yaml")
    formula_result = DeterministicVerifier().run(
        workbook_path=workbook_path,
        sheet_name="Sheet1",
        structure_text=(formula_dir / "structure_after.yaml").read_text(encoding="utf-8"),
        iteration_dir=formula_dir,
    )

    cached_dir = tmp_path / "cached-view"
    cached_dir.mkdir()
    _structure(cached_dir / "structure_after.yaml")
    cached_result = DeterministicVerifier(data_only=True).run(
        workbook_path=workbook_path,
        sheet_name="Sheet1",
        structure_text=(cached_dir / "structure_after.yaml").read_text(encoding="utf-8"),
        iteration_dir=cached_dir,
    )

    assert formula_result.status == "good"
    assert cached_result.status == "not_good"
