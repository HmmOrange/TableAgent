from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from TableAgent.environment.qa_env import QAEnvironment
from TableAgent.QA.actions.llm_code_generation import get_structure_summary, get_table_catalog_summary
from TableAgent.prompts.planner import PLANNER_SYSTEM_PROMPT
from TableAgent.prompts.react import REACT_SYSTEM_PROMPT
from TableAgent.prompts.synthesis import SYNTHESIS_SYSTEM_PROMPT


SAMPLE_WORKBOOK = Path("sample/multitab.xlsx")
SAMPLE_STRUCTURE = Path("sample/multitab_structure.yaml")


@pytest.fixture
def multitab_env():
    env = QAEnvironment(str(SAMPLE_STRUCTURE), str(SAMPLE_WORKBOOK))
    try:
        yield env
    finally:
        env.workbook.close()


def test_embedded_relations_are_not_loaded_as_a_table(multitab_env):
    assert multitab_env.operators.list_tables() == ["table1", "table2"]
    assert [relation["id"] for relation in multitab_env.operators.list_relations("table2")] == [
        "rel_salary_calc",
        "cell_bonus_emp01",
    ]


def test_find_table_routes_salary_question_to_employee_table(multitab_env):
    question = "Nếu Lương cơ bản của Trần Thị B tăng thêm 2.000.000 VND, hãy tính lại Thành tiền."

    assert multitab_env.operators.find_table(question) == ["table2"]


def test_relation_metadata_is_exposed_to_planning_and_inspection(multitab_env):
    catalog = get_table_catalog_summary(multitab_env)
    structure = get_structure_summary(multitab_env, "table2")

    assert "formula_relations: rel_salary_calc, cell_bonus_emp01" in catalog
    assert "Formula Relations:" in structure
    assert "rel_salary_calc" in structure
    assert "Thành tiền = Lương cơ bản * Hệ số KPI" in structure


def test_prompts_require_relation_backed_formula_evaluation():
    assert "calls `evaluate_formula` with the mutation" in PLANNER_SYSTEM_PROMPT
    assert "operators.evaluate_formula(...)" in REACT_SYSTEM_PROMPT
    assert "deterministic `value`" in SYNTHESIS_SYSTEM_PROMPT


def test_evaluate_formula_applies_salary_mutation_without_changing_workbook(multitab_env):
    before = openpyxl.load_workbook(SAMPLE_WORKBOOK, data_only=False)
    try:
        original_salary = before["Data_Synthesis"]["C13"].value
        original_formula = before["Data_Synthesis"]["E13"].value
    finally:
        before.close()

    result = multitab_env.operators.evaluate_formula(
        "rel_salary_calc",
        target_cell="E13",
        mutations={"C13": 18_000_000 + 2_000_000},
    )

    assert result["table_id"] == "table2"
    assert result["target_cell"] == "Data_Synthesis!E13"
    assert result["formula"] == "=C13*D13"
    assert result["value"] == pytest.approx(18_000_000)

    after = openpyxl.load_workbook(SAMPLE_WORKBOOK, data_only=False)
    try:
        assert after["Data_Synthesis"]["C13"].value == original_salary
        assert after["Data_Synthesis"]["E13"].value == original_formula
    finally:
        after.close()


def test_evaluate_formula_recursively_resolves_formula_dependencies(multitab_env):
    result = multitab_env.operators.evaluate_formula(
        "cell_bonus_emp01",
        mutations={"C11": 20_000_000},
    )

    assert result["formula"] == "=E11*0.1"
    assert result["value"] == pytest.approx(2_400_000)


def test_evaluate_formula_supports_excel_functions(multitab_env):
    multitab_env.relations.append(
        {
            "id": "rel_region_total",
            "category": "aggregate_formulas",
            "table_id": "table1",
            "description": "Use IF and SUM to total the first region row.",
            "formula": {
                "cell": "D5",
                "raw": "=IF(B5>100,SUM(B5:D5),0)",
            },
        }
    )

    result = multitab_env.operators.evaluate_formula("rel_region_total", table_id="table1")

    assert result["value"] == 450


def test_join_union_and_groupby_are_deterministic(multitab_env):
    employees = pd.DataFrame(
        {
            "emp_id": ["EMP01", "EMP02"],
            "name": ["A", "B"],
        }
    )
    salaries = pd.DataFrame(
        {
            "emp_id": ["EMP02", "EMP03"],
            "salary": [15, 18],
        }
    )

    joined = multitab_env.operators.join_tables(employees, salaries)
    assert joined.to_dict("records") == [{"emp_id": "EMP02", "name": "B", "salary": 15}]

    first_quarter = pd.DataFrame({"region": ["North"], "revenue": [100]})
    second_quarter = pd.DataFrame({"revenue": [150], "region": ["North"]})
    unioned = multitab_env.operators.union_tables(
        [first_quarter, second_quarter],
        source_column="source_table",
    )
    assert unioned.to_dict("records") == [
        {"region": "North", "revenue": 100, "source_table": "table_1"},
        {"region": "North", "revenue": 150, "source_table": "table_2"},
    ]

    grouped = multitab_env.operators.groupby(
        unioned,
        by="region",
        aggregations={"revenue": "sum"},
    )
    assert grouped.to_dict("records") == [{"region": "North", "revenue": 250}]


def test_union_rejects_incompatible_schemas(multitab_env):
    with pytest.raises(ValueError, match="incompatible schema"):
        multitab_env.operators.union_tables(
            [
                pd.DataFrame({"id": [1], "value": [2]}),
                pd.DataFrame({"id": [1], "other": [3]}),
            ]
        )
