from __future__ import annotations
import os
import collections
import datetime
import json
import re
import subprocess
import sys
import pytest
from pathlib import Path

from TableAgent.domain import AxisSelection, Cell, CellRange, Header
from TableAgent.stages.qa.experience import ExperienceRecord
from TableAgent.utils import (
    col_name_to_num,
    col_num_to_name,
    parse_a1_cell,
    parse_a1_range,
    cell_to_a1,
    range_to_a1,
    load_table_structures,
)
from TableAgent.stages.qa.environment.qa_env import QAEnvironment
from TableAgent.stages.qa import TableQARunner
from TableAgent.stages.qa.runner import TokenCountingLLM
from TableAgent.llm import LLMResponse
from tests.mock_policy import MockActionPolicy
from TableAgent.stages.qa.agents import TableQAPlanner, TableQAAgent
from TableAgent.stages.qa.actions.write_plan import parse_planner_output

# Setup paths
STRUCTURE_PATH = "sample/structure.yaml"
WORKBOOK_PATH = "sample/QA_sample.xlsx"


def _llm_json(data: dict) -> str:
    return "```json\n" + json.dumps(data) + "\n```"


def _two_step_plan_json() -> str:
    return _llm_json({
        "subtasks": [
            {
                "id": "inspect_fields",
                "layer": "inspect",
                "depends_on": [],
                "description": "Inspect the fields needed to answer the question.",
            },
            {
                "id": "synthesize_answer",
                "layer": "synthesis",
                "depends_on": ["inspect_fields"],
                "description": "Use inspected variables to compute final_answer.",
            },
        ],
    })


def test_a1_conversions():
    # column name to number
    assert col_name_to_num("A") == 1
    assert col_name_to_num("Z") == 26
    assert col_name_to_num("AA") == 27
    assert col_name_to_num("AZ") == 52

    # column number to name
    assert col_num_to_name(1) == "A"
    assert col_num_to_name(26) == "Z"
    assert col_num_to_name(27) == "AA"

    # parse cell
    assert parse_a1_cell("B2") == (2, 2)
    assert parse_a1_cell("AA15") == (15, 27)

    # cell to A1
    assert cell_to_a1(2, 2) == "B2"
    assert cell_to_a1(15, 27) == "AA15"

    # parse range
    rng = parse_a1_range("B2:D22", "Sheet1")
    assert rng.start_row == 2
    assert rng.start_col == 2
    assert rng.end_row == 22
    assert rng.end_col == 4
    assert rng.sheet == "Sheet1"

    # single cell range
    rng_single = parse_a1_range("A1")
    assert rng_single.start_row == 1
    assert rng_single.start_col == 1
    assert rng_single.end_row == 1
    assert rng_single.end_col == 1
    
    assert range_to_a1(rng) == "B2:D22"
    assert range_to_a1(rng_single) == "A1"

def test_range_operations():
    # Intersection
    r1 = CellRange(1, 1, 10, 10)
    r2 = CellRange(5, 5, 15, 15)
    intersect = r1.intersection(r2)
    assert intersect == CellRange(5, 5, 10, 10)

    # Crossing (different orientation)
    # Column G (col 7, rows 3 to 22)
    col_g = CellRange(3, 7, 22, 7)
    # Row 5 (row 5, cols 1 to 10)
    row_5 = CellRange(5, 1, 5, 10)
    crossing = col_g.intersection(row_5)
    # The crossing cell should be G5 (row 5, col 7)
    assert crossing == CellRange(5, 7, 5, 7)

    # Union (adjacent columns sharing same row bounds)
    c1 = CellRange(3, 2, 22, 2) # Col B
    c2 = CellRange(3, 3, 22, 3) # Col C
    union_res = c1.union(c2)
    assert len(union_res) == 1
    assert union_res[0] == CellRange(3, 2, 22, 3) # B3:C22

    # Difference (columns same row bounds)
    r_all = CellRange(3, 2, 22, 4) # B3:D22
    r_sub = CellRange(3, 3, 22, 3) # C3:C22
    diff_res = r_all.difference(r_sub)
    assert len(diff_res) == 2
    assert CellRange(3, 2, 22, 2) in diff_res # Col B
    assert CellRange(3, 4, 22, 4) in diff_res # Col D

def test_load_structures():
    structures = load_table_structures(STRUCTURE_PATH)
    assert "table1" in structures
    table = structures["table1"]
    assert table["name"] == "People Nested Headers"
    
    headers = table["headers"]
    # Top-level headers: no, name, date_of, score
    assert len(headers) == 4
    assert headers[0].id == "no"
    assert headers[0].label == "No"
    
    name_hdr = next(h for h in headers if h.id == "name")
    assert name_hdr.orientation == "column_group"
    assert len(name_hdr.sub_headers) == 3
    assert name_hdr.sub_headers[0].id == "first_name"
    assert name_hdr.sub_headers[0].orientation == "column"


def test_load_structures_allows_null_ranges(tmp_path: Path):
    import openpyxl
    import yaml

    workbook_path = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "Metric"
    ws["B1"] = "Value"
    ws["A2"] = "Revenue"
    ws["B2"] = 100
    wb.save(workbook_path)

    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(
        yaml.safe_dump(
            {
                "table1": {
                    "id": "table1",
                    "name": "Table 1",
                    "description": "Contains one valid header and one null-range header.",
                    "sheet": "Sheet1",
                    "headers": [
                        {
                            "id": "metric",
                            "label": "Metric",
                            "description": "Metric name",
                            "orientation": "column",
                            "header_range": "A1",
                            "data_range": "A2:A2",
                            "sub_headers": [],
                        },
                        {
                            "id": "missing",
                            "label": "Missing",
                            "description": "Unverified sparse field",
                            "orientation": "column",
                            "header_range": None,
                            "data_range": None,
                            "sub_headers": [],
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    env = QAEnvironment(str(structure_path), str(workbook_path))
    headers = env.operators.list_headers("table1")

    assert headers[1].header_range is None
    assert headers[1].data_range is None
    assert env.operators.read_table_as_dataframe("table1", has_headers=False).shape == (2, 1)

def test_environment_and_operators():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    
    # Check structures
    assert env.get_table_structure("table1") is not None
    
    # Check sheets
    active_sheet_name = env.get_active_sheet_name()
    assert env.get_sheet(active_sheet_name) is not None

    # Test operators list and find headers
    headers = env.operators.list_headers("table1")
    # no (1), name (1), first (1), middle (1), last (1), date_of (1), birth (1), admission (1), score (1) -> total 9 headers!
    # Let's count leaf headers: no, first, middle, last, birth, admission, score -> total 7 leaf headers.
    # Total headers = 9. Let's verify:
    assert len(headers) == 9

    # Find headers
    score_hdrs = env.operators.find_headers("table1", "score")
    assert len(score_hdrs) > 0
    assert score_hdrs[0].id == "score"

    # Read range
    val = env.operators.read_range("B3:C4")
    assert val == [["Ha", "Minh"], ["An", "Gia"]]

    # Read range flat
    flat_val = env.operators.read_range_flat("B3:C4")
    assert flat_val == ["Ha", "Minh", "An", "Gia"]

    # Basic stats
    scores = env.operators.read_range_flat("G3:G22")
    assert len(scores) == 20
    numeric_scores = [score for score in scores if isinstance(score, (int, float))]
    assert sum(numeric_scores) > 0
    assert 60 <= sum(numeric_scores) / len(numeric_scores) <= 95

def test_filter_operator_returns_axis_selection_for_row_linked_fields():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    table_id = env.default_table_id()
    last_hdr = env.operators.get_header(table_id, "last_name")
    birth_hdr = env.operators.get_header(table_id, "birth_date")
    score_hdr = env.operators.get_header(table_id, "score")

    tran_rows = env.operators.filter_values(last_hdr.data_range, contains="Trần", ignore_accents=True)
    birth_month_9_rows = env.operators.filter_values(
        birth_hdr.data_range,
        predicate=lambda value: getattr(value, "month", None) == 9,
    )
    selected_rows = env.operators.selection_intersection(tran_rows, birth_month_9_rows)
    score_ranges = env.operators.project_selection(selected_rows, score_hdr.data_range)
    scores = env.operators.read_selection(selected_rows, score_hdr.data_range)

    assert tran_rows.axis == "row"
    assert tran_rows.positions == (11, 12)
    assert birth_month_9_rows.axis == "row"
    assert selected_rows.axis == "row"
    assert selected_rows.positions == (11,)
    assert [range_to_a1(rng) for rng in score_ranges] == ["G11"]
    assert scores == [86]
    assert sum(scores) == 86

def test_filter_operator_adapts_to_column_axis_for_horizontal_ranges():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)

    selected_cols = env.operators.filter_values("B3:D3", contains="Đặng", ignore_accents=True)
    projected = env.operators.project_selection(selected_cols, "B22:D22")

    assert selected_cols == AxisSelection("col", (4,))
    assert [range_to_a1(rng) for rng in projected] == ["D22"]
    assert env.operators.read_selection(selected_cols, "B22:D22") == ["Nguyen"]

def test_planner_json_depends_on_and_runner_topological_sort():
    plan = parse_planner_output(
        """
        ```json
        {
          "subtasks": [
            {"id": "synthesis", "layer": "synthesis", "depends_on": ["join"], "description": "finish"},
            {"id": "filter", "layer": "inspect", "depends_on": [], "description": "filter rows"},
            {"id": "score", "layer": "inspect", "depends_on": [], "description": "find scores"},
            {"id": "join", "layer": "inspect", "depends_on": ["filter", "score"], "description": "read target scores"}
          ]
        }
        ```
        """
    )
    runner = TableQARunner(STRUCTURE_PATH, WORKBOOK_PATH, policy=MockActionPolicy())
    ordered = runner._topological_sort(plan)

    assert [task.id for task in ordered] == ["filter", "score", "join", "synthesis"]
    assert plan[0].depends_on == ["join"]

def test_topological_sort_invalid_planning():
    runner = TableQARunner(STRUCTURE_PATH, WORKBOOK_PATH, policy=MockActionPolicy())

    # 1. Cycle detection
    cycle_plan = parse_planner_output(
        """
        ```json
        {
          "subtasks": [
            {"id": "task_a", "layer": "inspect", "depends_on": ["task_b"], "description": "A"},
            {"id": "task_b", "layer": "inspect", "depends_on": ["task_a"], "description": "B"}
          ]
        }
        ```
        """
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        runner._topological_sort(cycle_plan)

    # 2. Unknown dependency
    unknown_dep_plan = parse_planner_output(
        """
        ```json
        {
          "subtasks": [
            {"id": "task_a", "layer": "inspect", "depends_on": ["missing_task"], "description": "A"}
          ]
        }
        ```
        """
    )
    with pytest.raises(ValueError, match="depends on unknown subtasks"):
        runner._topological_sort(unknown_dep_plan)

    # 3. Duplicate subtask IDs
    from TableAgent.stages.qa.models.subtask import SubTask
    dup_plan = [
        SubTask(id="task_a", description="A first", layer="inspect", depends_on=[]),
        SubTask(id="task_a", description="A second", layer="inspect", depends_on=[])
    ]
    with pytest.raises(ValueError, match="Duplicate subtask id"):
        runner._topological_sort(dup_plan)

def test_runner_fails_on_invalid_planning():
    # Setup FakeLLM to return a cyclic plan
    cyclic_plan_json = _llm_json({
        "subtasks": [
            {"id": "task_a", "layer": "inspect", "depends_on": ["task_b"], "description": "A"},
            {"id": "task_b", "layer": "inspect", "depends_on": ["task_a"], "description": "B"}
        ]
    })
    runner = TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=FakeLLM({"Table Structure": cyclic_plan_json}),
        policy=MockActionPolicy(),
    )
    result = runner.run("Dummy question?")
    assert not result.success
    assert "Cycle detected" in result.error


def test_operator_catalog_is_prompt_ready_without_arithmetic_helpers():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)

    catalog = env.operators.operator_catalog()

    assert "operators.list_tables()" in catalog
    assert "operators.read_range" in catalog
    assert "operators.resolve_ranges" in catalog
    assert "operators.filter_values" in catalog
    assert "ignore_accents=True" in catalog
    assert "operators.read_selection" in catalog
    assert "Write normal Python/pandas/numpy code for arithmetic" in catalog
    assert "operators.mean(" not in catalog
    assert "operators.sum(" not in catalog

def test_shared_execution_namespace():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    
    # Execute code that assigns variables
    code = (
        "scores = operators.read_range_flat('G3:G7')\n"
        "numeric_scores = [score for score in scores if isinstance(score, (int, float))]\n"
        "mean_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0\n"
    )
    output, error, success, updates = env.execute_code(code)
    
    assert success
    assert not error
    assert "scores" in updates
    assert "mean_score" in updates
    expected_scores = env.operators.read_range_flat('G3:G7')
    expected_numeric_scores = [score for score in expected_scores if isinstance(score, (int, float))]
    assert updates["mean_score"] == sum(expected_numeric_scores) / len(expected_numeric_scores)
    
    # Execute subsequent code referencing the same variables
    code2 = "final_answer = f'Mean is {mean_score:.1f}'"
    output2, error2, success2, updates2 = env.execute_code(code2)
    
    assert success2
    assert not error2
    assert env.execution_namespace.get("final_answer") == f"Mean is {updates['mean_score']:.1f}"

def test_react_loop_and_retry_self_repair():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    
    # We will simulate an error on round 1 by passing simulate_error=True
    policy = MockActionPolicy(simulate_error=True)
    agent = TableQAAgent(env, policy=policy, max_retries=3)
    
    from TableAgent.stages.qa.models.subtask import SubTask
    subtask = SubTask(id="test_subtask", description="Get scores", layer="inspect")
    
    output = agent.run_subtask(question="What is the average score?", subtask=subtask)
    
    # Subtask should succeed because of round 2 self-repair
    assert output.success
    assert output.attempt_count == 2
    assert subtask.status == "success"
    
    # Experience pool should contain 2 attempts (one failed with score 0.0, one succeeded with score 1.0)
    records = env.experience_pool.records
    assert len(records) >= 2
    assert any(r.score == 0.0 for r in records)
    assert any(r.score == 1.0 for r in records)

    # Check formatting of experience
    formatted_exp = env.experience_pool.format()
    assert '<attempt round="1" subtask="test_subtask">' in formatted_exp
    assert '<attempt round="2" subtask="test_subtask">' in formatted_exp
    assert "Error during execution:" in formatted_exp


def test_llm_call_metrics_record_tokens_duration_and_caps():
    class CappedClient:
        def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
            return LLMResponse(
                content="done",
                prompt_tokens=12,
                completion_tokens=2048,
                token_capped=True,
            )

    client = TokenCountingLLM(CappedClient())

    client.generate("question")

    assert client.token_usage() == {"prompt": 12, "completion": 2048}
    metrics = client.call_metrics()
    assert len(metrics) == 1
    assert metrics[0]["index"] == 1
    assert metrics[0]["duration_ms"] >= 0
    assert metrics[0]["prompt_tokens"] == 12
    assert metrics[0]["completion_tokens"] == 2048
    assert metrics[0]["token_capped"] is True
    assert metrics[0]["success"] is True
    assert metrics[0]["error_type"] is None


def test_runner_keeps_subtask_retries_separate_from_replans():
    runner = TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=FakeLLM({"Table Structure": _two_step_plan_json()}),
        policy=MockActionPolicy(simulate_error=True),
        config={"table_agent": {"qa_max_retries": 3, "qa_max_replans": 0}},
    )

    result = runner.run("What is the average score of all people?")

    assert result.success
    assert result.replan_count == 0
    assert result.subtask_retry_count == 2
    assert result.qa_max_retries == 3

def test_full_runner_pipeline():
    # Test 1: Average score question
    runner = TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=FakeLLM({"Table Structure": _two_step_plan_json()}),
        policy=MockActionPolicy(),
    )
    result = runner.run("What is the average score of all people?")
    
    assert result.success
    assert not result.error
    assert len(result.plan) == 2
    assert result.plan[0].layer == "inspect"
    assert result.plan[1].layer == "synthesis"
    
    # Verify average score computation:
    # G3:G22 scores are: 92, 60, 92, 64, 91, 94, 74, 83, ...
    # The average of the sample rows (20 rows) should be evaluated to a float string.
    assert result.final_answer is not None
    # Check that it parses to a valid float
    avg_val = float(result.final_answer)
    assert 75 <= avg_val <= 85

    # Test 2: Birth date question
    runner_birth = TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=FakeLLM({"Table Structure": _two_step_plan_json()}),
        policy=MockActionPolicy(),
    )
    result_birth = runner_birth.run("What is the birth date of An Gia Pham?")
    
    assert result_birth.success
    assert not result_birth.error
    assert result_birth.final_answer == "1995-02-14"


def test_runner_humanizes_header_id_in_final_answer():
    from TableAgent.stages.qa.actions.base_action import CodeGenerationRequest, CodeGenerationResult

    class HeaderIdAnswerPolicy:
        def run(self, request: CodeGenerationRequest) -> CodeGenerationResult:
            if request.layer == "inspect":
                code = "selected_column = table_df.columns[-1]"
            else:
                code = "final_answer = selected_column"
            return CodeGenerationResult(
                code=code,
                description="Simulates a dataframe operation that returns an internal header ID.",
                reasoning="Exercise the user-facing header-label boundary.",
            )

    runner = TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=FakeLLM({"Table Structure": _two_step_plan_json()}),
        policy=HeaderIdAnswerPolicy(),
    )

    result = runner.run("Which field is the last column?")

    assert result.success
    assert result.final_answer == "Score"
    assert runner.env.execution_namespace["final_answer"] == "Score"
    assert runner.env.operators.get_header("table1", "score").label == "Score"

def test_base_abstractions_usable():
    from TableAgent.stages.qa import BaseCodeGenerationAction, BaseReActAgent
    from TableAgent.stages.qa.actions.base_action import CodeGenerationRequest, CodeGenerationResult
    from TableAgent.stages.qa.models.results import AgentOutput
    from TableAgent.stages.qa.models.subtask import SubTask
    
    # 1. Test subclassing BaseCodeGenerationAction
    class CustomPolicy(BaseCodeGenerationAction):
        def run(self, request: CodeGenerationRequest) -> CodeGenerationResult:
            return CodeGenerationResult(
                code="print('hello')",
                description="custom action description",
            )

    policy = CustomPolicy()
    code, desc = policy.generate("test?", "subtask_1", "inspect", 1)
    assert code == "print('hello')"
    assert desc == "custom action description"

    # 2. Test subclassing BaseReActAgent
    class CustomAgent(BaseReActAgent):
        def run_subtask(self, question: str, subtask: SubTask) -> AgentOutput:
            code, desc = self.code_action.generate(question, subtask.id, subtask.layer, 1)
            return AgentOutput(
                subtask_id=subtask.id,
                description=desc,
                code=code,
                success=True,
                observation="done",
                namespace_updates={}
            )

    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    agent = CustomAgent(env, policy=policy)
    assert agent.env is env
    assert agent.policy is policy
    assert agent.max_retries == 3

    subtask = SubTask(id="sub_1", description="dummy", layer="inspect")
    output = agent.run_subtask("hello", subtask)
    assert output.success
    assert output.code == "print('hello')"
    assert output.observation == "done"

class FakeLLM:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls = []

    def generate(self, prompt: str, system_prompt: str = None) -> Any:
        from TableAgent.llm import LLMResponse
        self.calls.append((prompt, system_prompt))
        if "Review whether this attempt" in prompt:
            return LLMResponse(content=_llm_json({
                "accepted": True,
                "score": 1.0,
                "feedback": "Accepted.",
            }))
        for key, val in self.responses.items():
            if key in prompt:
                return LLMResponse(content=val)
        return LLMResponse(content=_llm_json({
            "reasoning": "Default code response.",
            "code": "pass",
            "description": "Default no-op code.",
        }))

def test_notebook_restrictions_and_persistence():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    
    # 1. Verify restricted imports fail
    code_bad = "import os\nprint(os.getcwd())"
    out, error, success, updates = env.execute_code(code_bad)
    assert not success
    assert "restricted" in error

    # 2. Verify allowed imports succeed
    code_good = "import math\nval = math.sqrt(16)"
    out, error, success, updates = env.execute_code(code_good)
    assert success
    assert not error
    assert updates.get("val") == 4.0

    # 3. Verify namespace persistence
    code_persisted = "val_check = val * 2"
    out2, error2, success2, updates2 = env.execute_code(code_persisted)
    assert success2
    assert updates2.get("val_check") == 8.0

def test_operators_generic():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    # Check listing headers for table1
    headers = env.operators.list_headers("table1")
    assert len(headers) > 0
    # Check listing headers for non-existent table id returns empty list (generic behavior, doesn't crash)
    headers_empty = env.operators.list_headers("non_existent_table")
    assert len(headers_empty) == 0

def test_runner_with_fake_llm_and_logging():
    # Setup FakeLLM responses
    responses = {
        # Planner prompt matches
        "Table Structure": _two_step_plan_json(),
        # Inspect prompt matches
        "Assigned Subtask:": _llm_json({
            "reasoning": "Inspect the first-name field as a small targeted check.",
            "code": "table_id = env.default_table_id()\nfirst_hdr = operators.get_header(table_id, 'first_name')\nfirsts = operators.read_range_flat(first_hdr.data_range)",
            "description": "Reads first names into the notebook namespace.",
        }),
        # Synthesis prompt matches
        "Variables in namespace:": _llm_json({
            "reasoning": "The test expects a fixed final answer.",
            "code": "final_answer = '82.5'",
            "description": "Sets the final answer for the test.",
        })


    }
    fake_llm = FakeLLM(responses)
    
    runner = TableQARunner(STRUCTURE_PATH, WORKBOOK_PATH, llm_client=fake_llm)
    result = runner.run("What is the average score?")
    
    assert result.success
    assert result.final_answer == "82.5"
    assert len(result.plan) == 2
    assert any("operators.list_headers" in (system_prompt or "") for _, system_prompt in fake_llm.calls)
    assert all("operators.mean(" not in (system_prompt or "") for _, system_prompt in fake_llm.calls)
    
    # Verify logs/events
    logs = result.logs
    assert len(logs) > 0
    
    # Check that logs contain key info: question, subtask input, generated code, observations, final answer
    log_types = [event.get("event_type") for event in logs]
    assert "run_start" in log_types
    assert "planning_start" in log_types
    assert "subtask_start" in log_types
    assert "execute_code" in log_types
    assert "subtask_complete" in log_types
    assert "run_complete" in log_types
    
    # Verify events content
    start_event = next(e for e in logs if e.get("event_type") == "run_start")
    assert start_event["question"] == "What is the average score?"
    
    complete_event = next(e for e in logs if e.get("event_type") == "run_complete")
    assert complete_event["final_answer"] == "82.5"


def test_runner_persists_per_run_artifacts(tmp_path):
    runner = TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=FakeLLM({"Table Structure": _two_step_plan_json()}),
        policy=MockActionPolicy(),
        config={"table_agent": {"qa_log_path": str(tmp_path / "qa_runner.log")}},
    )

    result = runner.run("What is the average score?")

    assert result.success
    artifacts = result.artifacts
    assert artifacts["run_dir"].startswith(str(tmp_path / "qa_runs"))
    assert os.path.exists(artifacts["events_jsonl"])
    assert os.path.exists(artifacts["plan_json"])
    assert os.path.exists(artifacts["result_json"])
    assert os.path.exists(artifacts["answer_py"])
    assert os.path.exists(artifacts["notebook_ipynb"])
    assert os.path.isdir(artifacts["cells_dir"])
    assert os.listdir(artifacts["cells_dir"])

    with open(artifacts["answer_py"], "r", encoding="utf-8") as f:
        assert "final_answer" in f.read()

    with open(artifacts["events_log"], "r", encoding="utf-8") as f:
        events_log = f.read()
    assert events_log.startswith("=" * 120 + "\n== SUMMARY")
    assert "Attempt 1 (initial plan" in events_log
    assert "-- SUBTASK 1/" in events_log
    assert "EXECUTE CODE" in events_log
    assert "LLM CALL" in events_log
    with open(artifacts["events_jsonl"], "r", encoding="utf-8") as f:
        planning = [
            json.loads(line) for line in f if '"planning_complete"' in line
        ]
    assert isinstance(planning[0]["subtasks"][0], dict)


def test_format_events_log_compacts_prompts_code_and_plan():
    from TableAgent.stages.qa.events_log import format_events_log

    base_prompt = "\n".join(f"context line {i}" for i in range(30))
    code = "x = 1\nprint(x)"
    events = [
        {"timestamp": "2026-01-01T00:00:00", "event_type": "run_start", "question": "Q?"},
        {"timestamp": "2026-01-01T00:00:00", "event_type": "planning_complete", "subtasks": [
            {"id": "inspect_a", "layer": "inspect", "category": "normal",
             "depends_on": [], "description": "Read column A."},
        ]},
        {"timestamp": "2026-01-01T00:00:01", "event_type": "subtask_start",
         "subtask_id": "inspect_a", "layer": "inspect", "category": "normal"},
        {"timestamp": "2026-01-01T00:00:01", "event_type": "generate_call",
         "system_prompt": "SYS", "prompt": base_prompt + "\nround 1"},
        {"timestamp": "2026-01-01T00:00:02", "event_type": "llm_call", "duration_ms": 1500,
         "prompt_tokens": 1000, "completion_tokens": 50, "success": True},
        {"timestamp": "2026-01-01T00:00:02", "event_type": "generate_response",
         "content": '{"code": "x = 1\\nprint(x)"}'},
        {"timestamp": "2026-01-01T00:00:02", "event_type": "generate_parsed",
         "reasoning": "why", "code": code},
        {"timestamp": "2026-01-01T00:00:03", "event_type": "execute_code",
         "cell_id": "cell_1", "success": False, "code": code,
         "stdout_preview": "1\n", "stdout_truncated": True, "stdout_chars": 2},
        {"timestamp": "2026-01-01T00:00:03", "event_type": "generate_call",
         "system_prompt": "SYS", "prompt": base_prompt + "\nround 2"},
        {"timestamp": "2026-01-01T00:00:04", "event_type": "subtask_complete",
         "subtask_id": "inspect_a", "success": True, "code": code},
        {"timestamp": "2026-01-01T00:00:05", "event_type": "run_complete",
         "success": True, "final_answer": "1", "execution_time": 5.0, "replan_count": 0},
    ]

    text = format_events_log(events)
    body = text.split("== RUN", 1)[1]

    assert "final answer: 1" in text
    assert "llm calls: 1 (generate 1)" in text
    assert "tokens: 1,000 in + 50 out = 1,050" in text
    assert "inspect_a | inspect | -          | OK" in text
    assert "1. Read column A." in text
    assert "#005 [00:00:02.000 | t=2.00s | Δ1.00s] LLM CALL  1.50s · 1,000 in / 50 out tokens" in text
    assert "response: (raw text omitted; parsed result in #007)" in text
    assert "code: (shown with execution in #008)" in text
    assert "code: (same as #008 code)" in text
    assert body.count("x = 1") == 1
    assert "system_prompt: (same as #004)" in text
    assert "prompt (diff vs #004: +1 -1 lines)" in text
    assert "│ -round 1" in text and "│ +round 2" in text
    assert "more lines (full text: events.jsonl line 4)" in text
    assert "stdout (truncated)" in text
    assert "stdout_chars" not in text


def test_llm_code_generation_repairs_invalid_json_response():
    from TableAgent.stages.qa.actions.base_action import CodeGenerationRequest
    from TableAgent.stages.qa.actions.llm_code_generation import LLMCodeGenerationAction
    from TableAgent.stages.qa.models.subtask import SubTask
    from TableAgent.llm import LLMResponse

    class RepairingLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompt: str, system_prompt: str = None) -> Any:
            self.calls.append((prompt, system_prompt))
            if "previous response for a TableAgent code-generation action was invalid" in prompt:
                return LLMResponse(content=_llm_json({
                    "reasoning": "Repair the invalid prose by producing a compact executable inspection cell.",
                    "code": "table_id = env.default_table_id()\nprint(table_id)",
                    "description": "Prints the default table id.",
                }))
            return LLMResponse(content="We need to inspect the table first. Let's list tables.")

    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    subtask = SubTask(
        id="inspect_table",
        description="Inspect the default table id.",
        layer="inspect",
        metadata={"table_id": env.default_table_id()},
    )
    action = LLMCodeGenerationAction(RepairingLLM(), env=env, output_format_retries=1)

    result = action.run(CodeGenerationRequest(
        question="What table is loaded?",
        subtask_id=subtask.id,
        layer=subtask.layer,
        round_num=1,
        subtask=subtask,
    ))

    assert "env.default_table_id()" in result.code
    event_types = [event["event_type"] for event in env.logger.events]
    assert "generate_parse_error" in event_types
    assert "generate_repair_call" in event_types
    assert "generate_repair_parsed" in event_types


def test_security_import_bypass_fails():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)

    # Proving __builtins__['__import__']('os') fails
    code_bypass = "func = __builtins__['__import__']\nfunc('os')"
    out, error, success, updates = env.execute_code(code_bypass)
    assert not success
    assert "restricted" in error or "Access to" in error

    # Proving import os fails
    code_import_os = "import os"
    out2, error2, success2, updates2 = env.execute_code(code_import_os)
    assert not success2
    assert "restricted" in error2

    # Proving getattr(__builtins__, '__import__')('os') fails
    code_getattr_bypass = "getattr(__builtins__, '__import__')('os')"
    out3, error3, success3, updates3 = env.execute_code(code_getattr_bypass)
    assert not success3
    assert "restricted" in error3 or "Access to" in error3

    # Proving no leaked variables in namespace
    assert "os" not in env.execution_namespace

def test_allowed_imports_work():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)

    # math
    out, error, success, updates = env.execute_code("import math\nres = math.sqrt(25)")
    assert success
    assert not error
    assert updates.get("res") == 5.0

    # pandas
    out, error, success, updates = env.execute_code("import pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})")
    assert success
    assert not error

    # datetime
    out, error, success, updates = env.execute_code("import datetime\nd = datetime.date(2026, 6, 26)")
    assert success
    assert not error

    # time
    out, error, success, updates = env.execute_code("import time\ntime.sleep(0.001)")
    assert success
    assert not error

def test_runner_with_non_default_table_id(tmp_path):
    # Read sample structure and replace table1 with people_table
    with open("sample/structure.yaml", "r") as f:
        struct_content = f.read()
    
    struct_content = struct_content.replace("table1:", "people_table:")
    
    struct_file = tmp_path / "structure_temp.yaml"
    with open(struct_file, "w") as f:
        f.write(struct_content)

    # Initialize runner with temp structure and check generic table_id selection
    runner = TableQARunner(
        str(struct_file),
        WORKBOOK_PATH,
        llm_client=FakeLLM({"Table Structure": _two_step_plan_json()}),
        policy=MockActionPolicy(),
    )
    assert runner.table_id is None
    
    # Run a question and make sure it plans and resolves table to "people_table"
    result = runner.run("What is the average score?")
    assert result.success
    # The default plan should resolve table_id to "people_table"
    assert result.plan[0].metadata["table_id"] == "people_table"
    assert runner.env.default_table_id() == "people_table"

def test_no_table1_fallback_in_production_code():
    import pathlib
    # Check the canonical QA-stage implementation.
    qa_dir = pathlib.Path("TableAgent/stages/qa")
    
    # Check planner.py, runner.py, code-generation action
    files_to_check = [
        qa_dir / "agents/planner.py",
        qa_dir / "runner.py",
        qa_dir / "actions/llm_code_generation.py",
        qa_dir / "actions/write_plan.py",
    ]
    
    for file_path in files_to_check:
        with open(file_path, "r") as f:
            content = f.read()
            # Assert "table1" is not hard-coded in the source code as a literal
            assert "table1" not in content, f"Production file {file_path} contains hardcoded 'table1'"
            # Assert sample header ids are not hard-coded in fallback
            if "planner.py" in str(file_path):
                assert "first_name" not in content, "planner.py contains hardcoded first_name"
                assert "middle_name" not in content, "planner.py contains hardcoded middle_name"
                assert "last_name" not in content, "planner.py contains hardcoded last_name"
                assert "birth_date" not in content, "planner.py contains hardcoded birth_date"


def test_large_output_is_compacted_but_full_output_is_recoverable():
    env = QAEnvironment(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        max_observation_chars=120,
        max_error_chars=120,
        max_value_repr_chars=120,
    )

    code = "big_text = '0123456789' * 100\nprint(big_text)"
    output, error, success, updates = env.execute_code(code, cell_id="large_output")

    assert success
    assert not error
    assert len(output) < 200
    assert "[truncated]" in output
    assert "big_text" in updates

    full_output = env.get_cell_output("large_output")
    assert len(full_output) >= 1000
    assert full_output.startswith("0123456789")

    history = env.get_history(last_n=1, max_output_len=80)
    assert "large_output" in history
    assert "[truncated]" in history


def test_notebook_records_nbformat_cells(tmp_path):
    pytest.importorskip("nbformat")
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)

    output, error, success, updates = env.execute_code("answer_preview = 42\nprint(answer_preview)", cell_id="nb_cell")

    assert success
    assert not error
    assert output.strip() == "42"
    assert env.notebook.nb is not None
    assert env.notebook.nb.cells[-1].source == "answer_preview = 42\nprint(answer_preview)"
    assert env.notebook.nb.cells[-1].metadata["cell_id"] == "nb_cell"

    notebook_path = env.export_notebook(tmp_path / "qa_run.ipynb")
    assert notebook_path.exists()


def test_variable_preview_summarizes_large_values():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH, max_value_repr_chars=200)
    code = "import pandas as pd\ndf_large = pd.DataFrame({'a': range(50), 'b': range(50, 100)})"
    output, error, success, updates = env.execute_code(code)

    assert success
    assert not error
    assert "DataFrame(shape=(50, 2)" in env.notebook.summarize_value(env.execution_namespace["df_large"])

    preview = env.preview_variable("df_large", rows=3, max_chars=300)
    assert "DataFrame(shape=(50, 2)" in preview
    assert "0" in preview
    assert len(preview) <= 330


def test_experience_format_truncates_large_observations():
    from TableAgent.stages.qa.experience import ExperiencePool, ExperienceRecord

    pool = ExperiencePool(max_records=2, max_code_chars=80, max_observation_chars=80)
    pool.add(ExperienceRecord(
        subtask_id="inspect",
        description="large observation",
        code="x = 1\n" + ("#" * 200),
        observation="obs-" + ("y" * 200),
        reasoning="I should inspect only the relevant field before reading values.",
        score=1.0,
        round=1,
    ))

    formatted = pool.format()
    assert "large observation" in formatted
    assert "<reasoning>" in formatted
    assert "inspect only the relevant field" in formatted
    assert "[truncated]" in formatted
    assert len(formatted) < 400


def test_operator_modules_have_runnable_smoke_entrypoints():
    modules = [
            "TableAgent.stages.qa.operators.base_operator",
            "TableAgent.stages.qa.operators.range_operator",
            "TableAgent.stages.qa.operators.filter_operator",
            "TableAgent.stages.qa.operators.structure_operator",
            "TableAgent.stages.qa.operators.workbook_operator",
            "TableAgent.stages.qa.operators.table_operator",
    ]

    for module in modules:
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=os.getcwd(),
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, f"{module} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert result.stdout.strip()


def test_group_header_mask_resolves_leaf_columns():
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    try:
        dataframe = env.operators.read_table_as_dataframe("table1", has_headers=True)

        child_ids = env.operators.resolve_header_columns("table1", "name")
        mask = env.operators.group_header_mask(
            dataframe,
            "table1",
            "name",
            equals="Minh",
            mode="any",
        )

        assert child_ids == ["first_name", "middle_name", "last_name"]
        assert dataframe.loc[mask, "no"].tolist() == [1, 3, 5, 8]
    finally:
        env.workbook.close()


def test_planner_extracts_trailing_json_and_common_info_metadata():
    trailing = parse_planner_output(
        "I would inspect the workbook first.\n"
        '[{"id":"inspect","layer":"inspect","depends_on":[],"description":"Read data"},'
        '{"id":"finish","layer":"synthesis","depends_on":["inspect"],"description":"Answer"}]'
    )
    descriptive_plan = parse_planner_output(_llm_json({
        "subtasks": [
            {
                "id": "inspect_sheet_info",
                "layer": "inspect",
                "category": "common_info",
                "depends_on": [],
                "description": "Describe the OIL sheet structure.",
                "metadata": {
                    "common_info_scope": "sheet",
                    "target_names": ["OIL"],
                },
            },
            {
                "id": "synthesize_sheet_info",
                "layer": "synthesis",
                "category": "common_info",
                "depends_on": ["inspect_sheet_info"],
                "description": "Return the verified sheet summary.",
            },
        ],
    }))

    assert [subtask.id for subtask in trailing] == ["inspect", "finish"]
    assert descriptive_plan[0].metadata["target_names"] == ["OIL"]
    assert descriptive_plan[0].category == "common_info"
    assert descriptive_plan[0].metadata["common_info_scope"] == "sheet"


def test_runner_serializes_full_dataframe_final_answer():
    import pandas as pd

    from TableAgent.stages.qa.models.subtask import SubTask

    runner = TableQARunner(STRUCTURE_PATH, WORKBOOK_PATH, policy=MockActionPolicy())
    try:
        rows = 68
        runner.env.execution_namespace["final_answer"] = pd.DataFrame({
            "STT": range(1, rows + 1),
            "item": [f"part-{index}" for index in range(1, rows + 1)],
        })
        synthesis = SubTask(id="synthesize_answer", description="Return all rows.", layer="synthesis")

        answer = runner._final_answer([synthesis], [synthesis])

        assert answer is not None
        assert "part-68" in answer
        assert "..." not in answer
        assert len(answer.splitlines()) == rows + 2
    finally:
        runner.close()


def test_common_info_plan_uses_verified_metadata_route(tmp_path: Path):
    plan = _llm_json({
        "subtasks": [
            {
                "id": "inspect_table_info",
                "layer": "inspect",
                "category": "common_info",
                "depends_on": [],
                "description": "Describe the verified table.",
                "metadata": {
                    "common_info_scope": "table",
                    "target_names": ["People Nested Headers"],
                },
            },
            {
                "id": "synthesize_table_info",
                "layer": "synthesis",
                "category": "common_info",
                "depends_on": ["inspect_table_info"],
                "description": "Return the verified table summary.",
            },
        ],
    })
    answer = "## Table: People Nested Headers\n- Description: Verified people table."
    llm = FakeLLM({
        "Table Structure": plan,
        "Verified structural answer:": answer,
    })
    runner = TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=llm,
        config={"qa_artifact_dir": str(tmp_path / "qa")},
    )
    try:
        result = runner.run("Describe the People Nested Headers table.")

        assert result.success
        assert result.final_answer == answer
        assert [output.layer for output in result.subtask_outputs] == ["inspect", "synthesis"]
        assert all(output.category == "common_info" for output in result.subtask_outputs)
        assert all("business data" in output.reasoning for output in result.subtask_outputs)
    finally:
        runner.close()


class UnderstandingLLM(FakeLLM):
    """Answer the understanding call and fail the first planner call to force a replan."""

    def __init__(self, fail_understanding: bool = False):
        super().__init__({})
        self.fail_understanding = fail_understanding
        self.planner_prompts = []
        self.understanding_prompts = []

    def generate(self, prompt: str, system_prompt: str = None) -> Any:
        if "Excel Workbook Content" in prompt:
            self.understanding_prompts.append(prompt)
            if self.fail_understanding:
                raise RuntimeError("understanding unavailable")
            return LLMResponse(content="Intent: average of the score column.")
        if "Table Structure Summaries" in prompt:
            self.planner_prompts.append(prompt)
            if len(self.planner_prompts) == 1:
                return LLMResponse(content="not a plan")
            return LLMResponse(content=_two_step_plan_json())
        return super().generate(prompt, system_prompt)


def _understanding_runner(llm, enabled: bool = True) -> TableQARunner:
    settings = {} if enabled else {"qa_question_understanding": False}
    return TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=llm,
        policy=MockActionPolicy(),
        config={"table_agent": settings},
    )


def test_question_understanding_runs_once_and_reaches_every_plan():
    llm = UnderstandingLLM()
    result = _understanding_runner(llm).run("What is the average score?")

    assert result.success
    assert result.replan_count == 1
    assert len(llm.understanding_prompts) == 1
    assert "What is the average score?" in llm.understanding_prompts[0]
    assert "A1:" in llm.understanding_prompts[0]
    assert len(llm.planner_prompts) == 2
    assert all(
        "Question understanding" in prompt and "Intent: average of the score column." in prompt
        for prompt in llm.planner_prompts
    )
    event = next(e for e in result.logs if e.get("event_type") == "question_understanding")
    assert event["content"] == "Intent: average of the score column."


def test_question_understanding_can_be_disabled():
    llm = UnderstandingLLM()
    result = _understanding_runner(llm, enabled=False).run("What is the average score?")

    assert result.success
    assert llm.understanding_prompts == []
    assert all("Question understanding" not in prompt for prompt in llm.planner_prompts)


def test_question_understanding_failure_falls_back_to_plain_planning():
    llm = UnderstandingLLM(fail_understanding=True)
    result = _understanding_runner(llm).run("What is the average score?")

    assert result.success
    assert len(llm.understanding_prompts) == 1
    assert all("Question understanding" not in prompt for prompt in llm.planner_prompts)
    assert any(e.get("event_type") == "question_understanding_error" for e in result.logs)


def test_workbook_preview_respects_budget_and_excluded_sheets():
    import openpyxl
    from TableAgent.stages.qa.actions.understand_question import workbook_preview

    workbook = openpyxl.Workbook()
    kept = workbook.active
    kept.title = "Data"
    for index in range(200):
        kept.append([f"row{index}", index])
    workbook.create_sheet("Hidden")["A1"] = "secret"

    preview = workbook_preview(workbook, "book.xlsx", {"hidden"}, max_chars=200)

    assert "Sheet: 'Data'" in preview
    assert "Hidden" not in preview and "secret" not in preview
    assert "| A1:row0 | B1:0 |" in preview
    assert "omitted]" in preview and "A200:row199" in preview


def test_workbook_preview_caps_wide_rows_and_skips_chart_sheets():
    import openpyxl
    from TableAgent.stages.qa.actions.understand_question import workbook_preview

    workbook = openpyxl.Workbook()
    for _ in range(10):
        workbook.active.append(["x" * 1000] * 100)
    workbook.create_chartsheet("Chart")

    body = workbook_preview(workbook, "book.xlsx", max_chars=5000)

    assert "Chart" not in body
    assert len(body) < 5000 + 200


class _PromptRecordingLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt: str, system_prompt: str = None) -> Any:
        self.prompts.append(prompt)
        return LLMResponse(content=_llm_json({
            "reasoning": "Record the prompt.",
            "code": "selected_table_ids = [env.default_table_id()]\nfinal_answer = 1",
            "description": "Recorded.",
        }))


@pytest.mark.parametrize("layer", ["table_inspect", "inspect", "synthesis"])
@pytest.mark.parametrize("round_num", [1, 2])
def test_code_generation_prompt_carries_the_planner_subtask_description(layer, round_num):
    from TableAgent.stages.qa.actions.base_action import CodeGenerationRequest
    from TableAgent.stages.qa.actions.llm_code_generation import LLMCodeGenerationAction
    from TableAgent.stages.qa.models.subtask import SubTask

    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    goal = "Read the score of Alice only; do not aggregate other rows."
    subtask = SubTask(
        id="step_under_test",
        description=goal,
        layer=layer,
        metadata={"table_id": env.default_table_id()},
    )
    llm = _PromptRecordingLLM()
    LLMCodeGenerationAction(llm, env=env).run(CodeGenerationRequest(
        question="What is Alice's score?",
        subtask_id=subtask.id,
        layer=layer,
        round_num=round_num,
        subtask=subtask,
    ))

    assert goal in llm.prompts[0]


def test_code_generation_prompt_without_a_subtask_object_still_renders():
    from TableAgent.stages.qa.actions.base_action import CodeGenerationRequest
    from TableAgent.stages.qa.actions.llm_code_generation import LLMCodeGenerationAction

    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    llm = _PromptRecordingLLM()
    LLMCodeGenerationAction(llm, env=env).run(CodeGenerationRequest(
        question="What is Alice's score?", subtask_id="step_without_object", layer="inspect", round_num=1,
    ))

    assert "step_without_object" in llm.prompts[0]
    assert "Subtask goal" not in llm.prompts[0]


def test_runner_sends_each_planner_description_to_its_code_generation_prompt():
    llm = FakeLLM({
        "Table Structure": _two_step_plan_json(),
        "Assigned Subtask:": _llm_json({
            "reasoning": "Read one field so the inspection has evidence.",
            "code": "firsts = operators.read_range_flat(operators.get_header(env.default_table_id(), 'first_name').data_range)\nprint(firsts[:3])",
            "description": "Reads first names.",
        }),
        "Variables in namespace:": _llm_json({
            "reasoning": "Set a fixed final answer.",
            "code": "final_answer = '82.5'",
            "description": "Sets the final answer.",
        }),
    })
    result = TableQARunner(STRUCTURE_PATH, WORKBOOK_PATH, llm_client=llm).run("What is the average score?")

    assert result.success
    generation_prompts = [
        event["prompt"] for event in result.logs if event.get("event_type") == "generate_call"
    ]
    assert any("Inspect the fields needed to answer the question." in prompt for prompt in generation_prompts)
    assert any("Use inspected variables to compute final_answer." in prompt for prompt in generation_prompts)


_DECISIONS = "5. **Required Operation**: Lookup the score of Alice only; keep {braces} literal."


class DecisionLLM(FakeLLM):
    """Scripted LLM that answers understanding, planning, code, and final-review prompts."""

    def __init__(self):
        super().__init__({
            "Verify coverage, exact target identity": _llm_json({
                "accepted": True, "score": 1.0, "feedback": "Accepted.",
            }),
            "Table Structure": _two_step_plan_json(),
            "Assigned Subtask:": _llm_json({
                "reasoning": "Read one field so the inspection has evidence.",
                "code": "firsts = operators.read_range_flat(operators.get_header(env.default_table_id(), 'first_name').data_range)\nprint(firsts[:3])",
                "description": "Reads first names.",
            }),
            "Variables in namespace:": _llm_json({
                "reasoning": "Set a fixed final answer.",
                "code": "final_answer = '82.5'",
                "description": "Sets the final answer.",
            }),
        })

    def generate(self, prompt: str, system_prompt: str = None) -> Any:
        if "Excel Workbook Content" in prompt:
            self.calls.append((prompt, system_prompt))
            return LLMResponse(content=_DECISIONS)
        return super().generate(prompt, system_prompt)


def _decision_run(**settings):
    runner = TableQARunner(
        STRUCTURE_PATH,
        WORKBOOK_PATH,
        llm_client=DecisionLLM(),
        config={"table_agent": {"qa_final_answer_review": True, **settings}},
    )
    result = runner.run("What is the average score?")
    assert result.success
    prompts = collections.defaultdict(list)
    for event in result.logs:
        if event.get("event_type") in {
            "planner_prompt", "generate_call", "review_prompt", "final_answer_review_prompt",
        }:
            prompts[event["event_type"]].append(event["prompt"])
    return prompts


def test_decisions_reach_code_generation_review_and_final_review():
    prompts = _decision_run()

    inspect_prompts = [p for p in prompts["generate_call"] if "Assigned Subtask:" in p]
    synthesis_prompts = [p for p in prompts["generate_call"] if "Variables in namespace:" in p]
    assert inspect_prompts and synthesis_prompts
    assert all(_DECISIONS in prompt for prompt in inspect_prompts + synthesis_prompts)
    assert prompts["review_prompt"] and all(_DECISIONS in prompt for prompt in prompts["review_prompt"])
    final_review = prompts["final_answer_review_prompt"]
    assert final_review and all(_DECISIONS in prompt for prompt in final_review)
    # Final review checks both directions: answer against decisions and decisions against the question.
    assert all("decisions themselves" in prompt for prompt in final_review)
    assert all("do not put data values or reasoning" in prompt for prompt in prompts["planner_prompt"])


def test_turning_off_decision_propagation_keeps_decisions_in_the_planner_only():
    prompts = _decision_run(qa_propagate_decisions=False)

    assert all(_DECISIONS in prompt for prompt in prompts["planner_prompt"])
    for kind in ("generate_call", "review_prompt", "final_answer_review_prompt"):
        assert prompts[kind] and not any(_DECISIONS in prompt for prompt in prompts[kind])
    assert not any("do not put data values or reasoning" in prompt for prompt in prompts["planner_prompt"])


def test_no_decision_block_without_question_understanding():
    prompts = _decision_run(qa_question_understanding=False)

    for kind in ("planner_prompt", "generate_call", "review_prompt", "final_answer_review_prompt"):
        assert prompts[kind] and not any("Question decisions" in prompt for prompt in prompts[kind])


class DecisionRetryLLM(DecisionLLM):
    """Force one replan, one failed inspect attempt, and one failed synthesis attempt."""

    def __init__(self, fail_understanding: bool = False):
        super().__init__()
        self.fail_understanding = fail_understanding
        self.seen = collections.Counter()

    def generate(self, prompt: str, system_prompt: str = None) -> Any:
        if "Excel Workbook Content" in prompt and self.fail_understanding:
            raise RuntimeError("understanding unavailable")
        for key, broken in (
            ("Table Structure", "not a plan"),
            ("Assigned Subtask:", _llm_json({"reasoning": "Fail once.", "code": "raise ValueError('boom')", "description": "Fails."})),
            ("Variables in namespace:", _llm_json({"reasoning": "Forget the answer once.", "code": "draft = 1", "description": "No final answer."})),
        ):
            if key in prompt and "Review whether this attempt" not in prompt:
                self.seen[key] += 1
                if self.seen[key] == 1:
                    self.calls.append((prompt, system_prompt))
                    return LLMResponse(content=broken)
        return super().generate(prompt, system_prompt)


def _prompts_after(events, start_index):
    return [
        event["prompt"] for event in events[start_index:]
        if event.get("event_type") in {"generate_call", "review_prompt", "final_answer_review_prompt"}
    ]


def test_decisions_are_identical_through_replans_and_revision_prompts():
    runner = TableQARunner(
        STRUCTURE_PATH, WORKBOOK_PATH, llm_client=DecisionRetryLLM(),
        config={"table_agent": {"qa_final_answer_review": True}},
    )
    result = runner.run("What is the average score?")

    assert result.success and result.replan_count >= 1
    generation = [e["prompt"] for e in result.logs if e.get("event_type") == "generate_call"]
    inspect_revisions = [p for p in generation if "Your previous code execution failed" in p]
    synthesis_revisions = [p for p in generation if "previous synthesis attempt failed" in p]
    assert inspect_revisions and synthesis_revisions
    planner_prompts = [e["prompt"] for e in result.logs if e.get("event_type") == "planner_prompt"]
    assert len(planner_prompts) >= 2
    for prompt in planner_prompts + _prompts_after(result.logs, 0):
        assert prompt.count(_DECISIONS) == 1


def test_a_reused_runner_does_not_carry_decisions_into_the_next_question():
    llm = DecisionRetryLLM()
    runner = TableQARunner(
        STRUCTURE_PATH, WORKBOOK_PATH, llm_client=llm,
        config={"table_agent": {"qa_final_answer_review": True}},
    )
    first = runner.run("What is the average score?")
    assert any(_DECISIONS in prompt for prompt in _prompts_after(first.logs, 0))

    llm.fail_understanding = True
    start = len(runner.env.logger.events)
    second = runner.run("What is the average score?")
    later = _prompts_after(second.logs, start)
    assert later and not any("Question decisions" in prompt for prompt in later)


def test_understanding_prompt_sent_at_run_time_carries_the_answer_form_rules():
    from TableAgent.stages.qa.prompts.understanding import ANSWER_FORM_RULES

    llm = UnderstandingLLM()
    _understanding_runner(llm).run("What is the average score?")

    assert llm.understanding_prompts and ANSWER_FORM_RULES in llm.understanding_prompts[0]


def test_workbook_preview_of_a_long_sheet_keeps_its_head_and_tail_and_marks_the_gap():
    import openpyxl
    from TableAgent.stages.qa.actions.understand_question import workbook_preview

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet.append(["Industry", "Rate"])
    for index in range(1, 400):
        sheet.append([f"industry {index}", index])
    sheet.append(["Note: data for rail transportation are provided by an outside agency.", None])

    preview = workbook_preview(workbook, "book.xlsx", max_chars=2000)

    assert "A1:Industry" in preview
    assert "Note: data for rail transportation" in preview
    gap = re.search(r"\[rows (\d+)-(\d+) omitted\]", preview)
    assert gap is not None
    first_hidden, last_hidden = int(gap.group(1)), int(gap.group(2))
    assert f"A{first_hidden - 1}:" in preview and f"A{last_hidden + 1}:" in preview
    assert f"A{first_hidden}:" not in preview and f"A{last_hidden}:" not in preview
    assert len(preview) < 2000 + 400


def test_workbook_preview_of_a_short_sheet_has_no_gap_marker():
    import openpyxl
    from TableAgent.stages.qa.actions.understand_question import workbook_preview

    workbook = openpyxl.Workbook()
    for index in range(5):
        workbook.active.append([f"row {index}", index])

    assert "omitted" not in workbook_preview(workbook, "book.xlsx", max_chars=2000)


def test_workbook_preview_keeps_the_tail_when_the_first_row_is_too_long():
    import openpyxl
    from TableAgent.stages.qa.actions.understand_question import workbook_preview

    workbook = openpyxl.Workbook()
    workbook.active.append(["x" * 900])
    for index in range(50):
        workbook.active.append([f"row {index}"])
    workbook.active.append(["Source: survey footnote"])

    preview = workbook_preview(workbook, "book.xlsx", max_chars=600)
    assert "Source: survey footnote" in preview
    assert re.search(r"\[rows \d+-\d+ omitted\]", preview)


class _FixedPlanLLM:
    def __init__(self, content: str):
        self.content = content

    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        return LLMResponse(content=self.content)


@pytest.mark.parametrize("table_is_fixed", [True, False])
def test_planner_drops_table_selection_only_when_the_table_is_fixed(table_is_fixed):
    from TableAgent.stages.qa.actions.base_action import PlanGenerationRequest
    from TableAgent.stages.qa.actions.write_plan import WriteQAPlanAction

    plan_json = _llm_json({"subtasks": [
        {"id": "select_tables", "layer": "table_inspect", "depends_on": [], "description": "Select the table."},
        {"id": "inspect_fields", "layer": "inspect", "depends_on": ["select_tables"], "description": "Read the fields."},
        {"id": "synthesize_answer", "layer": "synthesis", "depends_on": ["inspect_fields"], "description": "Answer."},
    ]})
    env = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    try:
        table_id = env.operators.list_tables()[0] if table_is_fixed else None
        plan = WriteQAPlanAction(env, _FixedPlanLLM(plan_json)).run(
            PlanGenerationRequest(question="What is the total?", table_id=table_id)
        ).subtasks
    finally:
        env.workbook.close()

    by_id = {subtask.id: subtask for subtask in plan}
    if table_is_fixed:
        assert list(by_id) == ["inspect_fields", "synthesize_answer"]
        assert by_id["inspect_fields"].depends_on == []
    else:
        assert list(by_id) == ["select_tables", "inspect_fields", "synthesize_answer"]
        assert by_id["inspect_fields"].depends_on == ["select_tables"]
