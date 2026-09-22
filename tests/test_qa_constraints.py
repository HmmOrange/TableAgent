"""Constraints that keep a QA run on the same path across non-deterministic samples.

The QA stage runs against an MoE model under continuous batching, so identical inputs
do not guarantee identical generations even at temperature 0. Benchmark repeats showed
most losses came from runs diverging onto a different execution path rather than from
questions the pipeline cannot answer. These tests pin the constraints that remove those
forks: a gated raw-sheet read, lookups that fail loudly, and an answer written in the
form the question asked for.
"""

from __future__ import annotations

import json

import pytest

from TableAgent.stages.qa.actions.execute_notebook import positional_cell_access
from TableAgent.stages.qa.actions.review_final_answer import ReviewFinalAnswerAction
from TableAgent.stages.qa.environment.qa_env import QAEnvironment
from TableAgent.stages.qa.experience import ExperiencePool, ExperienceRecord
from TableAgent.stages.qa.operators.table_routing_operator import TableRef
from TableAgent.stages.qa.runner_support import QARunnerSupportMixin

STRUCTURE_PATH = "sample/structure.yaml"
WORKBOOK_PATH = "sample/QA_sample.xlsx"


@pytest.fixture
def env():
    environment = QAEnvironment(STRUCTURE_PATH, WORKBOOK_PATH)
    environment.structured_probe_log = []
    return environment


@pytest.fixture
def gated_env(env):
    """The raw-sheet gate is opt-in, so a test about it has to ask for it."""
    env.qa_raw_sheet_gate = True
    return env


# --- table ids answer `.id` like every other structure object --------------------

def test_table_ref_is_a_string_that_also_answers_id():
    ref = TableRef("table1", label="Employees", sheet="Sheet1", score=2.5)
    assert ref == "table1"
    assert ref.id == "table1" and ref.table_id == "table1"
    assert {ref: 1}["table1"] == 1
    assert json.dumps([ref]) == '["table1"]'
    assert ref.label == "Employees"


def test_find_tables_supports_both_access_patterns(env):
    tables = env.operators.find_tables("score", top_k=2)
    assert tables, "the sample workbook should match at least one table"
    # Both spellings appear in generated code; neither may raise.
    assert [t.id for t in tables] == [str(t) for t in tables]
    assert all(t in env.operators.list_tables() for t in tables)


# --- missing headers and groups fail loudly ---------------------------------------

def test_get_header_raises_with_the_available_ids(env):
    table_id = env.default_table_id()
    with pytest.raises(KeyError) as excinfo:
        env.operators.get_header(table_id, "no_such_header")
    message = str(excinfo.value)
    assert "no_such_header" in message
    assert "Available header ids" in message
    assert "find_headers" in message


def test_has_header_probes_without_raising(env):
    table_id = env.default_table_id()
    known = env.operators.list_headers(table_id)[0].id
    assert env.operators.has_header(table_id, known) is True
    assert env.operators.has_header(table_id, "no_such_header") is False


def test_get_group_raises_with_the_available_ids(env):
    table_id = env.default_table_id()
    with pytest.raises(KeyError) as excinfo:
        env.operators.get_group(table_id, "no_such_group")
    assert "no_such_group" in str(excinfo.value)
    assert env.operators.has_group(table_id, "no_such_group") is False


# --- the raw-sheet read is gated ---------------------------------------------------

def test_raw_sheet_read_is_refused_before_the_structured_path_fails(gated_env):
    with pytest.raises(PermissionError) as excinfo:
        gated_env.operators.read_sheet_as_dataframe()
    message = str(excinfo.value)
    assert "gated" in message
    assert "find_headers" in message


def test_raw_sheet_read_opens_after_a_structure_lookup_comes_back_empty(gated_env):
    table_id = gated_env.default_table_id()
    assert gated_env.operators.find_headers(table_id, "zzz_nothing_matches_this") == []
    frame = gated_env.operators.read_sheet_as_dataframe()
    assert not frame.empty


def test_raw_sheet_read_opens_after_a_missing_header_raises(gated_env):
    table_id = gated_env.default_table_id()
    with pytest.raises(KeyError):
        gated_env.operators.get_header(table_id, "no_such_header")
    assert not gated_env.operators.read_sheet_as_dataframe().empty


def test_raw_sheet_read_opens_after_a_missing_group_raises(gated_env):
    table_id = gated_env.default_table_id()
    with pytest.raises(KeyError):
        gated_env.operators.get_group(table_id, "no_such_group")
    assert not gated_env.operators.read_sheet_as_dataframe().empty


def test_a_resolved_lookup_does_not_open_the_gate(gated_env):
    table_id = gated_env.default_table_id()
    assert gated_env.operators.list_headers(table_id)
    with pytest.raises(PermissionError):
        gated_env.operators.read_sheet_as_dataframe()


def test_gate_opens_when_no_verified_structure_can_serve_the_read(gated_env):
    # A table the structure stage never covered has no structured path to try first,
    # so the gate must not demand one.
    with pytest.raises(PermissionError):
        gated_env.operators.read_sheet_as_dataframe()
    gated_env.structures = {}
    assert gated_env.operators._structure_is_unusable() is True
    assert not gated_env.operators.read_sheet_as_dataframe().empty


def test_the_gate_is_off_unless_asked_for(env):
    """A benchmark round measured the gate as neutral at best while the prompt paragraph
    that accompanied it tracked a drop in review-accepted attempts, so it is opt-in."""
    assert not env.operators.read_sheet_as_dataframe().empty


# --- positional cell access is reported, not blocked --------------------------------

@pytest.mark.parametrize(
    "code, expected",
    [
        ("value = df.iloc[3, 5]", [".iloc[3, 5]"]),
        ("value = rows.values[2, 1]", [".values[2, 1]"]),
        ("first = matches.iloc[0]", []),
        ("subset = df.iloc[mask, :]", []),
        ("name = df.loc[3, 'label']", []),
        ("broken = (", []),
    ],
)
def test_positional_cell_access_flags_only_fixed_coordinates(code, expected):
    assert positional_cell_access(code) == expected


# --- the answer is written in the form the question asked for -----------------------

@pytest.mark.parametrize(
    "answer, question, expected",
    [
        ("True", "Is the April index higher than March?", "Yes"),
        ("False", "Does employment exceed 2000?", "No"),
        ("true", "Did the rate rise?", "Yes"),
        ("True", "Which group had the higher rate?", "True"),
        ("7668.0", "What is the difference between X and Y?", "7668"),
        ("9.440000000000001", "What is the difference in wages?", "9.44"),
        ("185.57799999999997", "What was the average CPI?", "185.578"),
        (
            "The gap is 0.10000000000000053 points",
            "What was the change?",
            "The gap is 0.1 points",
        ),
        ("2005, 132", "In which year was the count highest?", "2005, 132"),
        ("Service-providing industries", "Which industry is highest?", "Service-providing industries"),
        ("", "Is it higher?", ""),
    ],
)
def test_answer_contract_normalises_form_without_changing_value(answer, question, expected):
    assert QARunnerSupportMixin._apply_answer_contract(answer, question) == expected


def test_answer_contract_keeps_precision_a_spreadsheet_can_carry():
    # Anything a workbook cell actually holds survives untouched.
    for value in ("3.14159265", "1.23456789012", "0.000123456", "1234567.89"):
        assert QARunnerSupportMixin._apply_answer_contract(value, "What is it?") == value


def test_answer_contract_rounds_past_twelve_significant_digits():
    # Documented limit: noise and genuine precision are indistinguishable this far out,
    # and spreadsheet-sourced answers never carry thirteen meaningful digits.
    assert (
        QARunnerSupportMixin._apply_answer_contract("3.14159265358979", "What is pi?")
        == "3.14159265359"
    )


# --- experience is scoped to the subtask that asks for it ----------------------------

def _record(subtask_id: str, round_num: int, score: float) -> ExperienceRecord:
    return ExperienceRecord(
        subtask_id=subtask_id,
        description=f"{subtask_id} round {round_num}",
        code="pass",
        observation="obs",
        score=score,
        round=round_num,
    )


def test_experience_selection_leads_with_the_subtasks_own_attempts():
    """Scoping the pool to the asking subtask alone was measured to hurt: a first
    attempt was left with no examples at all. Own attempts lead; the rest is filled."""
    pool = ExperiencePool(max_records=5)
    for index in range(5):
        pool.add(_record("other_subtask", index + 1, 1.0))
    pool.add(_record("current_subtask", 1, 0.0))

    selected = pool.select("current_subtask")
    assert selected[0].subtask_id == "current_subtask"
    assert len(selected) == 5

    # Without a scope the run-wide, score-ranked view is unchanged.
    assert all(record.score == 1.0 for record in pool.select())


def test_experience_format_shows_the_failure_that_prompted_the_retry():
    pool = ExperiencePool(max_records=5)
    for index in range(5):
        pool.add(_record("earlier", index + 1, 1.0))
    pool.add(_record("current", 1, 0.0))
    formatted = pool.format(subtask_id="current")
    assert 'subtask="current"' in formatted


# --- a weak derivation does not discard a correct answer ------------------------------

class _StubLLM:
    def __init__(self, payload: dict):
        self.payload = payload

    def generate(self, prompt: str, system_prompt: str | None = None):
        class _Response:
            content = "```json\n" + json.dumps(self.payload) + "\n```"

        return _Response()


def _review(env, payload: dict):
    action = ReviewFinalAnswerAction(env, llm_client=_StubLLM(payload))
    return action.run(question="How many?", plan=[], outputs=[], final_answer="12")


def test_final_review_separates_a_wrong_answer_from_a_weak_derivation(env):
    weak = _review(env, {"accepted": False, "answer_wrong": False, "score": 0.4, "feedback": "loose filter"})
    assert weak.accepted is False and weak.answer_wrong is False

    wrong = _review(env, {"accepted": False, "answer_wrong": True, "score": 0.1, "feedback": "wrong column"})
    assert wrong.answer_wrong is True


def test_final_review_without_the_field_keeps_the_original_behaviour(env):
    legacy = _review(env, {"accepted": False, "score": 0.2, "feedback": "no verdict field"})
    assert legacy.answer_wrong is True
    accepted = _review(env, {"accepted": True, "score": 1.0, "feedback": "fine"})
    assert accepted.answer_wrong is False


# --- generation is constrained to the schema where the backend supports it -----------

class _SchemaAwareClient:
    def __init__(self):
        self.seen: list[dict | None] = []

    def generate(self, prompt, system_prompt=None, response_schema=None):
        self.seen.append(response_schema)

        class _Response:
            content = "{}"

        return _Response()


class _LegacyClient:
    """The two-argument contract other stages and mocks implement."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system_prompt=None):
        self.calls += 1

        class _Response:
            content = "{}"

        return _Response()


def test_schema_is_passed_only_to_clients_that_accept_it():
    from TableAgent.stages.qa.schemas import CODE_SCHEMA, generate_json

    aware = _SchemaAwareClient()
    generate_json(aware, "p", system_prompt="s", schema=CODE_SCHEMA)
    assert aware.seen == [CODE_SCHEMA]

    legacy = _LegacyClient()
    generate_json(legacy, "p", system_prompt="s", schema=CODE_SCHEMA)
    assert legacy.calls == 1


def test_client_capability_is_never_inferred_from_another_client():
    """CPython reuses the id of a collected object; capability must not be cached by id."""
    from TableAgent.stages.qa.schemas import CODE_SCHEMA, _accepts_response_schema

    aware_id = id(_SchemaAwareClient())  # immediately collectable
    legacy = _LegacyClient()
    if id(legacy) == aware_id:
        assert _accepts_response_schema(legacy) is False
    # Whether or not the ids collide, each client must be judged on its own signature.
    assert _accepts_response_schema(_LegacyClient()) is False
    assert _accepts_response_schema(_SchemaAwareClient()) is True
    generate_json_ok = _LegacyClient()
    assert _accepts_response_schema(generate_json_ok) is False
    del CODE_SCHEMA


def test_kwargs_client_is_treated_as_schema_aware():
    from TableAgent.stages.qa.schemas import _accepts_response_schema

    class _Forwarding:
        def generate(self, prompt, system_prompt=None, **kwargs):
            return None

    assert _accepts_response_schema(_Forwarding()) is True
    assert _accepts_response_schema(object()) is False


# --- one pydantic model per reply: it constrains generation and validates parsing ------

def test_emitted_schemas_are_self_contained():
    """Guided-decoding backends differ on `$ref`; these schemas carry no references."""
    from TableAgent.stages.qa.schemas import (
        CODE_SCHEMA,
        FINAL_REVIEW_SCHEMA,
        PLAN_SCHEMA,
        REVIEW_SCHEMA,
    )

    for schema in (PLAN_SCHEMA, CODE_SCHEMA, REVIEW_SCHEMA, FINAL_REVIEW_SCHEMA):
        blob = json.dumps(schema)
        assert "$ref" not in blob and "$defs" not in blob
        assert '"title"' not in blob


def test_generation_is_constrained_more_tightly_than_parsing():
    """A field with a parse-side default must still be stated when decoding is guided."""
    from TableAgent.stages.qa.schemas import FINAL_REVIEW_SCHEMA, PLAN_SCHEMA, PlannedSubtask

    item = PLAN_SCHEMA["properties"]["subtasks"]["items"]
    assert set(item["required"]) == {"id", "description", "layer", "category", "depends_on"}
    assert item["properties"]["layer"]["enum"] == ["table_inspect", "inspect", "synthesis"]
    # ... while the parser still fills those in when they are absent.
    parsed = PlannedSubtask.model_validate({"id": "a", "description": "d"})
    assert parsed.layer == "inspect" and parsed.category == "normal"

    assert "answer_wrong" in FINAL_REVIEW_SCHEMA["required"]
    assert FINAL_REVIEW_SCHEMA["properties"]["answer_wrong"] == {"type": "boolean"}


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"id": "a", "description": "d", "depends_on": "x, y"}, ["x", "y"]),
        ({"id": "a", "description": "d", "depends_on": ["x", " y "]}, ["x", "y"]),
        ({"id": "a", "description": "d", "depends_on": None}, []),
        ({"id": "a", "description": "d"}, []),
    ],
)
def test_dependencies_are_accepted_however_the_model_writes_them(payload, expected):
    from TableAgent.stages.qa.schemas import PlannedSubtask

    assert PlannedSubtask.model_validate(payload).depends_on == expected


def test_scope_written_beside_the_other_fields_is_folded_into_metadata():
    from TableAgent.stages.qa.schemas import PlannedSubtask

    subtask = PlannedSubtask.model_validate(
        {"id": "a", "description": "d", "category": "COMMON_INFO", "common_info_scope": "sheet"}
    )
    assert subtask.category == "common_info"
    assert subtask.metadata["common_info_scope"] == "sheet"


def test_a_common_info_subtask_without_a_scope_is_refused():
    from TableAgent.stages.qa.schemas import QAPlan, ValidationError

    with pytest.raises(ValidationError):
        QAPlan.model_validate(
            {"subtasks": [{"id": "a", "description": "d", "category": "common_info"}]}
        )
    # A synthesis common-info subtask needs no scope.
    QAPlan.model_validate(
        {
            "subtasks": [
                {"id": "a", "description": "d", "category": "common_info", "layer": "synthesis"}
            ]
        }
    )


def test_an_unknown_layer_is_refused():
    from TableAgent.stages.qa.schemas import QAPlan, ValidationError

    with pytest.raises(ValidationError):
        QAPlan.model_validate({"subtasks": [{"id": "a", "description": "d", "layer": "guess"}]})


@pytest.mark.parametrize(
    "payload, expected_score",
    [
        ({"accepted": True}, 1.0),
        ({"accepted": False}, 0.0),
        ({"accepted": True, "score": 1.5}, 1.0),
        ({"accepted": True, "score": -2}, 0.0),
        ({"accepted": False, "score": "nonsense"}, 0.0),
    ],
)
def test_a_verdict_survives_an_unusable_score(payload, expected_score):
    """The number is secondary; losing the verdict over it would cost a whole attempt."""
    from TableAgent.stages.qa.schemas import SubtaskReview

    assert SubtaskReview.model_validate(payload).score == expected_score


# --- an unknown attribute answers itself instead of dead-ending -----------------------

def test_an_unknown_attribute_names_the_real_ones():
    from TableAgent.domain import CellRange, Header

    header = Header("h1", "Score", "d", "column", None, None)
    with pytest.raises(AttributeError) as excinfo:
        header.title
    message = str(excinfo.value)
    assert "Closest available: 'label'" in message
    assert "data_range" in message and "sub_headers" in message

    with pytest.raises(AttributeError) as excinfo:
        CellRange(1, 1, 4, 3).start_cell
    assert "start_col" in str(excinfo.value)


def test_a_declared_surface_keeps_inherited_noise_out_of_the_message():
    """A str subclass would otherwise bury `id` and `label` under forty string methods."""
    from TableAgent.stages.qa.operators.table_routing_operator import TableRef

    ref = TableRef("employment", label="Employment by industry")
    with pytest.raises(AttributeError) as excinfo:
        ref.name
    message = str(excinfo.value)
    assert "Closest available: 'label'" in message
    assert "Available: id, label, score, sheet, table_id." in message
    assert "isnumeric" not in message and "casefold" not in message


def test_the_string_behaviour_of_a_table_ref_is_untouched():
    from TableAgent.stages.qa.operators.table_routing_operator import TableRef

    ref = TableRef("t1")
    assert ref == "t1" and ref.upper() == "T1" and ref.split("1") == ["t", ""]
    assert {ref: "v"}["t1"] == "v"


def test_private_probes_fall_through_untouched():
    """copy, pickle and pandas look for private protocol hooks; they get a bare miss."""
    import copy
    import pickle

    from TableAgent.domain import CellRange

    cell_range = CellRange(1, 1, 4, 3, "S1")
    with pytest.raises(AttributeError) as excinfo:
        cell_range.__deepcopy__
    assert str(excinfo.value) == "__deepcopy__"
    assert copy.deepcopy(cell_range) == cell_range
    assert pickle.loads(pickle.dumps(cell_range)) == cell_range


def test_the_catalog_states_the_attributes_before_a_guess_is_needed(env):
    catalog = env.operators.operator_catalog()
    assert "return types:" in catalog
    for expected in ("Header: data_range", "TableRef: id, label", "CellRange: contains"):
        assert expected in catalog
    # The range-is-None trap is named rather than left to be discovered at runtime.
    assert "is None when the structure did not record one" in catalog


# --- the retry prompt keeps both the lesson and the demonstrations ---------------------

def test_a_first_attempt_still_sees_working_examples():
    """A subtask with no attempts of its own must not be left with an empty prompt."""
    pool = ExperiencePool(max_records=5)
    for index in range(4):
        pool.add(_record("earlier", index + 1, 1.0))

    selected = pool.select("fresh")
    assert [record.subtask_id for record in selected] == ["earlier"] * 4
    assert pool.format(subtask_id="fresh") != "No previous experience."


def test_the_subtasks_own_failures_come_first_and_are_never_crowded_out():
    pool = ExperiencePool(max_records=5)
    for index in range(10):
        pool.add(_record("earlier", index + 1, 1.0))
    pool.add(_record("current", 1, 0.0))
    pool.add(_record("current", 2, 0.0))

    selected = pool.select("current")
    assert [record.subtask_id for record in selected[:2]] == ["current", "current"]
    assert len(selected) == 5
    assert all(record.subtask_id == "earlier" for record in selected[2:])


def test_own_attempts_are_rendered_last_nearest_the_instruction():
    pool = ExperiencePool(max_records=5)
    pool.add(_record("earlier", 1, 1.0))
    pool.add(_record("current", 1, 0.0))
    tags = [
        line for line in pool.format(subtask_id="current").splitlines()
        if line.startswith("<attempt")
    ]
    assert 'subtask="earlier"' in tags[0]
    assert 'subtask="current"' in tags[-1]


def test_a_subtask_that_fills_the_budget_alone_shows_only_itself():
    pool = ExperiencePool(max_records=2)
    pool.add(_record("earlier", 1, 1.0))
    pool.add(_record("current", 1, 0.0))
    pool.add(_record("current", 2, 0.0))
    assert {r.subtask_id for r in pool.select("current")} == {"current"}


# --- a missing column answers with the columns that exist ------------------------------

def test_a_missing_column_names_the_available_ones():
    import pandas as pd

    from TableAgent.stages.qa.environment.notebook import describe_missing_key

    namespace = {
        "table_df": pd.DataFrame({"occupation": [1], "employed_total": [2]}),
        "pd": pd,
        "count": 3,
    }
    try:
        namespace["table_df"]["Occupation"]
    except KeyError as error:
        hint = describe_missing_key(error, namespace)
    assert "table_df: occupation, employed_total" in hint
    assert "closest to 'Occupation': 'occupation'" in hint


def test_the_column_hint_stays_quiet_when_it_has_nothing_to_add():
    import pandas as pd

    from TableAgent.stages.qa.environment.notebook import describe_missing_key

    # Not a KeyError, no frames in scope, and a frame that does have the column.
    assert describe_missing_key(ValueError("x"), {}) == ""
    assert describe_missing_key(KeyError("a"), {"n": 1}) == ""
    frame = pd.DataFrame({"a": [1]})
    assert describe_missing_key(KeyError("a"), {"f": frame}) == ""


def test_a_missing_column_is_reported_through_the_notebook(env):
    frame_setup = "df = operators.read_table_as_dataframe(env.default_table_id())"
    env.execute_code(frame_setup)
    output, error, success, _ = env.execute_code("value = df['NoSuchColumn']")
    assert success is False
    assert "columns available" in error and "NoSuchColumn" in error


def test_the_environment_describes_itself(env):
    with pytest.raises(AttributeError) as excinfo:
        env.store_variable
    message = str(excinfo.value)
    assert "store_variable" in message and "execution_namespace" in message


# --- constrained decoding is a per-run experiment, not a default ----------------------

def test_a_schema_only_constrains_generation_when_the_run_asks(env):
    """The first backend this reached emitted JSON strings with no newline escape, which
    collapsed a generated Python cell onto one line. It stays opt-in."""
    from TableAgent.stages.qa.schemas import PLAN_SCHEMA, schema_if_enabled

    assert schema_if_enabled(env, PLAN_SCHEMA) is None
    env.qa_structured_output = True
    assert schema_if_enabled(env, PLAN_SCHEMA) is PLAN_SCHEMA


def test_code_generation_is_never_grammar_constrained():
    """`code` carries Python, so it needs real newlines; nothing may constrain it."""
    import inspect

    from TableAgent.stages.qa.actions import llm_code_generation

    source = inspect.getsource(llm_code_generation.LLMCodeGenerationAction)
    assert "schema=" not in source


# --- the notebook starts warm, without spending a model call on it --------------------

def test_the_first_subtask_sees_the_table_in_the_notebook_history():
    """Dropping the table-selection subtask also dropped the cell it left behind, and
    review rejections rose six points because every inspection then started cold."""
    from TableAgent.stages.qa import TableQARunner
    from tests.mock_policy import MockActionPolicy

    runner = TableQARunner(STRUCTURE_PATH, WORKBOOK_PATH, policy=MockActionPolicy())
    runner._warm_start_notebook(runner.env.default_table_id())

    history = runner.env.get_history(last_n=3, include_output=True)
    assert "No code has been executed yet." not in history
    assert "table_df shape" in history and "columns:" in history


def test_the_warm_start_does_not_rebuild_what_the_runner_preloaded():
    """It prints; it must not assign. Rebuilding `table_df` here once produced integer
    column labels and silently changed what every later cell selected."""
    from TableAgent.stages.qa import TableQARunner
    from tests.mock_policy import MockActionPolicy

    runner = TableQARunner(STRUCTURE_PATH, WORKBOOK_PATH, policy=MockActionPolicy())
    table_id = runner.env.default_table_id()
    runner._set_active_tables([table_id])
    before = runner.env.execution_namespace["table_df"]

    runner._warm_start_notebook(table_id)

    assert runner.env.execution_namespace["table_df"] is before
    assert list(before.columns) == list(
        runner.env.execution_namespace["table_df"].columns
    )


def test_the_warm_start_leaves_a_worked_example_behind():
    """The subtask it replaced left two things: a printed cell and an accepted attempt.
    Only seeding both keeps the first inspection from opening with no examples at all."""
    from TableAgent.stages.qa import TableQARunner
    from tests.mock_policy import MockActionPolicy

    runner = TableQARunner(STRUCTURE_PATH, WORKBOOK_PATH, policy=MockActionPolicy())
    table_id = runner.env.default_table_id()
    runner._set_active_tables([table_id])
    assert runner.env.experience_pool.format(subtask_id="first_inspect") == "No previous experience."

    runner._warm_start_notebook(table_id)

    formatted = runner.env.experience_pool.format(subtask_id="first_inspect")
    assert formatted != "No previous experience."
    assert "read_table_as_dataframe" in formatted or "table_df" in formatted


def test_a_call_reports_whether_it_actually_carried_a_schema():
    """The metric used to read the client's capability, so a run that sent no schema at
    all still reported itself as constrained."""
    from TableAgent.stages.qa.runner import TokenCountingLLM

    class _Client:
        structured_output_mode = "response_format"

        def generate(self, prompt, system_prompt=None, response_schema=None):
            class _R:
                content = "{}"
                prompt_tokens = completion_tokens = 0

            return _R()

    client = TokenCountingLLM(_Client())
    client.generate("p")
    client.generate("p", response_schema={"type": "object"})
    plain, constrained = client.call_metrics()
    assert plain["schema_sent"] is False and plain["structured_mode"] == "off"
    assert constrained["schema_sent"] is True
    assert constrained["structured_mode"] == "response_format"
