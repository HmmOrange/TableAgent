from pathlib import Path

import openpyxl
import yaml

from TableAgent.stages.qa.environment.qa_env import QAEnvironment
from TableAgent.stages.qa.group_hints import question_group_hints
from TableAgent.stages.retrieval.contracts import SourceCandidate
from TableAgent.stages.retrieval.perfect import PerfectRetrievalMixin
from TableAgent.stages.structure.card_builders import (
    build_sheet_metadata_payload,
    build_source_retrieval_card,
)
from TableAgent.stages.structure.group_enrichment import (
    GROUP_USER_PROMPT_TEMPLATE,
    GroupEnrichmentStage,
    parse_group_enrichment,
)
from TableAgent.llm import LLMResponse
from TableAgent.stages.structure.layout.agent import _merge_existing_groups
from TableAgent.stages.structure.layout.parsing import (
    extract_layout_structure_result,
    nullify_structure_ranges,
)
from TableAgent.stages.structure.verification.checks import verify_structure
from TableAgent.utils import load_table_structures, range_to_a1


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    workbook_path = tmp_path / "groups.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["Metric", "Value", "Other"])
    sheet.append(["Men", None, None])
    sheet.append(["Participation rate", 70, 60])
    sheet.append(["Population", 100, 90])
    sheet.append(["Women", None, None])
    sheet.append(["Participation rate", 65, 55])
    workbook.save(workbook_path)

    structure_path = tmp_path / "structure.yaml"
    structure_path.write_text(yaml.safe_dump({
        "employment": {
            "id": "employment",
            "name": "Employment",
            "description": "Employment statistics.",
            "sheet": "Sheet1",
            "headers": [
                {"id": "metric", "label": "Metric", "description": "Metric label.", "orientation": "column", "header_range": "A1", "data_range": "A2:A6", "sub_headers": []},
                {"id": "value", "label": "Value", "description": "Value.", "orientation": "column", "header_range": "B1", "data_range": "B2:B6", "sub_headers": []},
            ],
            "groups": [
                {"id": "men", "label": "Men", "group_range": "A2:C4", "data_range": "B3:C4", "axis": "row", "description": "Statistics for men."},
                {"id": "women", "label": "Women", "group_range": "A5:C6", "data_range": "B6:C6", "axis": "row", "description": "Statistics for women."},
            ],
        }
    }, sort_keys=False), encoding="utf-8")
    return workbook_path, structure_path


def test_group_loader_operators_and_hints(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    structures = load_table_structures(str(structure_path))
    groups = structures["employment"]["groups"]
    assert [group.id for group in groups] == ["men", "women"]
    assert range_to_a1(groups[0].group_range) == "A2:C4"

    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        assert [group.id for group in env.operators.find_groups("employment", "men")] == ["men"]
        assert env.operators.read_group_data("employment", "men") == [[70, 60], [100, 90]]
        assert env.operators.find_in_group("employment", "men", "Participation rate")[0][0].row == 3
        assert range_to_a1(env.operators.intersect_group_with_header("employment", "men", "value")) == "B3:B4"
        hints = question_group_hints(env, "Compare Men and Women", ["employment"])
        assert "group_id=men" in hints and "group_id=women" in hints
    finally:
        env.workbook.close()


def test_group_parser_merge_and_nullification():
    response = yaml.safe_dump({"structure": {"table": {
        "headers": [{"label": "Metric", "description": "Metric.", "orientation": "column", "header_range": "A1", "data_range": "A2:A9", "sub_headers": []}],
        "groups": [
            {"id": "Men", "label": "Men", "group_range": "A2:B4", "data_range": "B3:B4", "axis": "ROW", "description": "First."},
            {"id": "Men", "label": "Men again", "group_range": "A4:B6", "data_range": "B5:B6", "axis": "row", "description": "Second."},
            {"label": "Bad", "axis": "diagonal"},
        ],
    }}})
    result = extract_layout_structure_result(response)
    parsed = yaml.safe_load(result.structure_text)
    assert [group["id"] for group in parsed["table"]["groups"]] == ["men", "men_2"]
    assert "axis must be row, column, or region" in result.preflight_errors[0]

    previous = {"table": {"groups": parsed["table"]["groups"][:2]}}
    updated = {"table": {"groups": [
        {**parsed["table"]["groups"][1], "description": "Updated."},
        {"id": "new", "label": "New", "group_range": "A7:B8", "data_range": "B8", "axis": "row", "description": "New."},
    ]}}
    assert _merge_existing_groups(previous, updated)
    assert [group["id"] for group in updated["table"]["groups"]] == ["men", "men_2", "new"]
    assert updated["table"]["groups"][1]["description"] == "Updated."

    nulled = yaml.safe_load(nullify_structure_ranges(result.structure_text, ["table.groups[0].group_range", "table.groups[0].data_range"]))
    assert nulled["table"]["groups"][0]["group_range"] is None
    assert nulled["table"]["groups"][0]["data_range"] is None


def test_post_structure_group_parser_accepts_only_group_list():
    structure = yaml.safe_dump({
        "table": {
            "id": "table",
            "name": "Example",
            "description": "Example table.",
            "sheet": "Sheet1",
            "headers": [{
                "id": "metric",
                "label": "Metric",
                "description": "Metric label.",
                "orientation": "column",
                "header_range": "A1",
                "data_range": "A2:A6",
                "sub_headers": [],
            }],
        },
    }, sort_keys=False)
    response = yaml.safe_dump({
        "groups": [{
                "id": "Men",
                "label": "Men",
                "group_range": "A2:C4",
                "data_range": "B3:C4",
                "axis": "ROW",
                "description": "Statistics for men.",
            }],
    }, sort_keys=False)

    groups, errors = parse_group_enrichment(response, "table")

    assert errors == []
    assert groups[0]["id"] == "men"
    assert "Completed table from structure.yaml" in GROUP_USER_PROMPT_TEMPLATE
    assert "group_range: exact cell or merged range containing the visible group label only" in GROUP_USER_PROMPT_TEMPLATE
    assert "Do not return the table key" in GROUP_USER_PROMPT_TEMPLATE

    for invalid in (
        yaml.safe_dump({"groups": {"table": []}}),
        yaml.safe_dump({"table": {"groups": []}}),
        yaml.safe_dump({"groups": [], "headers": []}),
    ):
        try:
            parse_group_enrichment(invalid, "table")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected groups-only list contract to reject {invalid!r}")


def test_group_enrichment_runs_once_per_table_and_preserves_structure(tmp_path: Path):
    class Renderer:
        def source_viewport_to_image(self, workbook_path, sheet_name, viewport_range, image_path):
            image_path.write_bytes(b"image")

    class VLM:
        def __init__(self):
            self.calls = []

        def generate_with_image(self, prompt, image_path, system_prompt=None):
            self.calls.append((prompt, image_path, system_prompt))
            label = "Men" if "Table: first" in prompt else "Women"
            return LLMResponse(content=yaml.safe_dump({"groups": [{
                "id": label,
                "label": label,
                "group_range": "A2:C4",
                "data_range": "B3:C4",
                "axis": "row",
                "description": f"Statistics for {label.lower()}.",
            }]}, sort_keys=False))

    original = {
        "first": {"id": "first", "headers": [{"id": "a"}]},
        "second": {"id": "second", "headers": [{"id": "b"}]},
    }
    structure_text = yaml.safe_dump(original, sort_keys=False)
    vlm = VLM()
    result = GroupEnrichmentStage(Renderer(), vlm).run(
        workbook_path=tmp_path / "book.xlsx",
        sheet_name="Sheet1",
        viewport_range="A1:C4",
        structure_text=structure_text,
        artifact_dir=tmp_path / "groups",
    )
    enriched = yaml.safe_load(result.structure_text)

    assert len(vlm.calls) == 2
    assert len(result.responses) == 2
    assert enriched["first"]["headers"] == original["first"]["headers"]
    assert enriched["second"]["headers"] == original["second"]["headers"]
    assert enriched["first"]["groups"][0]["id"] == "men"
    assert enriched["second"]["groups"][0]["id"] == "women"
    assert list(enriched["first"])[-1] == "groups"
    assert list(enriched["second"])[-1] == "groups"
    assert all("Return only:" in prompt for prompt, _, _ in vlm.calls)
    assert "second:" not in vlm.calls[0][0]
    assert "first:" not in vlm.calls[1][0]


def test_group_verification_and_repair(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    report = verify_structure(workbook_path, "Sheet1", structure_path)
    assert report["status"] == "good"

    structure = yaml.safe_load(structure_path.read_text(encoding="utf-8"))
    structure["employment"]["groups"][0]["label"] = "Missing label"
    structure_path.write_text(yaml.safe_dump(structure, sort_keys=False), encoding="utf-8")
    report = verify_structure(workbook_path, "Sheet1", structure_path)
    repaired = yaml.safe_load(report["repaired_structure_yaml"])
    assert repaired["employment"]["groups"][0]["group_range"] is None
    assert repaired["employment"]["groups"][0]["data_range"] is None


def test_group_retrieval_payload_and_exact_score(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    structure_text = structure_path.read_text(encoding="utf-8")
    card = build_source_retrieval_card(workbook_path, "Sheet1", structure_text, "")
    payload = build_sheet_metadata_payload(workbook_path, "Sheet1", structure_text, "")
    assert "Groups: Men (men): Statistics for men. [row]" in card
    assert payload["tables"][0]["groups"][0] == {
        "id": "men", "label": "Men", "description": "Statistics for men.",
        "axis": "row", "group_range": "A2:C4", "data_range": "B3:C4",
    }
    candidate = SourceCandidate(
        directory=tmp_path, workbook_path=workbook_path, sheet_name="Sheet1",
        image_path=tmp_path / "image.png", html_path=None, structure_text=structure_text,
        sheet_text="", score=0.0, lexical_score=1.0,
    )
    assert PerfectRetrievalMixin._perfect_question_score("Men participation rate", candidate) == 9.0
    assert PerfectRetrievalMixin._perfect_question_score("Women participation rate", candidate) == 9.0


def test_dataframe_carries_section_metadata_and_group_scoped_operators(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        frame = env.operators.read_table_as_dataframe("employment", has_headers=True)
        # Worksheet rows 2..6 land in the frame; rows 2 and 5 are the group labels.
        assert list(frame["__section__"]) == ["Men", "Men", "Men", "Women", "Women"]
        assert list(frame["__section_label_row__"]) == [True, False, False, True, False]

        records = frame[~frame["__section_label_row__"]]
        assert len(records) == 3

        mask = env.operators.group_row_mask(frame, "employment", "men")
        assert list(mask) == [False, True, True, False, False]
        assert env.operators.resolve_group_rows(frame, "employment", "men") == [1, 2]

        dropped = env.operators.read_table_as_dataframe(
            "employment", has_headers=True, drop_group_label_rows=True
        )
        assert len(dropped) == 3
        assert list(dropped["__section__"]) == ["Men", "Men", "Women"]

        plain = env.operators.read_table_as_dataframe(
            "employment", has_headers=True, include_group_column=False
        )
        assert "__section__" not in plain.columns
    finally:
        env.workbook.close()


def test_group_scoped_filter_and_find_threshold(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        # Both sections hold a "Participation rate" row; scoping keeps the match inside one.
        selection = env.operators.filter_in_group("employment", "men", "value", gte=100)
        assert selection.positions == (4,)
        assert env.operators.filter_in_group("employment", "women", "value", gte=100).positions == ()

        assert [group.id for group in env.operators.find_groups("employment", "men")] == ["men"]
        # A query sharing only short filler tokens must not look like a confident match.
        assert env.operators.find_groups("employment", "of an in") == []
        assert len(env.operators.find_groups("employment", "statistics", limit=1)) == 1
    finally:
        env.workbook.close()


def test_groups_reach_planner_and_react_prompts(tmp_path: Path):
    from TableAgent.stages.qa.actions.llm_code_generation import (
        get_structure_summary,
        get_table_catalog_summary,
    )

    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        summary = get_structure_summary(env, "employment")
        assert "Structure Groups" in summary
        assert "ID: men" in summary and "data_range: B3:C4" in summary

        catalog = get_table_catalog_summary(env)
        assert "Men (men): Statistics for men." in catalog

        operator_catalog = env.operators.operator_catalog()
        for advertised in (
            "operators.read_group_data(",
            "operators.find_in_group(",
            "operators.group_row_mask(",
            "operators.filter_in_group(",
            "__section_label_row__",
        ):
            assert advertised in operator_catalog
        assert "`.id` on both Header and StructureGroup" in operator_catalog

        hints = question_group_hints(env, "How many men participated", ["employment"])
        assert "Exact label matches" in hints and "group_id=men" in hints
    finally:
        env.workbook.close()


def test_find_in_group_searches_the_records_the_group_owns(tmp_path: Path):
    from TableAgent.utils import range_to_a1

    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        # A row-axis group owns whole rows, so the search region spans the table's columns
        # even though `data_range` lists only B:C.
        assert range_to_a1(env.operators.group_search_range("employment", "men")) == "A2:B4"
        assert range_to_a1(env.operators.group_search_range("employment", "women")) == "A5:B6"

        # "Participation rate" appears in both sections; the search must stay in one.
        men = env.operators.find_in_group("employment", "men", "Participation rate")
        women = env.operators.find_in_group("employment", "women", "Participation rate")
        assert [cell.row for cell, _ in men] == [3]
        assert [cell.row for cell, _ in women] == [6]

        # A record that belongs to the other section must not leak in.
        assert env.operators.find_in_group("employment", "women", "Population") == []
        assert [cell.row for cell, _ in env.operators.find_in_group("employment", "men", "Population")] == [4]
    finally:
        env.workbook.close()


def test_table_routing_scores_structure_groups(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        # "Women" names a group and no header, table name, or description.
        assert [str(ref) for ref in env.operators.find_tables("Women", top_k=2)] == ["employment"]
        assert "structure groups" in env.operators.operator_catalog()
    finally:
        env.workbook.close()


def test_group_hints_ignore_stopword_only_overlap(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        # The group descriptions read "Statistics for men."; sharing only "for" with the
        # question is not evidence that the question is about that section.
        assert question_group_hints(env, "Report for 2024", ["employment"]) == (
            "No group matched this question."
        )
        assert "group_id=men" in question_group_hints(env, "How many men participated", ["employment"])
    finally:
        env.workbook.close()


def test_header_hints_carry_ranges_like_group_hints(tmp_path: Path):
    from TableAgent.stages.qa.header_hints import question_header_hints

    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        hints = question_header_hints(env, "What is the Value recorded", ["employment"])
        assert "header_id=value" in hints
        assert "header_range=B1" in hints
        assert "data_range=B2:B6" in hints
    finally:
        env.workbook.close()
