from pathlib import Path

import openpyxl
import pytest
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
            "operators.find_cells(",
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


def test_find_cells_searches_the_whole_worksheet(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        # Unlike find_in_group, the label is found in every section, with spacing and case ignored.
        assert [(cell.row, cell.col) for cell, _ in env.operators.find_cells("participation  RATE")] == [(3, 1), (6, 1)]
        assert [value for _, value in env.operators.find_cells("Women", sheet="Sheet1")] == ["Women"]
        # Whole words only: "men" must not match inside "Women".
        assert [cell.row for cell, _ in env.operators.find_cells("men")] == [2]
        assert env.operators.find_cells("   ") == []
        # A single-sheet workbook resolves any sheet name to its only sheet.
        assert [cell.row for cell, _ in env.operators.find_cells("men", sheet="employment")] == [2]
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


def _labeled_env(tmp_path: Path, rows, headers, groups, *, merges=(), styles=None):
    """Build a one-table environment from raw rows and structure entries."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    workbook_path = tmp_path / "labeled.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    for row in rows:
        sheet.append(row)
    for merged in merges:
        sheet.merge_cells(merged)
    for coordinate, apply in (styles or {}).items():
        apply(sheet[coordinate])
    workbook.save(workbook_path)

    structure_path = tmp_path / "labeled.yaml"
    structure_path.write_text(yaml.safe_dump({
        "stats": {
            "id": "stats",
            "name": "Stats",
            "description": "Synthetic statistics table.",
            "sheet": "Sheet1",
            "headers": headers,
            "groups": groups,
        }
    }, sort_keys=False), encoding="utf-8")
    return QAEnvironment(str(structure_path), str(workbook_path))


def _leaf(header_id: str, header_range: str, data_range: str) -> dict:
    return {
        "id": header_id, "label": header_id, "description": header_id,
        "orientation": "column", "header_range": header_range,
        "data_range": data_range, "sub_headers": [],
    }


def _group(group_id: str, label: str, group_range: str, data_range: str, axis: str = "row") -> dict:
    return {
        "id": group_id, "label": label, "description": label, "axis": axis,
        "group_range": group_range, "data_range": data_range,
    }


# A section whose own label row also carries the section's figures, followed by its breakdown.
_SECTION_ROWS = [
    ["Age", "Employed", "Unemployed"],
    ["All ages", 300, 30],
    ["Young", 100, 10],
    ["Young, early", 60, 6],
    ["Young, late", 40, 4],
    ["Old", 200, 20],
]
_SECTION_HEADERS = [
    _leaf("age", "A1", "A2:A6"),
    _leaf("employed", "B1", "B2:B6"),
    _leaf("unemployed", "C1", "C2:C6"),
]
_SECTION_GROUP = _group("all_ages", "All ages", "A2", "B2:C6")


def test_group_header_values_mark_the_label_row_inside_the_block(tmp_path: Path):
    env = _labeled_env(tmp_path, _SECTION_ROWS, _SECTION_HEADERS, [_SECTION_GROUP])
    try:
        records = env.operators.read_group_header_values("stats", "all_ages", "employed")
        assert [record["row"] for record in records] == [2, 3, 4, 5, 6]
        assert [record["is_group_label_row"] for record in records] == [True, False, False, False, False]
        assert [record["values"] for record in records] == [
            {"employed": 300}, {"employed": 100}, {"employed": 60}, {"employed": 40}, {"employed": 200},
        ]
        assert [record["other_fields"]["age"] for record in records] == [
            "All ages", "Young", "Young, early", "Young, late", "Old",
        ]
        assert records[0]["other_fields"]["unemployed"] == 30
        assert "employed" not in records[0]["other_fields"]
    finally:
        env.workbook.close()


def test_group_header_values_when_the_label_sits_outside_and_data_includes_the_label_column(tmp_path: Path):
    rows = [
        ["Item", "Amount"],
        ["Revenues:", None],
        ["Segment A", 10],
        ["Segment B", 20],
        ["Costs:", None],
        ["Segment A", 4],
    ]
    headers = [_leaf("item", "A1", "A2:A6"), _leaf("amount", "B1", "B2:B6")]
    groups = [
        _group("revenues", "Revenues:", "A2", "A3:B4"),
        _group("costs", "Costs:", "A5", "A6:B6"),
    ]
    env = _labeled_env(tmp_path, rows, headers, groups)
    try:
        records = env.operators.read_group_header_values("stats", "revenues", "amount")
        assert [record["row"] for record in records] == [3, 4]
        assert not any(record["is_group_label_row"] for record in records)
        assert [(record["other_fields"]["item"], record["values"]["amount"]) for record in records] == [
            ("Segment A", 10), ("Segment B", 20),
        ]
    finally:
        env.workbook.close()


def test_group_header_values_expand_a_vertically_merged_group_label(tmp_path: Path):
    rows = [
        ["Sex", "Age", "Count"],
        ["Men", "Young", 5],
        [None, "Old", 7],
        ["Women", "Young", 6],
        [None, "Old", 8],
    ]
    headers = [_leaf("sex", "A1", "A2:A5"), _leaf("age", "B1", "B2:B5"), _leaf("count", "C1", "C2:C5")]
    groups = [_group("men", "Men", "A2:A3", "C2:C3"), _group("women", "Women", "A4:A5", "C4:C5")]
    env = _labeled_env(tmp_path, rows, headers, groups, merges=("A2:A3", "A4:A5"))
    try:
        records = env.operators.read_group_header_values("stats", "women", "count")
        assert [record["other_fields"] for record in records] == [
            {"sex": "Women", "age": "Young"}, {"sex": "Women", "age": "Old"},
        ]
        assert [record["values"] for record in records] == [{"count": 6}, {"count": 8}]
        # A label merged across the records labels each of them; no row holds it by itself.
        assert [record["is_group_label_row"] for record in records] == [False, False]
        assert range_to_a1(env.operators.intersect_group_with_header(
            "stats", "women", "count", include_group_label_row=False
        )) == "C4:C5"
    finally:
        env.workbook.close()


def test_group_header_values_keep_numeric_labels_of_a_time_series(tmp_path: Path):
    rows = [
        ["Year", "Month", "Sales"],
        [2023, "Jan", 100],
        [None, "Feb", 110],
        [2024, "Jan", 120],
    ]
    headers = [_leaf("year", "A1", "A2:A4"), _leaf("month", "B1", "B2:B4"), _leaf("sales", "C1", "C2:C4")]
    groups = [_group("y2023", "2023", "A2", "B2:C3"), _group("y2024", "2024", "A4", "B4:C4")]
    env = _labeled_env(tmp_path, rows, headers, groups)
    try:
        records = env.operators.read_group_header_values("stats", "y2023", "sales")
        # The label row is a structural fact; whether it is a total or a record is not decided here.
        assert [record["is_group_label_row"] for record in records] == [True, False]
        assert records[0]["other_fields"] == {"year": 2023, "month": "Jan"}
        assert records[1]["other_fields"] == {"month": "Feb"}
        assert [record["values"]["sales"] for record in records] == [100, 110]
    finally:
        env.workbook.close()


def test_group_header_values_split_a_parent_header_into_its_leaves(tmp_path: Path):
    rows = [
        ["Group", "Private", None, None, "Government"],
        [None, "Total", "Households", "Other", None],
        ["All workers", 128, 1, 127, 21],
    ]
    headers = [
        _leaf("group", "A1:A2", "A3:A3"),
        {
            "id": "private", "label": "Private", "description": "Private industries.",
            "orientation": "column", "header_range": "B1:D1", "data_range": "B3:D3",
            "sub_headers": [
                _leaf("private_total", "B2", "B3:B3"),
                _leaf("private_households", "C2", "C3:C3"),
                _leaf("private_other", "D2", "D3:D3"),
            ],
        },
        _leaf("government", "E1:E2", "E3:E3"),
    ]
    groups = [_group("all_workers", "All workers", "A3", "B3:E3")]
    env = _labeled_env(tmp_path, rows, headers, groups)
    try:
        parent = env.operators.read_group_header_values("stats", "all_workers", "private")
        assert list(parent[0]["values"].items()) == [
            ("private_total", 128), ("private_households", 1), ("private_other", 127),
        ]
        assert parent[0]["other_fields"] == {"group": "All workers", "government": 21}

        leaf = env.operators.read_group_header_values("stats", "all_workers", "private_other")
        assert leaf[0]["values"] == {"private_other": 127}
    finally:
        env.workbook.close()


def test_group_header_values_ignore_cell_formatting(tmp_path: Path):
    def indent(cell):
        cell.alignment = openpyxl.styles.Alignment(indent=2)

    def bold(cell):
        cell.font = openpyxl.styles.Font(bold=True)

    plain = _labeled_env(tmp_path / "plain", _SECTION_ROWS, _SECTION_HEADERS, [_SECTION_GROUP])
    styled = _labeled_env(
        tmp_path / "styled", _SECTION_ROWS, _SECTION_HEADERS, [_SECTION_GROUP],
        styles={"A3": bold, "A4": indent, "A5": indent},
    )
    try:
        assert (
            styled.operators.read_group_header_values("stats", "all_ages", "employed")
            == plain.operators.read_group_header_values("stats", "all_ages", "employed")
        )
    finally:
        plain.workbook.close()
        styled.workbook.close()


def test_group_header_values_reject_unsupported_or_unknown_inputs(tmp_path: Path):
    groups = [_SECTION_GROUP, _group("by_column", "By column", "B1", "B2:B6", axis="column")]
    env = _labeled_env(tmp_path, _SECTION_ROWS, _SECTION_HEADERS, groups)
    try:
        with pytest.raises(ValueError, match="row-axis"):
            env.operators.read_group_header_values("stats", "by_column", "employed")
        with pytest.raises(ValueError, match="missing_group"):
            env.operators.read_group_header_values("stats", "missing_group", "employed")
        with pytest.raises(ValueError, match="missing_header"):
            env.operators.read_group_header_values("stats", "all_ages", "missing_header")
    finally:
        env.workbook.close()


def test_intersect_can_leave_out_the_group_label_row(tmp_path: Path):
    env = _labeled_env(tmp_path, _SECTION_ROWS, _SECTION_HEADERS, [_SECTION_GROUP])
    try:
        default = env.operators.intersect_group_with_header("stats", "all_ages", "employed")
        assert range_to_a1(default) == "B2:B6"
        without_label = env.operators.intersect_group_with_header(
            "stats", "all_ages", "employed", include_group_label_row=False
        )
        assert range_to_a1(without_label) == "B3:B6"
    finally:
        env.workbook.close()


def test_intersect_without_label_row_is_unchanged_when_the_label_sits_outside(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        assert range_to_a1(env.operators.intersect_group_with_header(
            "employment", "men", "value", include_group_label_row=False
        )) == "B3:B4"
    finally:
        env.workbook.close()


def test_group_label_row_is_the_row_holding_the_label_even_when_group_range_spans_the_section(tmp_path: Path):
    rows = [
        ["Metric", "Value"],
        ["Men", 170],
        ["Participation rate", 70],
        ["Population", 100],
    ]
    headers = [_leaf("metric", "A1", "A2:A4"), _leaf("value", "B1", "B2:B4")]
    groups = [_group("men", "Men", "A2:B4", "B2:B4")]
    env = _labeled_env(tmp_path, rows, headers, groups)
    try:
        records = env.operators.read_group_header_values("stats", "men", "value")
        assert [record["is_group_label_row"] for record in records] == [True, False, False]
        assert range_to_a1(env.operators.intersect_group_with_header(
            "stats", "men", "value", include_group_label_row=False
        )) == "B3:B4"
    finally:
        env.workbook.close()


def test_group_header_values_leave_out_headers_that_do_not_cover_the_row(tmp_path: Path):
    rows = [
        ["Item", "Amount"],
        ["Total", 30],
        ["Part A", 10],
        ["Part B", 20],
        [None, None],
        ["Note: figures in dollars", None],
    ]
    headers = [
        _leaf("item", "A1", "A2:A4"),
        _leaf("amount", "B1", "B2:B4"),
        _leaf("note", "A6", "A6:A6"),
    ]
    env = _labeled_env(tmp_path, rows, headers, [_group("total", "Total", "A2", "B2:B4")])
    try:
        records = env.operators.read_group_header_values("stats", "total", "amount")
        assert [record["other_fields"] for record in records] == [
            {"item": "Total"}, {"item": "Part A"}, {"item": "Part B"},
        ]
    finally:
        env.workbook.close()


def test_group_label_matches_a_float_cell_holding_a_whole_number(tmp_path: Path):
    rows = [["Year", "Month", "Sales"], [2023.0, "Jan", 100], [None, "Feb", 110]]
    headers = [_leaf("year", "A1", "A2:A3"), _leaf("month", "B1", "B2:B3"), _leaf("sales", "C1", "C2:C3")]
    env = _labeled_env(tmp_path, rows, headers, [_group("y2023", "2023", "A2", "B2:C3")])
    try:
        records = env.operators.read_group_header_values("stats", "y2023", "sales")
        assert [record["is_group_label_row"] for record in records] == [True, False]
    finally:
        env.workbook.close()


def test_group_label_is_not_matched_inside_the_group_data_columns(tmp_path: Path):
    rows = [["Metric", "Value"], ["2023", None], ["Revenue", 2023], ["Cost", 7]]
    headers = [_leaf("metric", "A1", "A2:A4"), _leaf("value", "B1", "B2:B4")]
    env = _labeled_env(tmp_path, rows, headers, [_group("y2023", "2023", "A2:B4", "B3:B4")])
    try:
        records = env.operators.read_group_header_values("stats", "y2023", "value")
        assert [record["row"] for record in records] == [3, 4]
        assert not any(record["is_group_label_row"] for record in records)
    finally:
        env.workbook.close()


def test_group_header_values_key_a_multi_column_leaf_by_column(tmp_path: Path):
    rows = [["Region", "Rates", None], ["All", 1.5, 2.5], ["North", 1.0, 2.0]]
    headers = [_leaf("region", "A1", "A2:A3"), _leaf("rates", "B1:C1", "B2:C3")]
    env = _labeled_env(tmp_path, rows, headers, [_group("all", "All", "A2", "B2:C3")])
    try:
        records = env.operators.read_group_header_values("stats", "all", "rates")
        assert [record["values"] for record in records] == [
            {"rates:B": 1.5, "rates:C": 2.5}, {"rates:B": 1.0, "rates:C": 2.0},
        ]
    finally:
        env.workbook.close()


def test_intersect_refuses_to_cut_a_label_row_between_records(tmp_path: Path):
    rows = [["Item", "Amount"], ["Part A", 10], ["Total", 30], ["Part B", 20]]
    headers = [_leaf("item", "A1", "A2:A4"), _leaf("amount", "B1", "B2:B4")]
    env = _labeled_env(tmp_path, rows, headers, [_group("total", "Total", "A3", "B2:B4")])
    try:
        with pytest.raises(ValueError, match="read_group_header_values"):
            env.operators.intersect_group_with_header("stats", "total", "amount", include_group_label_row=False)
        records = env.operators.read_group_header_values("stats", "total", "amount")
        assert [record["is_group_label_row"] for record in records] == [False, True, False]
    finally:
        env.workbook.close()


def test_label_row_trimming_and_reads_reject_non_row_groups_and_headers_without_data(tmp_path: Path):
    # A header known only by a header_range that spans the records still has no cells to read.
    headers = _SECTION_HEADERS + [{**_leaf("empty", "D1:D6", "D2:D6"), "data_range": None}]
    groups = [_SECTION_GROUP, _group("by_column", "Employed", "B1", "B2:B6", axis="column")]
    env = _labeled_env(tmp_path, _SECTION_ROWS, headers, groups)
    try:
        with pytest.raises(ValueError, match="row-axis"):
            env.operators.intersect_group_with_header(
                "stats", "by_column", "employed", include_group_label_row=False
            )
        with pytest.raises(ValueError, match="no data_range"):
            env.operators.read_group_header_values("stats", "all_ages", "empty")
    finally:
        env.workbook.close()


def _two_sheet_env(tmp_path: Path):
    workbook_path, structure_path = _fixture(tmp_path)
    workbook = openpyxl.load_workbook(workbook_path)
    other = workbook.create_sheet("Table 2")
    other["A1"] = "Other value"
    workbook.save(workbook_path)
    return QAEnvironment(str(structure_path), str(workbook_path))


def test_a1_addresses_may_name_their_sheet(tmp_path: Path):
    from TableAgent.utils import parse_a1_range, range_to_a1

    assert (parse_a1_range("Sheet1!B3:C4").sheet, range_to_a1(parse_a1_range("Sheet1!B3:C4"))) == ("Sheet1", "B3:C4")
    assert parse_a1_range("'Table 2'!A1").sheet == "Table 2"
    assert parse_a1_range("'O''Brien'!A1").sheet == "O'Brien"
    assert parse_a1_range("A1", "Sheet1").sheet == "Sheet1"

    env = _two_sheet_env(tmp_path)
    try:
        assert env.operators.read_range("Sheet1!B3:C3") == [[70, 60]]
        assert env.operators.read_range("'Table 2'!A1") == [["Other value"]]
        # Sheet names match ignoring case and spacing.
        assert env.operators.read_range("A1", sheet=" table 2 ") == [["Other value"]]
        # With several sheets an unknown name fails instead of reading the active sheet.
        with pytest.raises(ValueError, match="Table 2"):
            env.operators.read_range("employment!A1")
    finally:
        env.workbook.close()


def test_intersection_falls_back_to_header_columns_when_its_rows_miss_the_group(tmp_path: Path):
    from TableAgent.utils import range_to_a1

    workbook_path, structure_path = _fixture(tmp_path)
    structure = yaml.safe_load(structure_path.read_text(encoding="utf-8"))
    # The header's recorded data_range stops before the women section starts.
    structure["employment"]["headers"][1]["data_range"] = "B2:B4"
    structure_path.write_text(yaml.safe_dump(structure, sort_keys=False), encoding="utf-8")
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        assert range_to_a1(env.operators.intersect_group_with_header("employment", "men", "value")) == "B3:B4"
        assert range_to_a1(env.operators.intersect_group_with_header("employment", "women", "value")) == "B6"
        assert [r["values"] for r in env.operators.read_group_header_values("employment", "women", "value")] == [
            {"value": 65}
        ]
    finally:
        env.workbook.close()


def test_read_cell_formats_reports_fill_font_and_indent(tmp_path: Path):
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook_path, structure_path = _fixture(tmp_path)
    workbook = openpyxl.load_workbook(workbook_path)
    sheet = workbook.active
    sheet["A2"].fill = PatternFill("solid", fgColor="FF0000FF")
    sheet["A2"].font = Font(bold=True, color="FFFF0000")
    sheet["A3"].alignment = Alignment(indent=2)
    workbook.save(workbook_path)
    env = QAEnvironment(str(structure_path), str(workbook_path))
    try:
        men, rate = env.operators.read_cell_formats("Sheet1!A2:A3")
        assert men == {
            "cell": "A2", "value": "Men", "fill": "FF0000FF", "font_color": "FFFF0000",
            "bold": True, "italic": False, "indent": 0,
        }
        assert (rate["cell"], rate["fill"], rate["bold"], rate["indent"]) == ("A3", None, False, 2)
        assert "operators.read_cell_formats(" in env.operators.operator_catalog()
    finally:
        env.workbook.close()
