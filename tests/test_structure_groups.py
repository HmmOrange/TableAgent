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
    assert "group_range" in GROUP_USER_PROMPT_TEMPLATE
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
