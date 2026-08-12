import pytest
import yaml

from TableAgent.stages.structure.layout.parsing import (
    _is_valid_structure,
    extract_layout_structure,
    extract_layout_structure_result,
)


@pytest.mark.parametrize(
    "table_key",
    ["table1", "table_1", "revenue_summary", "Revenue Table", "Bang doanh thu"],
)
def test_layout_parser_accepts_arbitrary_table_keys(table_key: str):
    response = yaml.safe_dump(
        {
            "structure": {
                table_key: {
                    "id": "revenue_summary",
                    "name": "Revenue summary",
                    "headers": [
                        {
                            "id": "revenue",
                            "label": "Revenue",
                            "orientation": "column",
                            "header_range": "A1",
                            "data_range": "A2:A3",
                            "sub_headers": [],
                        }
                    ],
                }
            },
            "changelog": "Created the revenue table.",
            "remaining_directions": [],
        },
        sort_keys=False,
    )

    structure_text, _, _, _ = extract_layout_structure(response)

    assert _is_valid_structure(structure_text)
    assert table_key in yaml.safe_load(structure_text)


def test_layout_parser_ignores_non_table_mappings():
    response = yaml.safe_dump(
        {
            "structure": {
                "metadata": {"description": "Not a table"},
                "actual revenue": {
                    "headers": [
                        {
                            "label": "Revenue",
                            "header_range": "A1",
                            "data_range": "A2:A3",
                        }
                    ]
                },
            }
        },
        sort_keys=False,
    )

    structure_text, _, _, _ = extract_layout_structure(response)

    assert list(yaml.safe_load(structure_text)) == ["actual revenue"]


def test_layout_parser_rejects_only_fc11_header_with_empty_label():
    response = yaml.safe_dump({
        "structure": {
            "labor_statistics": {
                "id": "labor_statistics",
                "headers": [
                    {
                        "id": "row_index",
                        "label": "",
                        "description": "Row index numbers",
                        "orientation": "column",
                        "header_range": "A1:A3",
                        "data_range": "A5:A20",
                        "sub_headers": [],
                    },
                    {
                        "id": "year",
                        "label": "Year",
                        "orientation": "column",
                        "header_range": "B3:B3",
                        "data_range": "B5:B20",
                        "sub_headers": [],
                    },
                    {
                        "id": "civilian_labor_force",
                        "label": "Civilian labor force",
                        "orientation": "column",
                        "header_range": "C2:F3",
                        "data_range": "C5:F20",
                        "sub_headers": [],
                    },
                ],
            }
        }
    }, sort_keys=False)

    result = extract_layout_structure_result(response)
    structure = yaml.safe_load(result.structure_text)

    assert [header["id"] for header in structure["labor_statistics"]["headers"]] == [
        "year",
        "civilian_labor_force",
    ]
    assert result.rejected_headers == [
        "labor_statistics.headers[0] was rejected because label is empty (id='row_index', "
        "header_range='A1:A3', data_range='A5:A20'). The candidate was removed; add it back only "
        "if the workbook shows a meaningful label, using that exact visible text. Otherwise leave it "
        "omitted. Preserve all accepted headers."
    ]


def test_layout_parser_keeps_structure_invalid_when_every_header_is_rejected():
    response = yaml.safe_dump({
        "structure": {
            "table1": {
                "headers": [
                    {"id": "empty_1", "label": "", "header_range": "A1"},
                    {"id": "empty_2", "label": None, "header_range": "B1"},
                ]
            }
        }
    }, sort_keys=False)

    result = extract_layout_structure_result(response)

    assert result.structure_text == ""
    assert not _is_valid_structure(result.structure_text)
