import pytest
import yaml

from TableAgent.structure.layout.parsing import _is_valid_structure, extract_layout_structure


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
