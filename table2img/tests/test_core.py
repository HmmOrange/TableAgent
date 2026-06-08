import json
import unittest

from table2img.core import document_from_json, rows_from_markdown


class Table2ImgCoreTests(unittest.TestCase):
    def test_markdown_rows(self):
        rows = rows_from_markdown(
            """
            | A | B |
            |---|---|
            | 1 | 2 |
            """
        )
        self.assertEqual(rows, [["A", "B"], ["1", "2"]])

    def test_hitab_merged_region_becomes_table_span(self):
        document = document_from_json(
            {
                "title": "sample",
                "texts": [["Region", "", "2018"], ["Allowance", "100", "90"]],
                "merged_regions": [
                    {
                        "first_row": 0,
                        "last_row": 0,
                        "first_column": 0,
                        "last_column": 1,
                    }
                ],
            }
        )

        self.assertIn('colspan="2"', document.html)
        self.assertIn("Region", document.html)
        self.assertIn("2018", document.html)
        self.assertIn("90", document.html)

    def test_mulhi_json_table_html(self):
        payload = {"tables": ["<table><tr><td>A</td><td>B</td></tr></table>"]}
        document = document_from_json(payload)
        self.assertIn("<table>", document.html)
        self.assertIn("<td>A</td>", document.html)

    def test_json_rows_from_dicts(self):
        document = document_from_json(json.loads('[{"name":"A","score":1},{"name":"B","score":2}]'))
        self.assertIn("name", document.html)
        self.assertIn("score", document.html)
        self.assertIn("B", document.html)


if __name__ == "__main__":
    unittest.main()
