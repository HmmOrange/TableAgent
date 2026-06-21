import json
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from table2img.core import _capture_with_chrome_cli, document_from_json, find_browsers, render_document, rows_from_markdown


class Table2ImgCoreTests(unittest.TestCase):
    def test_pillow_backend_renders_without_launching_browser(self):
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "table.png"
            document = document_from_json({"texts": [["Year", "Revenue"], ["2024", "100"]]})

            with patch("table2img.core._capture_with_chrome_cli") as browser_capture:
                result = render_document(document, output_path, backend="pillow")

            browser_capture.assert_not_called()
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(result.browser_path, Path("pillow"))

    def test_browser_discovery_deduplicates_candidates(self):
        browser = Path("browser.exe").resolve()
        with patch("table2img.core.shutil.which", return_value=str(browser)), patch.object(Path, "is_file", return_value=True):
            browsers = find_browsers()

        self.assertEqual(browsers.count(browser), 1)

    def test_chrome_capture_uses_isolated_profile_and_accepts_written_screenshot(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            html_path = root / "table.html"
            output_path = root / "table.png"
            html_path.write_text("<html><body>table</body></html>", encoding="utf-8")
            output_path.write_bytes(b"stale")

            def fake_run(args, **kwargs):
                self.assertTrue(any(arg.startswith("--user-data-dir=") for arg in args))
                self.assertFalse(output_path.exists())
                output_path.write_bytes(b"png")
                return CompletedProcess(args, returncode=0, stdout="", stderr="")

            with patch("table2img.core.subprocess.run", side_effect=fake_run):
                _capture_with_chrome_cli(
                    Path("chrome.exe"),
                    html_path,
                    output_path,
                    width=800,
                    height=600,
                    scale=2.0,
                    timeout_seconds=10,
                )

            self.assertEqual(output_path.read_bytes(), b"png")

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

    def test_document_from_xlsx_coordinates(self):
        import openpyxl
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from table2img.core import document_from_xlsx

        with TemporaryDirectory() as tmpdir:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "SheetTest"
            ws["B2"] = "ValB2"
            ws["C3"] = "ValC3"
            
            xlsx_path = Path(tmpdir) / "test.xlsx"
            wb.save(xlsx_path)
            
            doc = document_from_xlsx(xlsx_path, add_coordinates=True)
            
            self.assertIn("B", doc.html)
            self.assertIn("C", doc.html)
            self.assertIn("2", doc.html)
            self.assertIn("3", doc.html)
            self.assertIn('class="excel-coord"', doc.html)
            self.assertIn("ValB2", doc.html)
            self.assertIn("ValC3", doc.html)


if __name__ == "__main__":
    unittest.main()
