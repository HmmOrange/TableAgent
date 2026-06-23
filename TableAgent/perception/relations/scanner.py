from __future__ import annotations

from typing import Any
import openpyxl


def scan_formulas(workbook: openpyxl.Workbook) -> list[dict[str, Any]]:
    """
    Scans formula cells across every worksheet's used range exactly once.
    Retains 'sheet', 'cell', 'row', 'col', and 'raw' internally.
    """
    scanned = []
    for sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
        for row_cells in sheet.iter_rows():
            for cell in row_cells:
                val = cell.value
                if val is not None and isinstance(val, str) and val.startswith("="):
                    scanned.append({
                        "sheet": sheet_name,
                        "cell": cell,
                        "row": cell.row,
                        "col": cell.column,
                        "raw": val,
                    })
    return scanned
