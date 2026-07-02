from __future__ import annotations
from typing import Union, List, Any
import pandas as pd
from TableAgent.schema.range import CellRange
from TableAgent.utils import parse_a1_range, read_excel_range
from TableAgent.QA.operators.base_operator import BaseOperator

class WorkbookOperator(BaseOperator):
    """Operator for reading cell values and converting to structures like DataFrames."""
    name = "workbook"
    description = "Read workbook cells from an A1 range or CellRange; use selective ranges to keep observations compact."
    examples = (
        "operators.read_table_as_dataframe(table_id, has_headers=False) -> pandas.DataFrame",
        "operators.read_range(range_or_a1, sheet='') -> list[list[Any]]",
        "operators.read_range_flat(range_or_a1, sheet='') -> list[Any]",
        "operators.read_range_as_dataframe(range_or_a1, sheet='', has_headers=True) -> pandas.DataFrame",
    )

    def read_range(self, range_or_a1: Union[CellRange, str], sheet: str = "") -> List[List[Any]]:
        """Read raw cell values from the workbook range."""
        if isinstance(range_or_a1, str):
            cell_range = parse_a1_range(range_or_a1, sheet)
        else:
            cell_range = range_or_a1
            if sheet:
                cell_range = CellRange(
                    cell_range.start_row,
                    cell_range.start_col,
                    cell_range.end_row,
                    cell_range.end_col,
                    sheet
                )
        
        sheet_name = cell_range.sheet
        if not sheet_name:
            sheet_name = self.env.get_active_sheet_name()
        
        sheet_obj = self.env.get_sheet(sheet_name)
        if not sheet_obj:
            sheet_obj = self.env.get_active_sheet()
            
        return read_excel_range(sheet_obj, cell_range)

    def read_range_flat(self, range_or_a1: Union[CellRange, str], sheet: str = "") -> List[Any]:
        """Read values from range and flatten them into a single-dimensional list."""
        rows = self.read_range(range_or_a1, sheet)
        return [cell for row in rows for cell in row]

    def read_range_as_dataframe(self, range_or_a1: Union[CellRange, str], sheet: str = "", has_headers: bool = True) -> pd.DataFrame:
        """Read range and convert to a pandas DataFrame."""
        data = self.read_range(range_or_a1, sheet)
        if not data:
            return pd.DataFrame()
        if has_headers and len(data) > 1:
            headers = [str(h) if h is not None else f"Col{i}" for i, h in enumerate(data[0])]
            return pd.DataFrame(data[1:], columns=headers)
        return pd.DataFrame(data)

if __name__ == "__main__":
    import argparse
    from TableAgent.environment.qa_env import QAEnvironment

    parser = argparse.ArgumentParser(description="Smoke-test workbook range operators.")
    parser.add_argument("--structure", default="sample/structure.yaml")
    parser.add_argument("--workbook", default="sample/QA_sample.xlsx")
    parser.add_argument("--range", default="A1:C5")
    args = parser.parse_args()

    env = QAEnvironment(args.structure, args.workbook)
    op = WorkbookOperator(env)
    values = op.read_range(args.range)
    flat = op.read_range_flat(args.range)
    print(f"range={args.range}")
    print(f"rows={len(values)} cols={len(values[0]) if values else 0}")
    print(f"first_row={values[0] if values else []}")
    print(f"flat_preview={flat[:8]}")
