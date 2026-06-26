from __future__ import annotations
from typing import Any, List, Optional, Union
import pandas as pd
from TableAgent.schema.header import Header
from TableAgent.schema.range import AxisSelection, CellRange
from TableAgent.QA.operators.base_operator import BaseOperator
from TableAgent.QA.operators.structure_operator import StructureOperator
from TableAgent.QA.operators.range_operator import RangeOperator
from TableAgent.QA.operators.workbook_operator import WorkbookOperator
from TableAgent.QA.operators.filter_operator import FilterOperator

class TableOperators(BaseOperator):
    """
    Facade operator that delegates to structure, range, and workbook operators.
    Exposes a unified interface for code executed within the environment.
    """
    name = "table"
    description = "Unified facade exposed to agents as `operators`."

    def __init__(self, env: Any):
        super().__init__(env)
        self._structure = StructureOperator(env)
        self._range = RangeOperator(env)
        self._workbook = WorkbookOperator(env)
        self._filter = FilterOperator(env)
        self._catalog_sources = (self._structure, self._range, self._workbook, self._filter)

    def operator_catalog(self) -> str:
        """Return prompt-ready descriptions and examples for the exposed operators."""
        sections = [op.describe() for op in self._catalog_sources]
        sections.append(
            "calculation: Write normal Python/pandas/numpy code for arithmetic, aggregation, "
            "filtering, joins, formatting, and final answer construction. For example: "
            "`avg = sum(values) / len(values) if values else None`, "
            "`df.groupby(...)`, `max(rows, key=...)`."
        )
        sections.append(
            "workspace: Use `env.preview_variable(name, rows=5)` and "
            "`env.get_history(last_n=3, max_output_len=800)` for compact observations."
        )
        return "\n\n".join(sections)

    # Structure/header operators
    def list_tables(self) -> List[str]:
        return self._structure.list_tables()

    def list_headers(self, table_id: str) -> List[Header]:
        return self._structure.list_headers(table_id)

    def find_headers(self, table_id: str, query: str) -> List[Header]:
        return self._structure.find_headers(table_id, query)

    def get_header(self, table_id: str, header_id: str) -> Optional[Header]:
        return self._structure.get_header(table_id, header_id)

    # Range operators
    def resolve_ranges(
        self,
        op: str,
        range1: Union[CellRange, str],
        range2: Union[CellRange, str],
        sheet: str = ""
    ) -> Union[List[CellRange], CellRange, None]:
        return self._range.resolve_ranges(op, range1, range2, sheet)

    def union(self, range1: Union[CellRange, str], range2: Union[CellRange, str], sheet: str = "") -> List[CellRange]:
        return self._range.union(range1, range2, sheet)

    def intersection(self, range1: Union[CellRange, str], range2: Union[CellRange, str], sheet: str = "") -> Optional[CellRange]:
        return self._range.intersection(range1, range2, sheet)

    def crossing(self, range1: Union[CellRange, str], range2: Union[CellRange, str], sheet: str = "") -> Optional[CellRange]:
        return self._range.crossing(range1, range2, sheet)

    def difference(self, range1: Union[CellRange, str], range2: Union[CellRange, str], sheet: str = "") -> List[CellRange]:
        return self._range.difference(range1, range2, sheet)

    def selection_intersection(self, *selections: AxisSelection) -> AxisSelection:
        return self._range.selection_intersection(*selections)

    def selection_union(self, *selections: AxisSelection) -> AxisSelection:
        return self._range.selection_union(*selections)

    def selection_difference(self, selection: AxisSelection, *others: AxisSelection) -> AxisSelection:
        return self._range.selection_difference(selection, *others)

    def project_selection(self, selection: AxisSelection, target_range: Union[CellRange, str], sheet: str = "") -> List[CellRange]:
        return self._range.project_selection(selection, target_range, sheet)

    # Workbook operators
    def read_range(self, range_or_a1: Union[CellRange, str], sheet: str = "") -> List[List[Any]]:
        return self._workbook.read_range(range_or_a1, sheet)

    def read_range_flat(self, range_or_a1: Union[CellRange, str], sheet: str = "") -> List[Any]:
        return self._workbook.read_range_flat(range_or_a1, sheet)

    def read_range_as_dataframe(self, range_or_a1: Union[CellRange, str], sheet: str = "", has_headers: bool = True) -> pd.DataFrame:
        return self._workbook.read_range_as_dataframe(range_or_a1, sheet, has_headers)

    # Value filter / sparse selection operators
    def filter_values(self, range_or_a1: Union[CellRange, str], **kwargs: Any) -> AxisSelection:
        return self._filter.filter_values(range_or_a1, **kwargs)

    def read_selection(self, selection: AxisSelection, target_range: Union[CellRange, str], sheet: str = "") -> List[Any]:
        return self._filter.read_selection(selection, target_range, sheet=sheet)

if __name__ == "__main__":
    import argparse
    from TableAgent.environment.qa_env import QAEnvironment

    parser = argparse.ArgumentParser(description="Smoke-test unified TableOperators facade.")
    parser.add_argument("--structure", default="sample/structure.yaml")
    parser.add_argument("--workbook", default="sample/QA_sample.xlsx")
    parser.add_argument("--query", default="score")
    args = parser.parse_args()

    env = QAEnvironment(args.structure, args.workbook)
    op = TableOperators(env)
    table_id = env.default_table_id()
    matches = op.find_headers(table_id, args.query)
    print("operator_catalog:")
    print(op.operator_catalog())
    print(f"tables={op.list_tables()}")
    print(f"default_table={table_id}")
    print(f"query={args.query}")
    print(f"matches={[h.id for h in matches[:5]]}")
    if matches:
        values = op.read_range_flat(matches[0].data_range)
        print(f"first_match_range_values_preview={values[:8]}")
