from __future__ import annotations
from typing import Union, List, Optional
from TableAgent.schema.range import AxisSelection, CellRange
from TableAgent.utils import parse_a1_range
from TableAgent.QA.operators.base_operator import BaseOperator

class RangeOperator(BaseOperator):
    """Operator for resolving and combining spreadsheet cell ranges."""
    name = "range"
    description = "Combine or compare CellRange/A1 ranges, including same-orientation and cross-orientation cases."
    examples = (
        "operators.resolve_ranges('intersection', range1, range2, sheet='') -> CellRange | list[CellRange] | None",
        "operators.union(range1, range2, sheet='') -> list[CellRange]",
        "operators.intersection(range1, range2, sheet='') -> CellRange | None",
        "operators.crossing(range1, range2, sheet='') -> CellRange | None",
        "operators.difference(range1, range2, sheet='') -> list[CellRange]",
        "operators.selection_intersection(sel1, sel2, ...) -> AxisSelection",
        "operators.project_selection(selection, target_range) -> list[CellRange]",
    )

    def resolve_ranges(
        self,
        op: str,
        range1: Union[CellRange, str],
        range2: Union[CellRange, str],
        sheet: str = ""
    ) -> Union[List[CellRange], CellRange, None]:
        """
        Resolve/combine ranges using union, intersection, difference, or crossing.
        """
        r1 = parse_a1_range(range1, sheet) if isinstance(range1, str) else range1
        r2 = parse_a1_range(range2, sheet) if isinstance(range2, str) else range2

        op_lower = op.lower().strip()
        if op_lower in ("intersection", "crossing"):
            return r1.intersection(r2)
        elif op_lower == "union":
            return r1.union(r2)
        elif op_lower == "difference":
            return r1.difference(r2)
        else:
            raise ValueError(f"Unknown range resolution operator: {op}")

    def union(self, range1: Union[CellRange, str], range2: Union[CellRange, str], sheet: str = "") -> List[CellRange]:
        return self.resolve_ranges("union", range1, range2, sheet)

    def intersection(self, range1: Union[CellRange, str], range2: Union[CellRange, str], sheet: str = "") -> Optional[CellRange]:
        return self.resolve_ranges("intersection", range1, range2, sheet)

    def crossing(self, range1: Union[CellRange, str], range2: Union[CellRange, str], sheet: str = "") -> Optional[CellRange]:
        return self.resolve_ranges("crossing", range1, range2, sheet)

    def difference(self, range1: Union[CellRange, str], range2: Union[CellRange, str], sheet: str = "") -> List[CellRange]:
        return self.resolve_ranges("difference", range1, range2, sheet)

    def selection_intersection(self, *selections: AxisSelection) -> AxisSelection:
        if not selections:
            raise ValueError("selection_intersection requires at least one selection.")
        result = selections[0]
        for selection in selections[1:]:
            result = result.intersection(selection)
        return result

    def selection_union(self, *selections: AxisSelection) -> AxisSelection:
        if not selections:
            raise ValueError("selection_union requires at least one selection.")
        result = selections[0]
        for selection in selections[1:]:
            result = result.union(selection)
        return result

    def selection_difference(self, selection: AxisSelection, *others: AxisSelection) -> AxisSelection:
        result = selection
        for other in others:
            result = result.difference(other)
        return result

    def project_selection(self, selection: AxisSelection, target_range: Union[CellRange, str], sheet: str = "") -> List[CellRange]:
        target = parse_a1_range(target_range, sheet) if isinstance(target_range, str) else target_range
        return selection.to_ranges(target)

if __name__ == "__main__":
    from TableAgent.utils import range_to_a1

    op = RangeOperator(env=None)
    intersection = op.intersection("B3:D10", "C2:E20")
    union = op.union("B3:B10", "D3:D10")
    difference = op.difference("B3:D10", "C3:C10")
    rows = AxisSelection("row", (3, 5, 8))
    projected = op.project_selection(rows, "G3:G10")
    print(f"intersection={range_to_a1(intersection) if intersection else None}")
    print(f"union={[range_to_a1(r) for r in union]}")
    print(f"difference={[range_to_a1(r) for r in difference]}")
    print(f"projected_selection={[range_to_a1(r) for r in projected]}")
