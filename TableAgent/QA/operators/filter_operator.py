from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Iterable, List, Optional, Union

import pandas as pd

from TableAgent.QA.operators.base_operator import BaseOperator
from TableAgent.QA.operators.structure_operator import StructureOperator
from TableAgent.QA.operators.workbook_operator import WorkbookOperator
from TableAgent.schema.range import AxisSelection, CellRange
from TableAgent.utils import parse_a1_range, range_to_a1


Predicate = Callable[[Any], bool]


class FilterOperator(BaseOperator):
    """Operator for filtering data values into sparse row/column selections."""
    name = "filter"
    description = (
        "Filter a one-dimensional data field into row/column positions, then project those positions "
        "onto another field. The agent must reason about which header/range and conditions to use; "
        "this operator only applies explicit conditions."
    )
    examples = (
        "operators.filter_values(header.data_range, contains='Tran', ignore_accents=True) -> AxisSelection(axis='row', positions=(...))",
        "operators.filter_values(score_header.data_range, gte=80, lt=90) -> AxisSelection",
        "operators.filter_values(status_header.data_range, in_values=['open', 'pending']) -> AxisSelection",
        "operators.filter_values(date_header.data_range, predicate=lambda v: getattr(v, 'month', None) == 9) -> AxisSelection",
        "operators.selection_intersection(sel1, sel2) -> AxisSelection",
        "operators.project_selection(selection, score_header.data_range) -> list[CellRange]",
        "operators.read_selection(selection, score_header.data_range) -> list[Any]",
        "operators.group_header_mask(df, table_id, parent_header_id, equals=0, mode='any') -> pandas.Series",
    )

    def __init__(self, env: Any):
        super().__init__(env)
        self._workbook = WorkbookOperator(env)
        self._structure = StructureOperator(env)

    def group_header_mask(
        self,
        dataframe: pd.DataFrame,
        table_id: str,
        header_id: str,
        *,
        mode: str = "any",
        missing_matches: bool = False,
        equals: Any = None,
        not_equals: Any = None,
        in_values: Optional[Iterable[Any]] = None,
        contains: Optional[str] = None,
        startswith: Optional[str] = None,
        endswith: Optional[str] = None,
        regex: Optional[str] = None,
        gt: Any = None,
        gte: Any = None,
        lt: Any = None,
        lte: Any = None,
        between: Optional[tuple[Any, Any]] = None,
        predicate: Optional[Predicate] = None,
        case_sensitive: bool = False,
        ignore_accents: bool = False,
    ) -> pd.Series:
        """Apply one condition across every leaf column under a parent header."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("group_header_mask requires a pandas DataFrame.")
        mode = str(mode).strip().lower()
        if mode not in {"any", "all"}:
            raise ValueError("group_header_mask mode must be 'any' or 'all'.")

        column_ids = self._structure.resolve_header_columns(table_id, header_id)
        missing_columns = [column_id for column_id in column_ids if column_id not in dataframe.columns]
        if missing_columns:
            raise KeyError(
                f"Resolved header columns are missing from the DataFrame: {missing_columns}. "
                f"Available columns: {list(dataframe.columns)}"
            )

        conditions = {
            "equals": equals,
            "not_equals": not_equals,
            "in_values": in_values,
            "contains": contains,
            "startswith": startswith,
            "endswith": endswith,
            "regex": regex,
            "gt": gt,
            "gte": gte,
            "lt": lt,
            "lte": lte,
            "between": between,
            "predicate": predicate,
            "case_sensitive": case_sensitive,
            "ignore_accents": ignore_accents,
        }
        if not any(
            value is not None
            for key, value in conditions.items()
            if key not in {"case_sensitive", "ignore_accents"}
        ):
            raise ValueError("group_header_mask requires at least one value condition.")

        def matches(value: Any) -> bool:
            is_missing = value is None
            if not is_missing:
                try:
                    is_missing = bool(pd.isna(value))
                except (TypeError, ValueError):
                    is_missing = False
            if is_missing:
                return bool(missing_matches)
            return self._matches(value, **conditions)

        masks = pd.DataFrame(
            {column_id: dataframe[column_id].map(matches) for column_id in column_ids},
            index=dataframe.index,
        )
        return masks.any(axis=1) if mode == "any" else masks.all(axis=1)

    def filter_values(
        self,
        range_or_a1: Union[CellRange, str],
        *,
        equals: Any = None,
        not_equals: Any = None,
        in_values: Optional[Iterable[Any]] = None,
        contains: Optional[str] = None,
        startswith: Optional[str] = None,
        endswith: Optional[str] = None,
        regex: Optional[str] = None,
        gt: Any = None,
        gte: Any = None,
        lt: Any = None,
        lte: Any = None,
        between: Optional[tuple[Any, Any]] = None,
        predicate: Optional[Predicate] = None,
        case_sensitive: bool = False,
        ignore_accents: bool = False,
        sheet: str = "",
    ) -> AxisSelection:
        """
        Filter a 1D field and return matching absolute worksheet row/column positions.

        Column-like ranges return `axis='row'`; row-like ranges return `axis='col'`.
        For 2D data, select a single field first to avoid ambiguous row/column semantics.
        """
        cell_range = self._coerce_range(range_or_a1, sheet)
        axis = self._selection_axis(cell_range)
        values = self._workbook.read_range(cell_range)
        flat_values = [row[0] for row in values] if axis == "row" else (values[0] if values else [])

        positions = []
        for idx, value in enumerate(flat_values):
            if self._matches(
                value,
                equals=equals,
                not_equals=not_equals,
                in_values=in_values,
                contains=contains,
                startswith=startswith,
                endswith=endswith,
                regex=regex,
                gt=gt,
                gte=gte,
                lt=lt,
                lte=lte,
                between=between,
                predicate=predicate,
                case_sensitive=case_sensitive,
                ignore_accents=ignore_accents,
            ):
                position = cell_range.start_row + idx if axis == "row" else cell_range.start_col + idx
                positions.append(position)
        return AxisSelection(axis=axis, positions=tuple(positions), sheet=cell_range.sheet, source_range=cell_range)

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

    def project_selection(
        self,
        selection: AxisSelection,
        target_range_or_a1: Union[CellRange, str],
        *,
        sheet: str = "",
    ) -> List[CellRange]:
        target_range = self._coerce_range(target_range_or_a1, sheet)
        return selection.to_ranges(target_range)

    def read_selection(
        self,
        selection: AxisSelection,
        target_range_or_a1: Union[CellRange, str],
        *,
        sheet: str = "",
    ) -> List[Any]:
        values: List[Any] = []
        for cell_range in self.project_selection(selection, target_range_or_a1, sheet=sheet):
            values.extend(self._workbook.read_range_flat(cell_range))
        return values

    def _coerce_range(self, range_or_a1: Union[CellRange, str], sheet: str = "") -> CellRange:
        if isinstance(range_or_a1, str):
            return parse_a1_range(range_or_a1, sheet)
        if sheet:
            return CellRange(
                range_or_a1.start_row,
                range_or_a1.start_col,
                range_or_a1.end_row,
                range_or_a1.end_col,
                sheet,
            )
        return range_or_a1

    def _selection_axis(self, cell_range: CellRange) -> str:
        if cell_range.start_col == cell_range.end_col:
            return "row"
        if cell_range.start_row == cell_range.end_row:
            return "col"
        raise ValueError(
            "filter_values requires a one-dimensional range. "
            f"Got {range_to_a1(cell_range)}; select a single header/field first."
        )

    def _matches(
        self,
        value: Any,
        *,
        equals: Any,
        not_equals: Any,
        in_values: Optional[Iterable[Any]],
        contains: Optional[str],
        startswith: Optional[str],
        endswith: Optional[str],
        regex: Optional[str],
        gt: Any,
        gte: Any,
        lt: Any,
        lte: Any,
        between: Optional[tuple[Any, Any]],
        predicate: Optional[Predicate],
        case_sensitive: bool,
        ignore_accents: bool,
    ) -> bool:
        checks = []
        if equals is not None:
            checks.append(self._equals(value, equals, case_sensitive, ignore_accents))
        if not_equals is not None:
            checks.append(not self._equals(value, not_equals, case_sensitive, ignore_accents))
        if in_values is not None:
            normalized_values = {self._text(item, case_sensitive, ignore_accents) for item in in_values}
            checks.append(self._text(value, case_sensitive, ignore_accents) in normalized_values)
        if contains is not None:
            checks.append(self._text(contains, case_sensitive, ignore_accents) in self._text(value, case_sensitive, ignore_accents))
        if startswith is not None:
            checks.append(self._text(value, case_sensitive, ignore_accents).startswith(self._text(startswith, case_sensitive, ignore_accents)))
        if endswith is not None:
            checks.append(self._text(value, case_sensitive, ignore_accents).endswith(self._text(endswith, case_sensitive, ignore_accents)))
        if regex is not None:
            flags = 0 if case_sensitive else re.IGNORECASE
            checks.append(re.search(regex, str(value or ""), flags=flags) is not None)
        if gt is not None:
            checks.append(self._comparable(value) > self._comparable(gt))
        if gte is not None:
            checks.append(self._comparable(value) >= self._comparable(gte))
        if lt is not None:
            checks.append(self._comparable(value) < self._comparable(lt))
        if lte is not None:
            checks.append(self._comparable(value) <= self._comparable(lte))
        if between is not None:
            low, high = between
            comparable = self._comparable(value)
            checks.append(self._comparable(low) <= comparable <= self._comparable(high))
        if predicate is not None:
            checks.append(bool(predicate(value)))
        if not checks:
            raise ValueError(
                "filter_values requires at least one condition: equals, not_equals, in_values, "
                "contains, startswith, endswith, regex, numeric/date bounds, or predicate."
            )
        return all(checks)

    def _equals(self, value: Any, expected: Any, case_sensitive: bool, ignore_accents: bool) -> bool:
        try:
            direct = value == expected
            if isinstance(direct, bool) and direct:
                return True
        except Exception:
            pass
        return self._text(value, case_sensitive, ignore_accents) == self._text(
            expected,
            case_sensitive,
            ignore_accents,
        )

    def _text(self, value: Any, case_sensitive: bool, ignore_accents: bool) -> str:
        text = "" if value is None else str(value)
        if ignore_accents:
            text = unicodedata.normalize("NFKD", text)
            text = "".join(ch for ch in text if not unicodedata.combining(ch))
            text = text.replace("Đ", "D").replace("đ", "d")
        return text if case_sensitive else text.casefold()

    def _comparable(self, value: Any) -> Any:
        import datetime as _datetime

        if isinstance(value, (_datetime.datetime, _datetime.date, int, float)):
            return value
        if value is None:
            raise ValueError("Cannot compare None in filter_values.")
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return _datetime.datetime.strptime(text, fmt).date()
            except ValueError:
                pass
        try:
            return float(text)
        except ValueError:
            return text


if __name__ == "__main__":
    import argparse
    from TableAgent.environment.qa_env import QAEnvironment

    parser = argparse.ArgumentParser(description="Smoke-test generic value filtering and row/column selections.")
    parser.add_argument("--structure", default="sample/structure.yaml")
    parser.add_argument("--workbook", default="sample/QA_sample.xlsx")
    parser.add_argument("--filter-range", default="D3:D22")
    parser.add_argument("--contains", default="Tran")
    parser.add_argument("--target-range", default="G3:G22")
    args = parser.parse_args()

    env = QAEnvironment(args.structure, args.workbook)
    op = FilterOperator(env)
    text_selection = op.filter_values(args.filter_range, contains=args.contains, ignore_accents=True)
    print(f"text_selection={text_selection.positions}")
    print(f"target_ranges={[range_to_a1(r) for r in op.project_selection(text_selection, args.target_range)]}")
    print(f"target_values={op.read_selection(text_selection, args.target_range)}")
