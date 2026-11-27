from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Union
import pandas as pd
from TableAgent.schema.header import Header
from TableAgent.schema.range import AxisSelection, CellRange
from TableAgent.QA.operators.base_operator import BaseOperator
from TableAgent.QA.operators.structure_operator import StructureOperator
from TableAgent.QA.operators.range_operator import RangeOperator
from TableAgent.QA.operators.workbook_operator import WorkbookOperator
from TableAgent.QA.operators.filter_operator import FilterOperator
from TableAgent.QA.operators.multitab_operator import MultiTableOperator
from TableAgent.QA.operators.media_operator import MEDIA_COLUMN, MediaOperator
from TableAgent.pipeline.retrieval import TableCandidate
from TableAgent.utils import parse_a1_range

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
        self._multitab = MultiTableOperator(env)
        self._media = MediaOperator(env)
        self._catalog_sources = (self._structure, self._range, self._workbook, self._filter, self._multitab)

    def operator_catalog(self) -> str:
        """Return prompt-ready descriptions and examples for the exposed operators."""
        sections = [op.describe() for op in self._catalog_sources]
        # Workbooks without pictures keep the catalog they always had.
        if self._media.has_media():
            sections.append(self._media.describe())
        else:
            brief = self._media.other_sheets_brief()
            if brief:
                sections.append(brief)
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

    def resolve_header_columns(self, table_id: str, header_id: str) -> List[str]:
        return self._structure.resolve_header_columns(table_id, header_id)

    def read_table_as_dataframe(self, table_id: str, has_headers: bool = False) -> pd.DataFrame:
        """Read the bounding range covered by a table's verified headers and data."""
        headers = self.list_headers(table_id)
        ranges = [
            cell_range
            for header in headers
            for cell_range in (header.header_range, header.data_range)
            if cell_range is not None
        ]
        if ranges:
            self._note_read(ranges[0].sheet)
        if not ranges:
            return pd.DataFrame()
        sheet = ranges[0].sheet
        table_range = CellRange(
            min(cell_range.start_row for cell_range in ranges),
            min(cell_range.start_col for cell_range in ranges),
            max(cell_range.end_row for cell_range in ranges),
            max(cell_range.end_col for cell_range in ranges),
            sheet,
        )
        if not has_headers:
            return self._with_range_media(
                self._workbook.read_range_as_dataframe(table_range, has_headers=False),
                table_range,
                table_range.start_row,
            )

        header_ranges = [header.header_range for header in headers if header.header_range is not None]
        if not header_ranges:
            return self._with_range_media(
                self._workbook.read_range_as_dataframe(table_range, has_headers=False),
                table_range,
                table_range.start_row,
            )

        bands = self._header_bands(headers)
        if len(bands) > 1:
            return self._read_table_blocks(bands, table_range)

        data_start_row = max(header_range.end_row for header_range in header_ranges) + 1
        if data_start_row > table_range.end_row:
            return pd.DataFrame()
        data_range = CellRange(
            data_start_row,
            table_range.start_col,
            table_range.end_row,
            table_range.end_col,
            sheet,
        )
        frame = self._logical_frame(headers, data_range)
        return self._with_range_media(frame, data_range, data_range.start_row)

    @staticmethod
    def _header_bands(headers: List[Header]) -> List[tuple[int, int, List[Header]]]:
        """Header blocks stacked down a table: ``(first header row, last header row, headers)``.

        Column headers whose rows touch or overlap form one block, and a header
        always shares the block of its sub-headers, however far below they sit.
        A table laid out with row headers is left as a single block.
        """
        with_ranges = [header for header in headers if header.header_range is not None]
        if not with_ranges or any(str(header.orientation).lower().startswith("row") for header in with_ranges):
            return [(0, 0, with_ranges)]
        # Each header starts with the rows of itself and every sub-header below it.
        spans: dict[int, tuple[int, int]] = {}

        def span_of(header: Header) -> tuple[int, int]:
            start, end = header.header_range.start_row, header.header_range.end_row  # type: ignore[union-attr]
            for child in header.sub_headers or []:
                if child.header_range is not None:
                    child_start, child_end = span_of(child)
                    start, end = min(start, child_start), max(end, child_end)
            spans[id(header)] = (start, end)
            return start, end

        for header in with_ranges:
            span_of(header)
        bands: List[tuple[int, int, List[Header]]] = []
        for header in sorted(with_ranges, key=lambda header: spans[id(header)]):
            start, end = spans[id(header)]
            if bands and start <= bands[-1][1] + 1:
                first, last, members = bands[-1]
                bands[-1] = (first, max(last, end), members + [header])
            else:
                bands.append((start, end, [header]))
        return bands

    def _read_table_blocks(self, bands: List[tuple[int, int, List[Header]]], table_range: CellRange) -> pd.DataFrame:
        """Rows of every header block, each named by its own headers.

        A block's rows run from below its headers to just above the next block's
        headers. The ``block`` column holds the block's first header label and
        the index is the sheet row number.
        """
        frames = []
        for position, (_, last_header_row, members) in enumerate(bands):
            start = last_header_row + 1
            end = bands[position + 1][0] - 1 if position + 1 < len(bands) else table_range.end_row
            if start > end:
                continue
            data_range = CellRange(start, table_range.start_col, end, table_range.end_col, table_range.sheet)
            frame = self._logical_frame(members, data_range)
            frame.index = pd.RangeIndex(start, end + 1)
            # The block is named by its top-left header.
            first = min(members, key=lambda header: (header.header_range.start_row, header.header_range.start_col))  # type: ignore[union-attr]
            frame.insert(0, "block", first.label)
            # object dtype keeps whole numbers whole when another block has no value in that column.
            frames.append(self._with_range_media(frame, data_range, start).astype(object))
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, sort=False)
        if MEDIA_COLUMN in combined.columns:
            # Keep the picture column last, as single-block tables have it.
            combined = combined[[column for column in combined.columns if column != MEDIA_COLUMN] + [MEDIA_COLUMN]]
        # Fields one block does not have are None, not NaN.
        return combined.astype(object).where(combined.notna(), None)

    def _logical_frame(self, headers: List[Header], data_range: CellRange) -> pd.DataFrame:
        """``data_range`` read as one column per header id."""
        rows = self._workbook.read_range(data_range, expand_merged=True)
        column_ids = []
        for column in range(data_range.start_col, data_range.end_col + 1):
            candidates = [
                header
                for header in headers
                if header.header_range is not None
                and header.header_range.start_col <= column <= header.header_range.end_col
            ]
            candidates.sort(key=lambda header: header.header_range.end_col - header.header_range.start_col)  # type: ignore[union-attr]
            column_ids.append(candidates[0].id if candidates else f"col_{column}")
        frame = pd.DataFrame(rows, columns=column_ids)
        # A verified header can span several physical columns. Collapse those columns into one
        # logical field so agents cannot accidentally select only the first duplicate column.
        if len(set(column_ids)) == len(column_ids):
            if not frame.columns.is_unique:
                raise ValueError("Logical table DataFrame contains duplicate column names.")
            return frame

        collapsed: dict[str, pd.Series] = {}
        for column_id in dict.fromkeys(column_ids):
            duplicate_columns = frame.loc[:, [value == column_id for value in column_ids]]
            if duplicate_columns.shape[1] == 1:
                collapsed[column_id] = duplicate_columns.iloc[:, 0]
                continue

            def combine_row(row: pd.Series) -> Any:
                values = []
                for value in row.tolist():
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        continue
                    text = str(value).strip()
                    if text and text not in values:
                        values.append(text)
                if not values:
                    return None
                if len(values) == 1:
                    original = next(
                        value
                        for value in row.tolist()
                        if value is not None and not (isinstance(value, float) and pd.isna(value))
                        and str(value).strip()
                    )
                    return original
                return "\n".join(values)

            collapsed[column_id] = duplicate_columns.apply(combine_row, axis=1)
        result = pd.DataFrame(collapsed, index=frame.index)
        if not result.columns.is_unique:
            raise ValueError("Logical table DataFrame contains duplicate column names after collapse.")
        return result

    def _with_media_column(
        self,
        frame: pd.DataFrame,
        sheet: str,
        rows: List[int],
        start_col: int,
        end_col: int,
    ) -> pd.DataFrame:
        """Add the picture tokens of each sheet row; rows without pictures leave the frame unchanged."""
        if frame.empty or MEDIA_COLUMN in frame.columns or len(rows) != len(frame.index):
            return frame
        tokens = self._media.tokens_for_rows(sheet, rows, start_col=start_col, end_col=end_col)
        if tokens is None:
            return frame
        # Rows that hold pictures are labelled by sheet row number, the `row` of `list_media`,
        # so a picture's row and the DataFrame row never need converting.
        frame = frame.set_axis(pd.Index(rows), axis=0)
        # A picture under a header is part of that header's value: a column named by the header
        # id shows the picture's tokens (after any text the cell holds), so selecting the column
        # keeps the picture. Numbers are left alone so they still compute.
        by_header = self._media.tokens_by_header(sheet, rows, start_col=start_col, end_col=end_col)
        for row, headers in by_header.items():
            for header_id, header_tokens in headers.items():
                if header_id not in frame.columns or frame.columns.tolist().count(header_id) != 1:
                    continue
                value = frame.at[row, header_id]
                joined = " ".join(header_tokens)
                if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
                    filled = joined
                elif isinstance(value, str):
                    filled = f"{value.strip()} {joined}"
                else:
                    continue
                if frame[header_id].dtype != object:
                    frame[header_id] = frame[header_id].astype(object)
                frame.at[row, header_id] = filled
        # object dtype keeps None for rows without pictures (a string column would hold NaN, which is truthy).
        return frame.assign(**{MEDIA_COLUMN: pd.Series(tokens, index=frame.index, dtype=object)})

    def _with_range_media(self, frame: pd.DataFrame, cell_range: CellRange, first_row: int) -> pd.DataFrame:
        sheet = cell_range.sheet or self.env.get_active_sheet_name()
        rows = list(range(first_row, first_row + len(frame.index)))
        return self._with_media_column(frame, sheet, rows, cell_range.start_col, cell_range.end_col)

    # Picture operators
    def list_media(self, sheet: str = "", table_id: Optional[str] = None) -> pd.DataFrame:
        return self._media.list_media(sheet=sheet, table_id=table_id)

    def find_media(self, query: str, sheet: str = "") -> pd.DataFrame:
        return self._media.find_media(query, sheet=sheet)

    def media_inventory(self, sheet: str) -> str:
        return self._media.sheet_inventory(sheet)

    def media_legend(self, answer: str) -> str:
        return self._media.answer_legend(answer)

    def find_same_media(self, media_id: str) -> pd.DataFrame:
        return self._media.find_same_media(media_id)

    def describe_media(self, media_id: str, question: str = "") -> str:
        return self._media.describe_media(media_id, question=question)

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
    def _note_read(self, sheet: Any = "", range_or_a1: Union[CellRange, str, None] = None) -> None:
        """Record the sheet a read touched, for the answer's sources."""
        name = str(sheet or "").strip()
        if not name and range_or_a1 is not None:
            try:
                parsed = parse_a1_range(range_or_a1, "") if isinstance(range_or_a1, str) else range_or_a1
                name = str(getattr(parsed, "sheet", "") or "")
            except Exception:
                name = ""
        if not name:
            try:
                name = self.env.get_active_sheet_name()
            except Exception:
                return
        note = getattr(self.env, "note_sheet_used", None)
        if callable(note):
            note(name)

    def read_range(self, range_or_a1: Union[CellRange, str], sheet: str = "") -> List[List[Any]]:
        self._note_read(sheet, range_or_a1)
        return self._workbook.read_range(range_or_a1, sheet)

    def read_range_flat(self, range_or_a1: Union[CellRange, str], sheet: str = "") -> List[Any]:
        self._note_read(sheet, range_or_a1)
        return self._workbook.read_range_flat(range_or_a1, sheet)

    def read_range_as_dataframe(self, range_or_a1: Union[CellRange, str], sheet: str = "", has_headers: bool = True) -> pd.DataFrame:
        self._note_read(sheet, range_or_a1)
        frame = self._workbook.read_range_as_dataframe(range_or_a1, sheet, has_headers)
        try:
            cell_range = parse_a1_range(range_or_a1, sheet) if isinstance(range_or_a1, str) else range_or_a1
        except Exception:
            return frame
        if sheet and not cell_range.sheet:
            cell_range = CellRange(cell_range.start_row, cell_range.start_col, cell_range.end_row, cell_range.end_col, sheet)
        span = cell_range.end_row - cell_range.start_row + 1
        # The first row became the column names when there was more than one row.
        first_row = cell_range.start_row + (1 if has_headers and span > 1 else 0)
        return self._with_range_media(frame, cell_range, first_row)

    def sheet_dimensions(self, sheet: str = "") -> dict[str, int | str]:
        return self._workbook.sheet_dimensions(sheet)

    def read_sheet_as_dataframe(
        self,
        sheet: str = "",
        *,
        min_row: int = 1,
        max_row: int | None = None,
        min_col: int = 1,
        max_col: int | None = None,
    ) -> pd.DataFrame:
        self._note_read(sheet)
        frame = self._workbook.read_sheet_as_dataframe(
            sheet,
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
        )
        sheet_name = sheet or self.env.get_active_sheet_name()
        # The frame is indexed by sheet row number.
        return self._with_media_column(
            frame,
            sheet_name,
            [int(row) for row in frame.index],
            min_col,
            min_col + len(frame.columns) - 1,
        )

    # Value filter / sparse selection operators
    def filter_values(self, range_or_a1: Union[CellRange, str], **kwargs: Any) -> AxisSelection:
        self._note_read(kwargs.get("sheet", ""), range_or_a1)
        return self._filter.filter_values(range_or_a1, **kwargs)

    def group_header_mask(
        self,
        dataframe: pd.DataFrame,
        table_id: str,
        header_id: str,
        **kwargs: Any,
    ) -> pd.Series:
        return self._filter.group_header_mask(dataframe, table_id, header_id, **kwargs)

    def read_selection(self, selection: AxisSelection, target_range: Union[CellRange, str], sheet: str = "") -> List[Any]:
        self._note_read(sheet, target_range)
        return self._filter.read_selection(selection, target_range, sheet=sheet)

    # Multi-table routing, relational operations, and formula evaluation
    def find_tables(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[str]:
        return self._multitab.find_tables(query, top_k=top_k, min_score=min_score)

    def find_table(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[str]:
        return self._multitab.find_table(query, top_k=top_k, min_score=min_score)

    def retrieve_tables(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[TableCandidate]:
        return self._multitab.retrieve_tables(query, top_k=top_k, min_score=min_score)

    def join_tables(self, left: str | pd.DataFrame, right: str | pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return self._multitab.join_tables(left, right, **kwargs)

    def union_tables(self, tables: Sequence[str | pd.DataFrame], **kwargs: Any) -> pd.DataFrame:
        return self._multitab.union_tables(tables, **kwargs)

    def groupby_table(
        self,
        table: str | pd.DataFrame,
        *,
        by: str | Sequence[str],
        aggregations: Mapping[str, str | Sequence[str]],
        dropna: bool = False,
        sort: bool = True,
    ) -> pd.DataFrame:
        return self._multitab.groupby_table(
            table,
            by=by,
            aggregations=aggregations,
            dropna=dropna,
            sort=sort,
        )

    def groupby(self, table: str | pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return self._multitab.groupby(table, **kwargs)

    def list_relations(self, table_id: str | None = None, *, category: str | None = None) -> list[dict[str, Any]]:
        return self._multitab.list_relations(table_id, category=category)

    def find_relation(self, query: str, *, table_id: str | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        return self._multitab.find_relation(query, table_id=table_id, top_k=top_k)

    def evaluate_formula(
        self,
        relation_id: str,
        *,
        target_cell: str | None = None,
        mutations: Mapping[str, Any] | None = None,
        table_id: str | None = None,
    ) -> dict[str, Any]:
        return self._multitab.evaluate_formula(
            relation_id,
            target_cell=target_cell,
            mutations=mutations,
            table_id=table_id,
        )

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
