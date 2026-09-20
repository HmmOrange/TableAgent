from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Union
import pandas as pd
from TableAgent.domain.group import (
    SECTION_COLUMN,
    SECTION_LABEL_ROW_COLUMN,
    SECTION_METADATA_COLUMNS,
    StructureGroup,
)
from TableAgent.domain.ranges import AxisSelection, Cell, CellRange
from TableAgent.domain.structure import Header
from TableAgent.stages.qa.operators.base_operator import BaseOperator
from TableAgent.stages.qa.operators.structure_operator import StructureOperator
from TableAgent.stages.qa.operators.range_operator import RangeOperator
from TableAgent.stages.qa.operators.workbook_operator import WorkbookOperator
from TableAgent.stages.qa.operators.filter_operator import FilterOperator
from TableAgent.stages.qa.operators.multitab_operator import MultiTableOperator
from TableAgent.stages.qa.operators.table_routing_operator import TableRef
from TableAgent.stages.retrieval import TableCandidate

class TableOperators(BaseOperator):
    """
    Facade operator that delegates to structure, range, and workbook operators.
    Exposes a unified interface for code executed within the environment.
    """
    name = "table"
    description = "Unified facade exposed to agents as `operators`."
    examples = (
        "operators.read_group(table_id, group_id) -> list[list[Any]]  # the group's visible label cells",
        "operators.read_group_data(table_id, group_id) -> list[list[Any]]  # the records the group owns",
        "operators.find_in_group(table_id, group_id, query) -> list[tuple[Cell, Any]]",
        "operators.resolve_group_rows(df, table_id, group_id) -> list[int]  # positional rows owned by the group",
        "operators.group_row_mask(df, table_id, group_id) -> pandas.Series  # boolean row mask for the group",
        "operators.filter_in_group(table_id, group_id, header_id, gte=1) -> AxisSelection",
        "operators.selection_union(sel1, sel2, ...) -> AxisSelection",
        "operators.selection_difference(sel1, sel2) -> AxisSelection",
        "operators.find_table(query) -> str | None",
        "operators.groupby_table(table_id, by=..., aggregations=...) -> pandas.DataFrame",
        f"read_table_as_dataframe adds `{SECTION_COLUMN}` (owning group label) and "
        f"`{SECTION_LABEL_ROW_COLUMN}` (True on a group's label row, which is metadata, not a record). "
        f"Exclude label rows before counting or aggregating: `df[~df['{SECTION_LABEL_ROW_COLUMN}']]`.",
    )

    def __init__(self, env: Any):
        super().__init__(env)
        self._structure = StructureOperator(env)
        self._range = RangeOperator(env)
        self._workbook = WorkbookOperator(env)
        self._filter = FilterOperator(env)
        self._multitab = MultiTableOperator(env)
        self._catalog_sources = (self._structure, self._range, self._workbook, self._filter, self._multitab)

    def operator_catalog(self) -> str:
        """Return prompt-ready descriptions and examples for the exposed operators."""
        sections = [op.describe() for op in self._catalog_sources]
        # The facade defines several operators of its own; without this they never reach any prompt.
        sections.append(self.describe())
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
        sections.append(self._return_type_catalog())
        return "\n\n".join(sections)

    @staticmethod
    def _return_type_catalog() -> str:
        """List what the objects these operators return actually carry.

        Derived from the types themselves, so it cannot drift from the code the way a
        hand-written list would, and so an attribute is documented once rather than
        being discovered through a failed execution round.
        """
        from TableAgent.domain.introspection import public_attribute_names

        specimens = (
            ("Header", Header("", "", "", "", None, None)),
            ("StructureGroup", StructureGroup("", "", "", "", None, None)),
            ("CellRange", CellRange(1, 1, 1, 1)),
            ("Cell", Cell(1, 1)),
            ("TableRef", TableRef("")),
        )
        lines = [
            "return types: attributes carried by the objects the operators above return. "
            "Reaching for anything else raises with the nearest valid name."
        ]
        for name, specimen in specimens:
            attributes = ", ".join(public_attribute_names(specimen))
            lines.append(f"- {name}: {attributes}")
        lines.append(
            "- a Header's or StructureGroup's `header_range`/`data_range`/`group_range` "
            "is None when the structure did not record one; check before reading `.sheet` "
            "or any other range attribute off it."
        )
        return "\n".join(lines)

    # Structure/header operators
    def list_tables(self) -> List[str]:
        return self._structure.list_tables()

    def list_headers(self, table_id: str) -> List[Header]:
        headers = self._structure.list_headers(table_id)
        self._note_probe(bool(headers), f"list_headers({table_id!r})")
        return headers

    def find_headers(self, table_id: str, query: str) -> List[Header]:
        headers = self._structure.find_headers(table_id, query)
        self._note_probe(bool(headers), f"find_headers({table_id!r}, {query!r})")
        return headers

    def get_header(self, table_id: str, header_id: str) -> Header:
        try:
            header = self._structure.get_header(table_id, header_id)
        except KeyError:
            self._note_probe(False, f"get_header({table_id!r}, {header_id!r})")
            raise
        self._note_probe(True, f"get_header({table_id!r}, {header_id!r})")
        return header

    def has_header(self, table_id: str, header_id: str) -> bool:
        return self._structure.has_header(table_id, header_id)

    def resolve_header_columns(self, table_id: str, header_id: str) -> List[str]:
        try:
            columns = self._structure.resolve_header_columns(table_id, header_id)
        except (KeyError, ValueError):
            self._note_probe(False, f"resolve_header_columns({table_id!r}, {header_id!r})")
            raise
        self._note_probe(bool(columns), f"resolve_header_columns({table_id!r}, {header_id!r})")
        return columns

    def list_groups(self, table_id: str) -> List[StructureGroup]:
        return self._structure.list_groups(table_id)

    def find_groups(
        self, table_id: str, query: str, *, limit: int = 5, min_overlap: int = 1
    ) -> List[StructureGroup]:
        groups = self._structure.find_groups(table_id, query, limit=limit, min_overlap=min_overlap)
        self._note_probe(bool(groups), f"find_groups({table_id!r}, {query!r})")
        return groups

    def get_group(self, table_id: str, group_id: str) -> StructureGroup:
        try:
            group = self._structure.get_group(table_id, group_id)
        except KeyError:
            self._note_probe(False, f"get_group({table_id!r}, {group_id!r})")
            raise
        self._note_probe(True, f"get_group({table_id!r}, {group_id!r})")
        return group

    def has_group(self, table_id: str, group_id: str) -> bool:
        return self._structure.has_group(table_id, group_id)

    def intersect_group_with_header(self, table_id: str, group_id: str, header_id: str) -> Optional[CellRange]:
        return self._structure.intersect_group_with_header(table_id, group_id, header_id)

    def _require_group(self, table_id: str, group_id: str) -> StructureGroup:
        return self.get_group(table_id, group_id)

    def read_group(self, table_id: str, group_id: str) -> List[List[Any]]:
        group = self._require_group(table_id, group_id)
        return self._workbook.read_range(group.group_range, expand_merged=True) if group.group_range else []

    def read_group_data(self, table_id: str, group_id: str) -> List[List[Any]]:
        group = self._require_group(table_id, group_id)
        return self._workbook.read_range(group.data_range) if group.data_range else []

    def find_in_group(self, table_id: str, group_id: str, query: str) -> list[tuple[Cell, Any]]:
        import re
        import unicodedata

        group = self._require_group(table_id, group_id)
        if group.group_range is None:
            return []
        normalized_query = " ".join(re.findall(
            r"[\w]+", unicodedata.normalize("NFKC", str(query)).casefold(), flags=re.UNICODE
        ))
        worksheet = self.env.get_sheet(group.group_range.sheet) or self.env.get_active_sheet()
        matches = []
        for row in worksheet.iter_rows(
            min_row=group.group_range.start_row,
            max_row=group.group_range.end_row,
            min_col=group.group_range.start_col,
            max_col=group.group_range.end_col,
        ):
            for cell in row:
                normalized_value = " ".join(re.findall(
                    r"[\w]+", unicodedata.normalize("NFKC", str(cell.value or "")).casefold(), flags=re.UNICODE
                ))
                if normalized_query and f" {normalized_query} " in f" {normalized_value} ":
                    matches.append((Cell(cell.row, cell.column), cell.value))
        return matches

    def read_table_as_dataframe(
        self,
        table_id: str,
        has_headers: bool = False,
        *,
        include_group_column: bool = True,
        drop_group_label_rows: bool = False,
    ) -> pd.DataFrame:
        """Read a table through its verified structure; see `_read_table_frame` for detail."""
        try:
            frame = self._read_table_frame(
                table_id,
                has_headers,
                include_group_column=include_group_column,
                drop_group_label_rows=drop_group_label_rows,
            )
        except Exception:
            self._note_probe(False, f"read_table_as_dataframe({table_id!r})")
            raise
        self._note_probe(
            frame is not None and not frame.empty, f"read_table_as_dataframe({table_id!r})"
        )
        return frame

    def _read_table_frame(
        self,
        table_id: str,
        has_headers: bool = False,
        *,
        include_group_column: bool = True,
        drop_group_label_rows: bool = False,
    ) -> pd.DataFrame:
        """Read the bounding range covered by a table's verified headers and data.

        The bounding box is derived from headers alone, so a row-axis group's visible
        label row (for example `Bargaining status` sitting above the rows it owns) lands
        in the frame as an ordinary record with empty measure columns. With
        `include_group_column` the frame carries `__section__`, the label of the group
        that owns each row, and `__section_label_row__`, True on those label rows, so a
        count or an aggregate can exclude them instead of silently absorbing them.
        `drop_group_label_rows` removes them outright; it renumbers the positional index,
        so leave it off when other code depends on row offsets.
        """
        headers = self.list_headers(table_id)
        ranges = [
            cell_range
            for header in headers
            for cell_range in (header.header_range, header.data_range)
            if cell_range is not None
        ]
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
            frame = self._workbook.read_range_as_dataframe(table_range, has_headers=False)
            return self._annotate_sections(
                frame, table_id, table_range.start_row,
                include_group_column=include_group_column,
                drop_group_label_rows=drop_group_label_rows,
            )

        header_ranges = [header.header_range for header in headers if header.header_range is not None]
        if not header_ranges:
            frame = self._workbook.read_range_as_dataframe(table_range, has_headers=False)
            return self._annotate_sections(
                frame, table_id, table_range.start_row,
                include_group_column=include_group_column,
                drop_group_label_rows=drop_group_label_rows,
            )

        data_start_row = max(header_range.end_row for header_range in header_ranges) + 1
        if data_start_row > table_range.end_row:
            return self._annotate_sections(
                pd.DataFrame(), table_id, data_start_row,
                include_group_column=include_group_column,
                drop_group_label_rows=drop_group_label_rows,
            )
        data_range = CellRange(
            data_start_row,
            table_range.start_col,
            table_range.end_row,
            table_range.end_col,
            sheet,
        )
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
            return self._annotate_sections(
                frame, table_id, data_range.start_row,
                include_group_column=include_group_column,
                drop_group_label_rows=drop_group_label_rows,
            )

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
        return self._annotate_sections(
            result, table_id, data_range.start_row,
            include_group_column=include_group_column,
            drop_group_label_rows=drop_group_label_rows,
        )

    # Group-to-DataFrame bridge
    @staticmethod
    def _row_span(cell_range: Optional[CellRange]) -> Optional[tuple[int, int]]:
        if cell_range is None:
            return None
        return cell_range.start_row, cell_range.end_row

    def _row_axis_groups(self, table_id: str) -> List[StructureGroup]:
        return [
            group
            for group in self.list_groups(table_id)
            if str(getattr(group, "axis", "") or "").strip().lower() == "row"
        ]

    def _annotate_sections(
        self,
        frame: pd.DataFrame,
        table_id: str,
        origin_row: int,
        *,
        include_group_column: bool,
        drop_group_label_rows: bool,
    ) -> pd.DataFrame:
        """Attach section metadata derived from the table's row-axis structure groups."""
        if frame is None:
            return frame
        if not include_group_column and not drop_group_label_rows:
            return frame

        groups = self._row_axis_groups(table_id)
        if not groups:
            # A table without row sections must stay byte-identical to the unannotated frame;
            # two all-empty metadata columns would be pure noise in every prompt that shows it.
            return frame
        if frame.empty:
            # Keep the column contract stable so group_row_mask reports an empty selection
            # rather than a missing-column error.
            frame = frame.copy()
            if include_group_column:
                frame[SECTION_COLUMN] = pd.Series(dtype="object")
                frame[SECTION_LABEL_ROW_COLUMN] = pd.Series(dtype="bool")
            return frame

        worksheet_rows = [origin_row + offset for offset in range(len(frame))]
        owners: list[Optional[str]] = [None] * len(frame)
        widths: list[Optional[int]] = [None] * len(frame)
        label_rows = [False] * len(frame)

        for group in groups:
            data_span = self._row_span(group.data_range)
            if data_span is not None:
                start, end = data_span
                width = end - start
                for index, row in enumerate(worksheet_rows):
                    # A narrower span wins so a nested section is not masked by its parent.
                    if start <= row <= end and (widths[index] is None or width < widths[index]):
                        owners[index] = group.label
                        widths[index] = width

        for group in groups:
            label_span = self._row_span(group.group_range)
            data_span = self._row_span(group.data_range)
            if label_span is None:
                continue
            start, end = label_span
            for index, row in enumerate(worksheet_rows):
                if not start <= row <= end:
                    continue
                # Some groups legitimately keep the label inside the block they own; only a
                # row that sits outside the owned data is pure metadata.
                if data_span is not None and data_span[0] <= row <= data_span[1]:
                    continue
                label_rows[index] = True
                if owners[index] is None:
                    owners[index] = group.label

        frame = frame.copy()
        if include_group_column:
            frame[SECTION_COLUMN] = owners
            frame[SECTION_LABEL_ROW_COLUMN] = label_rows
        if drop_group_label_rows and any(label_rows):
            keep = [not flag for flag in label_rows]
            frame = frame.loc[keep].reset_index(drop=True)
        return frame

    def resolve_group_rows(self, dataframe: pd.DataFrame, table_id: str, group_id: str) -> List[int]:
        """Positional row indices of `dataframe` that the group owns, label rows excluded."""
        mask = self.group_row_mask(dataframe, table_id, group_id)
        return [position for position, flag in enumerate(mask.tolist()) if flag]

    def group_row_mask(self, dataframe: pd.DataFrame, table_id: str, group_id: str) -> pd.Series:
        """Boolean row mask selecting the records a row-axis group owns.

        Mirrors `group_header_mask`, which scopes a DataFrame by header; this scopes it by
        worksheet section so a question about one block cannot pull rows from a sibling.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("group_row_mask requires a pandas DataFrame.")
        group = self._require_group(table_id, group_id)
        if str(getattr(group, "axis", "") or "").strip().lower() != "row":
            raise ValueError(
                f"Group {group_id!r} has axis {group.axis!r}; group_row_mask only applies to row-axis groups."
            )
        if SECTION_COLUMN not in dataframe.columns:
            raise KeyError(
                f"Column {SECTION_COLUMN!r} is missing. Build the frame with "
                f"operators.read_table_as_dataframe({table_id!r}, include_group_column=True)."
            )
        mask = dataframe[SECTION_COLUMN] == group.label
        if SECTION_LABEL_ROW_COLUMN in dataframe.columns:
            mask = mask & (~dataframe[SECTION_LABEL_ROW_COLUMN].astype(bool))
        return mask

    def filter_in_group(
        self,
        table_id: str,
        group_id: str,
        header_id: str,
        **conditions: Any,
    ) -> AxisSelection:
        """Filter one header's values restricted to the cells a group owns.

        This is the group-scoped counterpart of `filter_values(header.data_range, ...)`:
        the search range is the intersection of the group with the header, so a match
        cannot come from a row outside the requested section.
        """
        cell_range = self.intersect_group_with_header(table_id, group_id, header_id)
        if cell_range is None:
            raise ValueError(
                f"Group {group_id!r} and header {header_id!r} do not intersect in table {table_id!r}. "
                "Check that the group is row-axis and the header is column-axis (or the reverse)."
            )
        return self._filter.filter_values(cell_range, **conditions)

    def section_columns(self) -> tuple[str, ...]:
        """Names of the metadata columns injected by `read_table_as_dataframe`."""
        return SECTION_METADATA_COLUMNS

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

    # Structured-access gate
    def _note_probe(self, resolved: bool, detail: str) -> None:
        """Record whether a structure lookup produced anything, for the raw-read gate."""
        log = getattr(self.env, "structured_probe_log", None)
        if log is None:
            log = []
            setattr(self.env, "structured_probe_log", log)
        log.append((bool(resolved), detail))

    def _structure_is_unusable(self, sheet: str = "") -> bool:
        """True when no verified table can serve this read, so the gate opens on its own."""
        try:
            table_ids = self.list_tables()
        except Exception:
            return True
        if not table_ids:
            return True
        if sheet:
            wanted = str(sheet).strip().casefold()
            owning = [
                table_id
                for table_id in table_ids
                if str((self.env.get_table_structure(table_id) or {}).get("sheet", "")).strip().casefold()
                == wanted
            ]
            # A sheet that no verified table covers was never structured in the first place.
            if not owning:
                return True
            table_ids = owning
        return not any(self._structure.list_headers(table_id) for table_id in table_ids)

    def _require_failed_structured_probe(self, sheet: str = "") -> None:
        if not bool(getattr(self.env, "qa_raw_sheet_gate", False)):
            return
        log = getattr(self.env, "structured_probe_log", None) or []
        if any(not resolved for resolved, _ in log):
            return
        if self._structure_is_unusable(sheet):
            return
        attempted = ", ".join(detail for _, detail in log[-4:]) or "none"
        raise PermissionError(
            "read_sheet_as_dataframe is gated: the verified structure has not been shown to "
            "fail for this subtask yet. Resolve the field through the structure first -- "
            "operators.find_headers(table_id, query), operators.resolve_header_columns(...), "
            "operators.read_table_as_dataframe(table_id) -- and read the raw sheet only if "
            "one of those comes back empty or raises. Reading raw cells loses header "
            "ownership and group ranges, so a value read this way cannot be attributed to a "
            f"verified field. Structure lookups so far in this subtask: {attempted}."
        )

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
        """Read physical worksheet cells, once the verified structure has been shown to fail.

        Reading the raw sheet discards every guarantee the structure stage provides:
        header ownership, group ranges, merged-cell expansion. It is the right escape
        hatch for a table whose structure really is unusable, and the wrong default.
        The gate makes that distinction a runtime fact rather than a judgement call, so
        the same situation takes the same branch on every run.
        """
        self._require_failed_structured_probe(sheet)
        return self._workbook.read_sheet_as_dataframe(
            sheet,
            min_row=min_row,
            max_row=max_row,
            min_col=min_col,
            max_col=max_col,
        )

    # Value filter / sparse selection operators
    def filter_values(self, range_or_a1: Union[CellRange, str], **kwargs: Any) -> AxisSelection:
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
        return self._filter.read_selection(selection, target_range, sheet=sheet)

    # Multi-table routing, relational operations, and formula evaluation
    def find_tables(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[TableRef]:
        return self._multitab.find_tables(query, top_k=top_k, min_score=min_score)

    def find_table(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[TableRef]:
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
    from TableAgent.stages.qa.environment.qa_env import QAEnvironment

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
