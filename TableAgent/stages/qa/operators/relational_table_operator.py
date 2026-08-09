from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from TableAgent.QA.operators.base_operator import BaseOperator


class RelationalTableOperator(BaseOperator):
    """Perform deterministic relational operations over table-id or DataFrame inputs."""

    name = "multitab.relational"
    description = "Join tables, union matching schemas, and perform explicit grouped aggregations."
    examples = (
        "operators.join_tables('employees', 'salary', left_on='emp_id', right_on='emp_id') -> pandas.DataFrame",
        "operators.union_tables(['sales_q1', 'sales_q2'], source_column='source_table') -> pandas.DataFrame",
        "operators.groupby('sales', by='region', aggregations={'revenue': 'sum'}) -> pandas.DataFrame",
    )

    def join_tables(
        self,
        left: str | pd.DataFrame,
        right: str | pd.DataFrame,
        *,
        on: str | Sequence[str] | None = None,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        how: str = "inner",
        suffixes: tuple[str, str] = ("_left", "_right"),
        validate: str | None = None,
    ) -> pd.DataFrame:
        left_df = self._as_dataframe(left)
        right_df = self._as_dataframe(right)
        if on is None and left_on is None and right_on is None:
            shared = [column for column in left_df.columns if column in right_df.columns]
            if not shared:
                raise ValueError("No shared schema columns were found; pass on= or left_on=/right_on= explicitly.")
            key_candidates = [
                column
                for column in shared
                if str(column).lower() == "id" or str(column).lower().endswith("_id")
            ]
            if len(key_candidates) == 1:
                on = key_candidates[0]
            elif len(shared) == 1:
                on = shared[0]
            else:
                raise ValueError(
                    f"Multiple shared columns could be join keys: {shared}; pass on= explicitly."
                )
        if (left_on is None) != (right_on is None):
            raise ValueError("left_on and right_on must be provided together")
        return pd.merge(
            left_df,
            right_df,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            validate=validate,
        )

    def union_tables(
        self,
        tables: Sequence[str | pd.DataFrame],
        *,
        ignore_index: bool = True,
        strict_schema: bool = True,
        source_column: str | None = None,
    ) -> pd.DataFrame:
        if not tables:
            raise ValueError("union_tables requires at least one table")

        frames: list[pd.DataFrame] = []
        labels: list[str] = []
        for index, table in enumerate(tables):
            frames.append(self._as_dataframe(table).copy())
            labels.append(table if isinstance(table, str) else f"table_{index + 1}")

        canonical_columns = list(frames[0].columns)
        if strict_schema:
            canonical_set = set(canonical_columns)
            for index, (label, frame) in enumerate(zip(labels[1:], frames[1:]), start=1):
                if set(frame.columns) != canonical_set:
                    missing = sorted(canonical_set - set(frame.columns))
                    extra = sorted(set(frame.columns) - canonical_set)
                    raise ValueError(
                        f"Table {label!r} has an incompatible schema; missing={missing}, extra={extra}"
                    )
                frames[index] = frame[canonical_columns]

        if source_column:
            if any(source_column in frame.columns for frame in frames):
                raise ValueError(f"source_column {source_column!r} already exists")
            for label, frame in zip(labels, frames):
                frame[source_column] = label

        return pd.concat(frames, ignore_index=ignore_index, sort=False)

    def groupby_table(
        self,
        table: str | pd.DataFrame,
        *,
        by: str | Sequence[str],
        aggregations: Mapping[str, str | Sequence[str]],
        dropna: bool = False,
        sort: bool = True,
    ) -> pd.DataFrame:
        frame = self._as_dataframe(table)
        return (
            frame.groupby(by=by, dropna=dropna, sort=sort)
            .agg(dict(aggregations))
            .reset_index()
        )

    def groupby(self, table: str | pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return self.groupby_table(table, **kwargs)

    def _as_dataframe(self, table: str | pd.DataFrame) -> pd.DataFrame:
        if isinstance(table, pd.DataFrame):
            return table.copy()
        if table not in self.env.structures:
            raise KeyError(f"Unknown table: {table}")
        return self.env.operators.read_table_as_dataframe(table, has_headers=True)
