from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from TableAgent.QA.operators.base_operator import BaseOperator
from TableAgent.pipeline.retrieval import TableCandidate
from TableAgent.QA.operators.formula_relation_operator import FormulaRelationOperator
from TableAgent.QA.operators.relational_table_operator import RelationalTableOperator
from TableAgent.QA.operators.table_routing_operator import TableRoutingOperator


class MultiTableOperator(BaseOperator):
    """Facade for routing, relational, and formula-aware multi-table operations."""

    name = "multitab"
    description = "Route work across tables and execute deterministic cross-table calculations."
    examples: tuple[str, ...] = ()

    def __init__(self, env: Any):
        super().__init__(env)
        self.routing = TableRoutingOperator(env)
        self.relational = RelationalTableOperator(env)
        self.formula = FormulaRelationOperator(env)

    def describe(self) -> str:
        sections = [f"{self.name}: {self.description}"]
        sections.extend(
            operator.describe()
            for operator in (self.routing, self.relational, self.formula)
        )
        return "\n\n".join(sections)

    def find_tables(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[str]:
        return self.routing.find_tables(query, top_k=top_k, min_score=min_score)

    def find_table(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[str]:
        return self.routing.find_table(query, top_k=top_k, min_score=min_score)

    def retrieve_tables(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[TableCandidate]:
        return self.routing.retrieve_tables(query, top_k=top_k, min_score=min_score)

    def join_tables(self, left: str | pd.DataFrame, right: str | pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return self.relational.join_tables(left, right, **kwargs)

    def union_tables(self, tables: Sequence[str | pd.DataFrame], **kwargs: Any) -> pd.DataFrame:
        return self.relational.union_tables(tables, **kwargs)

    def groupby_table(
        self,
        table: str | pd.DataFrame,
        *,
        by: str | Sequence[str],
        aggregations: Mapping[str, str | Sequence[str]],
        dropna: bool = False,
        sort: bool = True,
    ) -> pd.DataFrame:
        return self.relational.groupby_table(
            table,
            by=by,
            aggregations=aggregations,
            dropna=dropna,
            sort=sort,
        )

    def groupby(self, table: str | pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return self.relational.groupby(table, **kwargs)

    def list_relations(self, table_id: str | None = None, *, category: str | None = None) -> list[dict[str, Any]]:
        return self.formula.list_relations(table_id, category=category)

    def find_relation(self, query: str, *, table_id: str | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        return self.formula.find_relation(query, table_id=table_id, top_k=top_k)

    def evaluate_formula(
        self,
        relation_id: str,
        *,
        target_cell: str | None = None,
        mutations: Mapping[str, Any] | None = None,
        table_id: str | None = None,
    ) -> dict[str, Any]:
        return self.formula.evaluate_formula(
            relation_id,
            target_cell=target_cell,
            mutations=mutations,
            table_id=table_id,
        )
