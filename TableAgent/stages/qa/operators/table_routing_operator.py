from __future__ import annotations

from pathlib import Path
from typing import Any

from TableAgent.domain.introspection import describe_attribute_miss

from TableAgent.stages.qa.operators.base_operator import BaseOperator
from TableAgent.stages.retrieval import TableCandidate, TableSearchRequest
from TableAgent.utils import _lexical_overlap_score


class TableRef(str):
    """A table id that also answers `.id`, `.table_id` and `.label`.

    `find_headers` and `find_groups` return objects carrying `.id`, so agents reach for
    `[t.id for t in operators.find_tables(...)]` out of habit. Returning a plain `str`
    made that the single most common runtime error in the QA stage. This stays a real
    `str` -- it compares, hashes, serialises and indexes exactly like the id it wraps --
    while also answering the attribute the rest of the operator surface taught.
    """

    __slots__ = ("label", "sheet", "score")

    #: What this type is for. `str` contributes forty-odd methods that are true but
    #: irrelevant here, and listing them would bury `id` and `label`.
    __agent_attributes__ = ("id", "table_id", "label", "sheet", "score")

    def __new__(cls, table_id: str, *, label: str = "", sheet: str = "", score: float = 0.0):
        instance = super().__new__(cls, str(table_id))
        instance.label = label or str(table_id)
        instance.sheet = sheet
        instance.score = score
        return instance

    @property
    def id(self) -> str:
        return str(self)

    @property
    def table_id(self) -> str:
        return str(self)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(describe_attribute_miss(self, name))

    def __repr__(self) -> str:
        return f"TableRef({str(self)!r})"


class TableRoutingOperator(BaseOperator):
    """Route questions or subtasks to tables using verified structure metadata."""

    name = "multitab.routing"
    description = "Rank relevant table ids from names, descriptions, and verified headers."
    examples = (
        "operators.find_tables(query, top_k=2) -> list[TableRef] (a str subclass; both `t` and `t.id` give the table id)",
        "operators.retrieve_tables(query, top_k=2) -> list[TableCandidate]",
    )

    def find_tables(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[TableRef]:
        return [
            TableRef(
                candidate.table_id,
                label=candidate.table_name,
                sheet=candidate.sheet_name,
                score=candidate.score,
            )
            for candidate in self.retrieve_tables(query, top_k=top_k, min_score=min_score)
        ]

    def find_table(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[str]:
        """Compatibility alias for the original list-returning method."""
        return self.find_tables(query, top_k=top_k, min_score=min_score)

    def retrieve_tables(
        self,
        query: str,
        *,
        top_k: int = 1,
        min_score: float = 0.0,
    ) -> list[TableCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_text = str(query).strip()
        if not query_text:
            return []

        retriever = getattr(self.env, "table_retriever", None)
        if retriever is not None:
            request = TableSearchRequest(
                query=query_text,
                top_k=top_k,
                allowed_table_ids=tuple(self.env.structures),
                workbook_paths=(Path(self.env.workbook_path).resolve(),),
                sheet_names=tuple(
                    dict.fromkeys(
                        str(structure.get("sheet", ""))
                        for structure in self.env.structures.values()
                        if structure.get("sheet")
                    )
                ),
                metadata={"min_score": min_score},
            )
            try:
                candidates = retriever.search(request)
            except NotImplementedError:
                candidates = []
            valid_table_ids = set(self.env.structures)
            validated = []
            seen = set()
            for candidate in candidates:
                if (
                    candidate.table_id in valid_table_ids
                    and candidate.table_id not in seen
                    and candidate.score > min_score
                ):
                    validated.append(candidate)
                    seen.add(candidate.table_id)
                if len(validated) >= top_k:
                    break
            if validated:
                return validated

        return self._lexical_candidates(query_text, top_k=top_k, min_score=min_score)

    def _lexical_candidates(
        self,
        query_text: str,
        *,
        top_k: int,
        min_score: float,
    ) -> list[TableCandidate]:
        scored: list[tuple[float, str]] = []
        query_lower = query_text.lower()
        for table_id, structure in self.env.structures.items():
            name = str(structure.get("name", ""))
            description = str(structure.get("description", ""))
            score = 3.0 * _lexical_overlap_score(query_text, f"{table_id} {name}")
            score += 2.0 * _lexical_overlap_score(query_text, description)
            if name and name.lower() in query_lower:
                score += 20.0

            for header in self.env.operators.list_headers(table_id):
                header_text = f"{header.id} {header.label} {header.description}"
                score += 2.0 * _lexical_overlap_score(query_text, header_text)
                if header.label and header.label.lower() in query_lower:
                    score += 10.0

            if score > min_score:
                scored.append((score, table_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            TableCandidate(
                table_id=table_id,
                sheet_name=str(self.env.structures[table_id].get("sheet", "")),
                table_name=str(self.env.structures[table_id].get("name", "")),
                description=str(self.env.structures[table_id].get("description", "")),
                score=score,
                lexical_score=score,
                reason="Built-in lexical fallback",
            )
            for score, table_id in scored[:top_k]
        ]
