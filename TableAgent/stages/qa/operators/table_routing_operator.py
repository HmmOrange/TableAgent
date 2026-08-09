from __future__ import annotations

from pathlib import Path

from TableAgent.QA.operators.base_operator import BaseOperator
from TableAgent.pipeline.retrieval import TableCandidate, TableSearchRequest
from TableAgent.utils import _lexical_overlap_score


class TableRoutingOperator(BaseOperator):
    """Route questions or subtasks to tables using verified structure metadata."""

    name = "multitab.routing"
    description = "Rank relevant table ids from names, descriptions, and verified headers."
    examples = (
        "operators.find_tables(query, top_k=2) -> list[str]",
        "operators.retrieve_tables(query, top_k=2) -> list[TableCandidate]",
    )

    def find_tables(self, query: str, *, top_k: int = 1, min_score: float = 0.0) -> list[str]:
        return [
            candidate.table_id
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
