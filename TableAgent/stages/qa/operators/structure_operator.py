from __future__ import annotations
from typing import Any, List, Optional
from TableAgent.domain.structure import Header
from TableAgent.domain.group import StructureGroup
from TableAgent.stages.qa.operators.base_operator import BaseOperator
from TableAgent.utils import flatten_headers, _lexical_overlap_score

class StructureOperator(BaseOperator):
    """Operator for querying table structures and headers."""
    name = "structure"
    description = "Query table ids and header metadata from the loaded structure.yaml."
    examples = (
        "operators.list_tables() -> list[str]",
        "operators.list_headers(table_id) -> list[Header]",
        "header = operators.get_header(table_id, header_id); attributes are header.id, header.label, "
        "header.description, header.orientation, header.header_range, header.data_range, header.sub_headers",
        "operators.find_headers(table_id, query) -> list[Header]",
        "operators.get_header(table_id, header_id) -> Header (raises if absent; probe with operators.has_header)",
        "operators.resolve_header_columns(table_id, parent_header_id) -> list[str]",
        "operators.list_groups(table_id) -> list[StructureGroup]",
        "group = operators.get_group(table_id, group_id); attributes are group.id, group.label, "
        "group.description, group.axis, group.group_range, group.data_range",
        "operators.find_groups(table_id, query, limit=5) -> list[StructureGroup]",
        "operators.get_group(table_id, group_id) -> StructureGroup (raises if absent; probe with operators.has_group)",
        "operators.intersect_group_with_header(table_id, group_id, header_id) -> CellRange | None",
        "NOTE: the identifier attribute is `.id` on both Header and StructureGroup. "
        "`header.header_id` and `group.group_id` do not exist and raise AttributeError.",
        "NOTE: a StructureGroup scopes a block of records (a worksheet section); a Header names a field. "
        "Cross them with intersect_group_with_header instead of assuming row offsets.",
    )

    def list_tables(self) -> List[str]:
        """List available table ids in the loaded structure."""
        return list(self.env.structures.keys())

    def list_headers(self, table_id: str) -> List[Header]:
        """List all headers for a given table, flattened into a single list."""
        table = self.env.get_table_structure(table_id)
        if not table:
            return []
        return flatten_headers(table["headers"])

    def find_headers(self, table_id: str, query: str) -> List[Header]:
        """Find relevant headers by checking lexical overlap with query in id, label, or description."""
        headers = self.list_headers(table_id)
        scored = []
        for h in headers:
            score = max(
                _lexical_overlap_score(query, h.id),
                _lexical_overlap_score(query, h.label),
                _lexical_overlap_score(query, h.description),
            )
            q_lower = query.lower()
            if q_lower in h.id.lower() or q_lower in h.label.lower() or q_lower in h.description.lower():
                score += 10.0
            
            if score > 0:
                scored.append((score, h))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored]

    def _find_header(self, table_id: str, header_id: str) -> Optional[Header]:
        """Look a header up without raising; callers that tolerate absence use this."""
        for h in self.list_headers(table_id):
            if h.id == header_id:
                return h
        return None

    def get_header(self, table_id: str, header_id: str) -> Header:
        """Get header info by explicit id.

        Raises with the available ids rather than returning None: a silent None turns
        into an opaque `AttributeError` on the next line, which costs a whole retry
        round to diagnose. Use `has_header` or `find_headers` to probe for existence.
        """
        header = self._find_header(table_id, header_id)
        if header is not None:
            return header
        raise KeyError(
            f"Header {header_id!r} was not found in table {table_id!r}. "
            f"Available header ids: {self._id_hint(self.list_headers(table_id))}. "
            "Probe with operators.has_header(table_id, header_id), or resolve the field "
            "with operators.find_headers(table_id, query) -- that returns a possibly "
            "empty list, so check it before indexing."
        )

    def has_header(self, table_id: str, header_id: str) -> bool:
        """True when the header id exists; the non-raising probe for `get_header`."""
        return self._find_header(table_id, header_id) is not None

    @staticmethod
    def _id_hint(items: List[Any], limit: int = 25) -> str:
        ids = [str(getattr(item, "id", "")) for item in items if getattr(item, "id", "")]
        if not ids:
            return "(none)"
        shown = ", ".join(ids[:limit])
        return shown if len(ids) <= limit else f"{shown}, ... (+{len(ids) - limit} more)"

    def resolve_header_columns(self, table_id: str, header_id: str) -> List[str]:
        """Resolve a header to the leaf column IDs represented in a table DataFrame."""
        header = self.get_header(table_id, header_id)

        def leaf_ids(node: Header) -> List[str]:
            if not node.sub_headers:
                return [node.id]
            result: List[str] = []
            for child in node.sub_headers:
                result.extend(leaf_ids(child))
            return result

        return leaf_ids(header)

    def list_groups(self, table_id: str) -> List[StructureGroup]:
        table = self.env.get_table_structure(table_id)
        return list(table.get("groups", [])) if table else []

    def find_groups(
        self,
        table_id: str,
        query: str,
        *,
        limit: int = 5,
        min_overlap: int = 1,
    ) -> List[StructureGroup]:
        """Rank groups by lexical overlap with the query.

        Tokens shorter than three characters are ignored so that a stray article or a
        stray digit cannot make an unrelated section look like a confident match. An
        empty result means no group was close enough, which is a useful signal: fall
        back to headers rather than picking an arbitrary section.
        """
        import re
        import unicodedata

        def normalize(value: str) -> str:
            text = unicodedata.normalize("NFKC", str(value)).casefold()
            return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))

        def content_tokens(text: str) -> set[str]:
            return {token for token in text.split() if len(token) >= 3}

        normalized_query = normalize(query)
        query_tokens = content_tokens(normalized_query)
        scored = []
        for group in self.list_groups(table_id):
            fields = [normalize(group.id), normalize(group.label), normalize(group.description)]
            exact = any(field and f" {field} " in f" {normalized_query} " for field in fields[:2])
            overlap = max((len(query_tokens & content_tokens(field)) for field in fields), default=0)
            if exact or overlap >= max(1, min_overlap):
                scored.append((10 if exact else 0, overlap, group))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        ranked = [group for _, _, group in scored]
        return ranked[:limit] if limit and limit > 0 else ranked

    def _find_group(self, table_id: str, group_id: str) -> Optional[StructureGroup]:
        """Look a group up without raising; callers that tolerate absence use this."""
        return next((g for g in self.list_groups(table_id) if g.id == group_id), None)

    def get_group(self, table_id: str, group_id: str) -> StructureGroup:
        """Get a structure group by explicit id, raising with the available ids."""
        group = self._find_group(table_id, group_id)
        if group is not None:
            return group
        raise KeyError(
            f"Group {group_id!r} was not found in table {table_id!r}. "
            f"Available group ids: {self._id_hint(self.list_groups(table_id))}. "
            "Probe with operators.has_group(table_id, group_id), or find the section with "
            "operators.find_groups(table_id, query) -- that returns a possibly empty "
            "list, so check it before indexing."
        )

    def has_group(self, table_id: str, group_id: str) -> bool:
        """True when the group id exists; the non-raising probe for `get_group`."""
        return self._find_group(table_id, group_id) is not None

    def intersect_group_with_header(
        self, table_id: str, group_id: str, header_id: str
    ):
        """Cells where a group's records cross a header's field.

        A row-axis group contributes rows and the header contributes columns, so the two
        are crossed rather than overlapped. A plain overlap returns nothing whenever the
        group's recorded data_range spans fewer columns than the table -- common enough
        in extracted structures that it made this operator unusable on the tables it
        matters most for.
        """
        from TableAgent.domain.ranges import CellRange

        group = self._find_group(table_id, group_id)
        header = self._find_header(table_id, header_id)
        if group is None or header is None:
            return None
        group_range = group.data_range or group.group_range
        header_range = header.data_range or header.header_range
        if group_range is None or header_range is None:
            return None

        axis = str(getattr(group, "axis", "") or "").strip().lower()
        if axis == "row":
            row_start = max(group_range.start_row, header_range.start_row)
            row_end = min(group_range.end_row, header_range.end_row)
            col_start, col_end = header_range.start_col, header_range.end_col
        elif axis == "column":
            col_start = max(group_range.start_col, header_range.start_col)
            col_end = min(group_range.end_col, header_range.end_col)
            row_start, row_end = header_range.start_row, header_range.end_row
        else:
            return group_range.intersection(header_range)

        if row_start > row_end or col_start > col_end:
            return None
        return CellRange(row_start, col_start, row_end, col_end, header_range.sheet)

if __name__ == "__main__":
    import argparse
    from TableAgent.stages.qa.environment.qa_env import QAEnvironment

    parser = argparse.ArgumentParser(description="Smoke-test structure/header operators.")
    parser.add_argument("--structure", default="sample/structure.yaml")
    parser.add_argument("--workbook", default="sample/QA_sample.xlsx")
    parser.add_argument("--query", default="score")
    args = parser.parse_args()

    env = QAEnvironment(args.structure, args.workbook)
    op = StructureOperator(env)
    table_id = env.default_table_id()
    headers = op.list_headers(table_id)
    matches = op.find_headers(table_id, args.query)
    print(f"tables={op.list_tables()}")
    print(f"default_table={table_id}")
    print(f"headers={len(headers)}")
    print(f"matches={[h.id for h in matches[:5]]}")
