from __future__ import annotations
from typing import List, Optional
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
        "operators.find_headers(table_id, query) -> list[Header]",
        "operators.get_header(table_id, header_id) -> Header | None",
        "operators.resolve_header_columns(table_id, parent_header_id) -> list[str]",
        "operators.list_groups(table_id) -> list[StructureGroup]",
        "operators.find_groups(table_id, query) -> list[StructureGroup]",
        "operators.get_group(table_id, group_id) -> StructureGroup | None",
        "operators.intersect_group_with_header(table_id, group_id, header_id) -> CellRange | None",
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

    def get_header(self, table_id: str, header_id: str) -> Optional[Header]:
        """Get header info by explicit id."""
        headers = self.list_headers(table_id)
        for h in headers:
            if h.id == header_id:
                return h
        return None

    def resolve_header_columns(self, table_id: str, header_id: str) -> List[str]:
        """Resolve a header to the leaf column IDs represented in a table DataFrame."""
        header = self.get_header(table_id, header_id)
        if header is None:
            raise ValueError(f"Header {header_id!r} was not found in table {table_id!r}.")

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

    def find_groups(self, table_id: str, query: str) -> List[StructureGroup]:
        scored = []
        for group in self.list_groups(table_id):
            score = max(
                _lexical_overlap_score(query, group.id),
                _lexical_overlap_score(query, group.label),
                _lexical_overlap_score(query, group.description),
            )
            if query.lower() in group.label.lower() or query.lower() in group.description.lower():
                score += 10.0
            if score > 0:
                scored.append((score, group))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [group for _, group in scored]

    def get_group(self, table_id: str, group_id: str) -> Optional[StructureGroup]:
        return next((g for g in self.list_groups(table_id) if g.id == group_id), None)

    def intersect_group_with_header(
        self, table_id: str, group_id: str, header_id: str
    ):
        group = self.get_group(table_id, group_id)
        header = self.get_header(table_id, header_id)
        if group is None or header is None:
            return None
        group_range = group.data_range or group.group_range
        header_range = header.data_range or header.header_range
        return group_range.intersection(header_range) if group_range and header_range else None

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
