from __future__ import annotations
from typing import List, Optional
from TableAgent.schema.header import Header
from TableAgent.QA.operators.base_operator import BaseOperator
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

if __name__ == "__main__":
    import argparse
    from TableAgent.environment.qa_env import QAEnvironment

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
