# TableAgent Retrieval

This package preserves the existing prepared-source API while providing the contract
for table-level retrieval.

## Existing pipeline compatibility

Existing code continues to use:

```python
from TableAgent.pipeline.retrieval import SourceRetriever
```

`SourceRetriever` still ranks prepared SiFlex workbook/sheet artifacts using lexical
scores, optional embeddings, and optional LLM reranking.

## Table retrieval extension point

Implement `TableRetrieverContract` or subclass the placeholder `TableRetriever`:

```python
from TableAgent.pipeline.retrieval import (
    TableCandidate,
    TableRetriever,
    TableSearchRequest,
)


class HybridTableRetriever(TableRetriever):
    def search(self, request: TableSearchRequest) -> list[TableCandidate]:
        # Add lexical retrieval, embeddings, reranking, and an optional LLM judge here.
        return []
```

Inject the implementation into `TableAgentPipeline`, `TableQARunner`, or
`QAEnvironment`. The QA-facing
`operators.find_tables(...)` method delegates to it, removes unknown/duplicate table
IDs, and uses its built-in lexical fallback when no backend is installed or the
placeholder raises `NotImplementedError`.

Shared reusable pieces are split by responsibility:

```text
cards.py             Retrieval-card construction
contracts.py         TableRetrieverContract and placeholder implementation
embeddings.py        Mock and OpenAI-compatible embedding clients
models.py            SourceCandidate, TableCandidate, TableSearchRequest
reranking.py         Structured selected-index handling
scoring.py           Score normalization, cosine similarity, hybrid scoring
source_retriever.py  Existing prepared workbook/sheet behavior
```

## Implementing table-level retrieval

The new implementation should move the retrieval unit from a prepared sheet to an
individual table while preserving the sheet-level retriever during migration.

### 1. Preserve the current source retriever

Do not remove or change the public behavior of `SourceRetriever.select()`. It is still
used by the prepared SiFlex path to choose one workbook and sheet. Existing imports
from `TableAgent.pipeline.retrieval` must continue to work.

Table-level retrieval can initially run after source preparation and before QA. Once it
is stable, the pipeline may use it directly instead of selecting only one sheet.

### 2. Build one candidate per table

New prepared source artifacts are organized by workbook and sheet. The retriever
also discovers the previous flattened directories for cache compatibility:

```text
sources/<workbook-id>/
  <sheet>/
    metadata.json
    structure.yaml
    sheet_text.txt
    table.png
    table.html
```

Read each `structure.yaml` and create one `TableCandidate` for every table entry. Do
not create a candidate for reserved blocks such as `relations`.

Each candidate should include:

```python
TableCandidate(
    table_id="table2",
    workbook_path=workbook_path,
    sheet_name="Data_Synthesis",
    table_name="Employee and Salary Information",
    description="Employee salary and KPI calculations.",
    structure_path=structure_path,
    retrieval_card=retrieval_card,
)
```

The retrieval card should contain compact semantic information rather than the full
table:

- Table ID, name, and description
- Workbook and sheet names
- Header IDs, labels, and descriptions
- Table range when available
- Formula relation IDs and descriptions
- A small sample of distinctive values, bounded by configuration

Keep the candidate linked to its workbook and sheet so the pipeline can still load the
correct source after table selection.

### 3. Respect the search request filters

Implement:

```python
class HybridTableRetriever(TableRetriever):
    def search(self, request: TableSearchRequest) -> list[TableCandidate]:
        ...
```

The implementation must honor:

- `allowed_table_ids`
- `workbook_paths`
- `sheet_names`
- `required_headers`
- `top_k`
- `rerank`

These filters prevent retrieval from escaping the workbooks and tables authorized for
the current sample.

### 4. Use staged hybrid ranking

Recommended ranking flow:

```text
Question or subtask
    ↓
Lexical table ranking
    ↓
Embedding similarity
    ↓
Weighted hybrid score
    ↓
Top-K candidates
    ↓
Optional LLM reranker/judge
    ↓
Final TableCandidate list
```

Reuse helpers from `embeddings.py` and `scoring.py` instead of duplicating the source
retriever's implementation. Cache table-card embeddings using a stable fingerprint of
the retrieval card and embedding model.

The lexical and embedding stages should only retrieve candidates. They must not decide
how to calculate the answer.

### 5. Add an optional LLM table judge

Give the LLM only the top candidates, not every table in every workbook. Require
structured output such as:

```json
{
  "selected_table_ids": ["table2"],
  "operation": "formula_evaluation",
  "join_keys": [],
  "relation_ids": ["rel_salary_calc"],
  "rationale": "Contains employee name, base salary, KPI factor, and salary formula."
}
```

For cross-table questions the judge may select multiple table IDs and identify a join,
union, or grouped aggregation. Validate every returned table ID and key against the
candidate metadata before accepting it.

The LLM selects context and operation intent only. Deterministic operators such as
`join_tables()` and `evaluate_formula()` must perform the calculation.

### 6. Connect it to QA through the existing contract

Inject the implementation when constructing the pipeline:

```python
retriever = HybridTableRetriever(...)
pipeline = TableAgentPipeline(
    llm_client=llm,
    layout_vlm_client=layout_vlm,
    table_retriever=retriever,
)
```

QA code continues to call:

```python
table_ids = operators.find_tables(question_or_subtask, top_k=5)
candidates = operators.retrieve_tables(question_or_subtask, top_k=5)
```

`TableRoutingOperator` already validates returned IDs and falls back to lexical routing
when the backend is absent or raises `NotImplementedError`.

Route using the full question for initial selection and the combined question plus
subtask description for later task-specific selection. Different subtasks may
legitimately select different tables.

### 7. Handle cross-table expansion

After selecting a table, optionally add related candidates when metadata proves that
another table is required:

- Shared join-key headers
- Compatible schemas for union
- Formula relations referencing another table or sheet
- Planner request for comparison or aggregation across periods

Expansion must remain inside the request's allowed workbook, sheet, and table filters.

### 8. Migration sequence

Implement in this order to avoid breaking the active pipeline:

1. Parse prepared sheet structures into table candidates.
2. Implement filtered lexical search and tests.
3. Add embedding scoring and caching.
4. Add optional reranking or LLM judging.
5. Inject the retriever into `TableAgentPipeline`.
6. Let QA use `find_tables()` for initial and subtask routing.
7. Only after verification, consider replacing the initial sheet-only selection.

Until step 7, `SourceRetriever` remains the first-stage sheet selector and
`TableRetriever` refines the selected sheet into tables.

### 9. Required tests

Add tests covering:

- Existing `SourceRetriever` imports and behavior remain unchanged.
- A sheet containing multiple unrelated tables produces separate candidates.
- Workbook, sheet, and allowed-table filters are enforced.
- Lexical-only fallback works without an embedding service.
- Hybrid scoring changes ranking when embeddings are enabled.
- Invalid LLM-selected table IDs and join keys are rejected.
- Multiple tables can be returned for join and union questions.
- `operators.find_tables()` delegates to the injected implementation.
- Missing or placeholder implementations fall back without breaking QA.
