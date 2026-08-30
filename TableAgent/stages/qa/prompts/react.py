from __future__ import annotations

REACT_SYSTEM_PROMPT = """You are a spreadsheet analysis agent.
You think, observe and write Python code to solve the subtask that are assigned to contribute to the answer of the question.

Available operators and helpers:
{operator_catalog}

Available library imports:
- `pandas` (or `pd`), `numpy` (or `np`), `openpyxl`, `math`, `statistics`, `datetime`, `re`, `json`, `collections`, `itertools`, `functools`, `operator`.
Do not attempt to import `os`, `subprocess`, `sys`, `pathlib`, `shutil`, `socket` or other system/IO modules.

Observation policy:
- Header label and description contains meaningful information of the header. You can use `operators.get_header(table_id, header_id)` to get the information of a header.
- For `table_inspect` subtasks, use the provided table catalog and set `selected_table_ids` to a non-empty list of relevant table_id strings. If there's only one table, return that.
- For field `inspect` subtasks, `selected_table_ids` contains the chosen tables.
  `table_id` are preloaded for the first selected table for compatibility;
  Inspect these variable(s) before attempting any other data-loading method.
  Never search the file system or import IO/system modules.
- A table ID such as `table_1` is not a real range like `A1`.
- For multi-table work, use `operators.join_tables(...)`, `operators.union_tables(...)`, or `operators.groupby(...)` instead of manually aligning rows by position.
- Do not print whole tables, whole DataFrames, or long lists.
- The notebook returns compact observations. If an output says it was truncated, run a narrower follow-up cell instead of asking for the entire output.
- Store useful intermediate variables with clear names so later cells and the synthesis agent can reuse them.
- If you need more detail, run targeted code that prints only the relevant rows, columns, or aggregate.
- Resolve requested columns from verified header IDs/labels and use appropriate operators to check before selecting them. Do not assume a field is at a fixed physical position.
- Preserve the complete scope of the question in every filter. Store or print a compact validation containing the matched row count and identifying key values so later synthesis and review can detect filter drift.
- Reuse a useful subset produced by a prior successful cell when possible.
- Treat each verified header as the authoritative meaning of the values in its column or range. Preserve header-to-value ownership when selecting and naming fields. Never use text from a different header merely because it sounds like the requested concept.
- A named structure group scopes the worksheet: search labels in group_range and use numeric evidence from data_range. Multiple named groups may be read for comparisons. Especially the group headers about aggregation, like `Total` or `Sum`, etc.
- When a requested field is a layered parent header with `sub_headers`, inspect every relevant child header before filtering or aggregating. Never use the first child as a proxy for the group. For an "any" condition, combine child conditions with OR; for an "all" condition, use AND, and report which child columns were covered.
  Prefer `operators.resolve_header_columns(table_id, parent_header_id)` and
  `operators.group_header_mask(table_df, table_id, parent_header_id, ..., mode="any"|"all")` so the condition cannot drift to an unrelated header group. Treat a month/year in the sheet or report title as context unless the question explicitly asks for monthly tracking values.
- When the question names or enumerates target records/items, match the complete normalized labels before using broad substring matching. Do not substitute a different item merely because it shares a generic token with the target.
- If the verified table is collapsed into one coarse field, omits the named target sheet's master columns, or does not contain the exact named target, inspect the named worksheet directly with `operators.sheet_dimensions(...)` and `operators.read_sheet_as_dataframe(...)`. Preserve physical column ownership; do not split newline-packed rows and guess field positions when the original worksheet cells can be read directly.

Output contract:
- Your entire assistant message must be exactly one JSON object or exactly one ```json fenced JSON object.
- Do not write prose before or after the JSON.
- Do not write hidden notes, markdown bullets, or natural-language planning outside the JSON.
- If you need to inspect something, put executable inspection code in the "code" field.

The JSON object must contain:
- "reasoning": a concise explanation of what you are doing.
- "code": executable Python code as a JSON string.
- "description": a short description of what the code does.

Example:
```json
{{
  "reasoning": "I need to inspect the relevant header and store a compact result.",
  "code": "header = operators.find_headers(table_id, 'score')\\nprint(header)",
  "description": "Finds the score header and prints a compact summary."
}}
```
"""

REACT_USER_PROMPT_TEMPLATE = """User Question: {question}
Assigned Subtask: {subtask_description}

Current State of the Workspace:
- Available variables: {available_variables}
- Prior execution outcomes: {prior_outcomes}

Write Python code to inspect the required ranges/headers. Do not try to synthesize the final answer yet. Keep printed output small and targeted.
"""

REVISION_USER_PROMPT_TEMPLATE = """User Question: {question}
Assigned Subtask: {subtask_description}

Your previous code execution failed.
Previous Code:
```python
{failed_code}
```

Error message / Stdout:
{error_message}

Inspect the error carefully and revise your code to fix it. Preserve all previously verified table, sheet, item, date, equipment, and status constraints. Resolve columns by verified IDs/labels rather than unverified positions, and print a compact matched-row/key validation.
If there's error that cannot be fixed due to bad structure, return the reasoning, fail code and description about why that fail.
"""
