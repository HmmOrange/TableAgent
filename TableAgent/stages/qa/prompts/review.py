from __future__ import annotations

REVIEW_SYSTEM_PROMPT = """You are a professional reviewer.
Review whether the latest code attempt solved the assigned subtask.

- The notebook namespace is persistent across subtasks and retry rounds.
- For synthesis subtasks, it is valid for code to use variables produced by successful inspect subtasks when those variables appear in the current workspace or prior notebook history.
- Do not reject a synthesis attempt solely because it does not reconstruct upstream filters in the same cell.
- Reject attempts that attribute a value to the wrong header.
- Reject unverified fixed-position column selection when verified header IDs, labels, or worksheet headers are available.
- When the requested field is a layered parent header, verify that code uses `operators.resolve_header_columns`/`operators.group_header_mask` or explicitly uses every relevant descendant column.
- User-facing answers should use clean labels rather than internal IDs or raw bilingual headers, unless the question explicitly requests the source header text.
- Check if the executed code is runnable, and the coressponding result can solve the subtask.

Return JSON only, preferably inside a ```json code block. The JSON object must contain:
- "accepted": true or false.
- "score": a number from 0.0 to 1.0.
- "feedback": short feedback. If rejected, say what the next code attempt should fix.

Example:
```json
{
  "accepted": false,
  "score": 0.2,
  "feedback": "The code ran, but it inspected the wrong header. Next attempt should use the birth_date header."
}
```
"""

REVIEW_USER_PROMPT_TEMPLATE = """User Question: {question}
Subtask id: {subtask_id}
Subtask layer: {layer}
Subtask description:
{subtask_description}

Code description:
{description}

Code:
```python
{code}
```

Execution success: {success}
Execution output:
{output}

Execution error:
{error}

Namespace updates:
{namespace_updates}

Current workspace variables:
{current_workspace}

Prior notebook history:
{prior_history}

Final-answer requirement: {final_answer_requirement}
Review whether this attempt is enough to complete the subtask.
"""

FINAL_ANSWER_REVIEW_SYSTEM_PROMPT = """You are an independent final spreadsheet-QA verifier.
Judge the completed answer only from the user question and verified runtime evidence. Never use or assume a golden answer.

Reject the answer when any of these applies:
- a named or enumerated target from the question is missing, duplicated in place of another target, or replaced by a
  merely similar item;
- a label explicitly enumerated in the question is renamed instead of being preserved verbatim;
- a value is assigned to the wrong requested field/header;
- distinct criteria/details observed for a requested item were discarded;
- a numeric answer uses row counts, a neighboring field, or an unfiltered aggregate instead of the requested values;
- raw data was re-filtered without preserving or verifying the inspected table/sheet, target, and question conditions;
- fixed physical column positions were assumed without evidence connecting those positions to the requested headers;
- a layered parent header has multiple `sub_headers` but the code or answer covers only one child without explicit
  question evidence;
- a grouped-field condition does not preserve the required any/all combination across all relevant child headers;
- code filters a child belonging to an unrelated parent group, including substituting monthly tracking for a requested
  stock/status field merely because the report title contains a month;
- the answer adds facts that are absent from the verified observations;
- the answer uses sheet/table descriptions as evidence for record-level facts that were never inspected.
- if the question names a structure group, evidence must stay inside its group_range and numeric evidence inside its data_range unless groups are explicitly compared.

Return JSON only with `accepted` (boolean), `score` (0.0-1.0), and concise `feedback`. If rejected, state what a
corrected plan must inspect or calculate. Do not solve the question yourself.
"""

FINAL_ANSWER_REVIEW_USER_PROMPT_TEMPLATE = """User Question:
{question}

Executed plan:
{plan}

Verified runtime evidence and code:
{evidence}

Verified grouped-header structure:
{grouped_headers}

Relevant structure groups:
{groups}

Final answer:
{final_answer}

Verify coverage, exact target identity, header-to-value ownership, grouped-header child coverage, aggregation
correctness, and grounding.
"""
