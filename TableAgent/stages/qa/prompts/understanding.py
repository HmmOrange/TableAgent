from __future__ import annotations

UNDERSTANDING_SYSTEM_PROMPT = """You are an expert Excel data analyst.
Decide once what a spreadsheet question asks in terms of the workbook, so that a planner
can follow a single interpretation. State decisions only. Do not answer the question."""

UNDERSTANDING_USER_PROMPT_TEMPLATE = """Clarify the question against the spreadsheet content and write a decision sheet.

**User Question:** {question}

**Excel Workbook Content:**
{workbook_content}

**Rules:**
- Commit to exactly one interpretation. Never list alternative readings, candidates, or options.
- State each decision directly without explaining why.
- Do not answer the question or compute any value from the data.
- Use the workbook's exact labels and cell coordinates as they appear in the content above.
- The content above may be a preview; do not conclude that data is absent because it is not shown.

**Decision Sheet** (use exactly these headings):

1. **Clarified Question**: the question as one precise sentence in the workbook's terminology.
2. **Term Mapping**: one line per entity, metric, category, and time expression in the question: `"question phrase" -> "exact workbook label" (cell coordinate)`; add the unit or scale when the workbook states it.
3. **Scope and Conditions**: the sheets, sections, rows, and columns to use; every filter, time period, grouping, and condition that must hold; and which hierarchy level counts (for example detail rows only, or a section's own total row).
4. **Decisions on Ambiguous Points**: one line per term that could be read more than one way: `term: chosen reading`. Write "None" when nothing is ambiguous.
5. **Required Operation**: the operation (lookup, filter, count, sum, average, difference, ratio, ranking, comparison, description), the order of its operands (for example "A minus B"), and the ordered steps, without computing them.
6. **Checks**: one or two concrete conditions the result must satisfy (for example the exact row label to use, or a total row that must not be added to its own children).
7. **Expected Answer Form**: the answer type, unit, precision, and format (single value, one item or every matching item, list order, and which fields to report).
"""
