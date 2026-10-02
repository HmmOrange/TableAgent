from __future__ import annotations

UNDERSTANDING_SYSTEM_PROMPT = """You are an expert Excel data analyst.
Decide once what a spreadsheet question asks in terms of the workbook, so that a planner
can follow a single interpretation. State decisions only. Do not answer the question."""

# General answer-form rules keyed to the shape of the question.
ANSWER_FORM_RULES = """- When the question asks which or who, or asks for a category or item, the answer is that item's name, not a yes/no or true/false value; when the workbook label omits a qualifier the question states for that item (such as its group), keep the question's wording.
- When the question offers alternatives (X or Y), the answer names the chosen alternative, even when it is phrased as a yes/no question.
- Only a question about whether a single statement holds is answered starting with Yes or No.
"""

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
""" + ANSWER_FORM_RULES + """
**Decision Sheet** (use exactly these headings):

1. **Clarified Question**: the question as one precise sentence in the workbook's terminology.
2. **Term Mapping**: one line per entity, metric, category, and time expression in the question: `"question phrase" -> "exact workbook label" (cell coordinate)`; add the unit or scale when the workbook states it.
3. **Scope and Conditions**: the sheets, sections, rows, and columns to use; every filter, time period, grouping, and condition that must hold; and which hierarchy level counts (for example detail rows only, or a section's own total row).
4. **Decisions on Ambiguous Points**: one line per term that could be read more than one way: `term: chosen reading`. Write "None" when nothing is ambiguous.
5. **Required Operation**: the operation (lookup, filter, count, sum, average, difference, ratio, ranking, comparison, description), the order of its operands (for example "A minus B"), and the ordered steps, without computing them.
6. **Checks**: one or two concrete conditions the result must satisfy (for example the exact row label to use, or a total row that must not be added to its own children).
7. **Expected Answer Form**: the answer type, unit, precision, and format (single value, one item or every matching item, list order, and which fields to report).
"""

# Blocks that carry the decision sheet to later stages; each is empty when no decisions exist.
DECISIONS_FOR_CODE = """Question decisions (follow them exactly; do only this subtask's part and do not compute extra totals, sums, or verification values the decisions do not ask for):
{decisions}
"""

DECISIONS_FOR_SYNTHESIS = """Question decisions (compute `final_answer` with exactly the Required Operation and report only what the Expected Answer Form lists):
{decisions}
"""

DECISIONS_FOR_REVIEW = """Question decisions (reject an attempt that departs from them):
{decisions}
"""

DECISIONS_FOR_FINAL_REVIEW = """Question decisions made before planning:
{decisions}
Check both that the final answer follows these decisions and that the decisions themselves match the question; reject when either fails and say which one.
"""

PLANNER_DECISIONS_RULE = "Write each subtask description as the action to take; do not put data values or reasoning in it."
