from __future__ import annotations

UNDERSTANDING_SYSTEM_PROMPT = """You are an expert Excel data analyst.
Decide once what a spreadsheet question asks in terms of the workbook, so that a planner
can follow a single interpretation. State decisions only. Do not answer the question."""

# General answer-form rules keyed to the shape of the question.
ANSWER_FORM_RULES = """- When the question asks which or who, or asks for a category or item, the answer is that item's name, not a yes/no or true/false value; when the workbook label omits a qualifier the question states for that item (such as its group), keep the question's wording.
- When the question offers alternatives (X or Y), the answer names the chosen alternative, even when it is phrased as a yes/no question.
- Only a question about whether a single statement holds is answered starting with Yes or No.
- When the question asks for the difference between two values without saying which is subtracted from which, the answer is its magnitude, a non-negative number; keep a sign only when the question asks how much one increased, decreased, or changed relative to the other.
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
- `[indent n]` before a label is its indentation level: a label is a component of the nearest label above it with a smaller level, and a label with a larger level below it is one of its components.
- When a label you need is not in the content above, map it as `"question phrase" -> "label" (not shown; search the sheet for it)` instead of guessing a coordinate.
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

UNDERSTANDING_REVISION_TEMPLATE = """{understanding_prompt}

**Previous Decision Sheet:**
{decisions}

**Rejected Answer:** {answer}

**Reviewer Feedback:** {feedback}

The reviewer found that the previous decisions read the question wrongly. Write a corrected decision sheet with the
same headings and rules: change the decisions the feedback shows to be wrong, keep the others, and do not repeat
the reading that led to the rejected answer.
"""

# Blocks that carry the decision sheet to later stages; each is empty when no decisions exist.
DECISIONS_FOR_CODE = """Question decisions (follow them exactly; do only this subtask's part and do not compute extra totals, sums, or verification values the decisions do not ask for):
{decisions}
The decisions' cell coordinates come from a preview of the sheet. When the label printed at a coordinate differs from the label the decisions name, or a label is marked "search the sheet", locate the label with `operators.find_cells(label, sheet=...)` and use where it is found.
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
