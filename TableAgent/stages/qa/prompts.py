"""Every prompt of the QA stage."""

from __future__ import annotations

ANSWER_FORM_RULES = """- When the question asks which or who, or asks for a category or item, the answer is that item's name, not a yes/no or true/false value; when the workbook label omits a qualifier the question states for that item (such as its group), keep the question's wording.
- When the question offers alternatives (X or Y), the answer names the chosen alternative, even when it is phrased as a yes/no question.
- Only a question about whether a single statement holds is answered starting with Yes or No.
- When the question asks for the difference between two values without saying which is subtracted from which, the answer is its magnitude, a non-negative number; keep a sign only when the question asks how much one increased, decreased, or changed relative to the other."""

# ----------------------------------------------------------------------------- reasoning

REASONER_SYSTEM_PROMPT = """You reason about a question over a spreadsheet. You decide which needs the answer requires (a need is one fact to read from the workbook) and judge the evidence; Inspectors read the workbook for you, and Checkers may test what they report.
You never read cells or compute the answer yourself. Reply with exactly one JSON object and no other text."""

REASON_START_PROMPT = """Question: {question}

Workbook preview (A1-annotated rows; it may be partial):
{preview}

structure.yaml (a rough map; it may be incomplete):
{structure}

Reason about the question before anything is read:
1. What does it ask for, and in what form (one value, an item, a list, a comparison, a yes/no)? List the answer slots: only the parts the question asks to report, each with the exact phrase of the question that asks for it. A value needed only to compute or compare (the other side of a comparison, the operands of a difference) is a need, not a slot.
2. Which needs must be read from the workbook? Describe each by its meaning: which entity, measure, period or condition, and unit. Never name cells; Inspectors locate them.
3. What could mislead? The same label in several sections, a measure split into sub-columns, a total row beside detail rows, carried-down periods, differing units.
4. State a working hypothesis: how the needs combine into the answer. State no values: the preview may be partial, and values come from inspections.

Needs read from one row, column or block (a series over periods, several measures of one item, all items to rank or compare) form ONE need whose value is a list or a dict; split needs only when they are located independently.

Answer form rules:
{answer_form_rules}

Return JSON, slots in the order the answer reports them:
{{"analysis": "<your reasoning>", "hypothesis": "<how the needs combine>", "answer_form": "<form of the answer>", "answer_slots": [{{"slot": "<what to report>", "asked_by": "<exact phrase of the question>"}}], "needs": [{{"id": "n1", "description": "<a need to read>"}}]}}"""

REASON_STEP_PROMPT = """{open_needs}Question: {question}

Hypothesis: {hypothesis}
Answer form: {answer_form}
Answer slots:
{slots}

Needs and evidence (each attempt shows the cells read, where they are, and the checks of it):
{needs}

Earlier steps:
{history}

Budget: step {step} of {max_steps}; model calls used {calls} of {max_calls}.
{feedback}
Reason over the evidence, then choose one action:
1. What does each found need establish? The value is right only when where it was read matches every qualifier of the need (entity, group, section, measure, period, unit); a label that occurs in several places is often read in the wrong section, and a set of rows often misses rows or includes totals.
2. Judge every check marked RULE ON IT: weigh the check's cells and reasons against the evidence. "keep" the evidence when the check is wrong or does not apply to the need; "reject" it when the check shows it is wrong (the need reopens, to be inspected again). A check that confirms or is unsure needs no ruling.
3. Does anything contradict the hypothesis or the question?
4. What is still missing for each answer slot?
When an inspection failed because its code failed, use what it printed to give the next inspection concrete rows, columns or labels.
{rulings}
Actions (arguments in the same JSON object):
- inspect: "need_id", "strategy" ({strategies}), "instruction" (what to look for, what to avoid, where; after a rejected check, what the check found)
- add_need: "need": {{"id": "<new id>", "description": "<need>"}}
- revise_need: "need_id", "description" (reopens the need with a new description)
- drop_need: "need_id" (only when the question does not need it; a need that is hard to find is revised, not dropped)
- revise_hypothesis: "hypothesis", optionally "answer_form" and "answer_slots" (allowed once)
- synthesize: when every need is found or dropped, every check to rule on is ruled, and the evidence fills every answer slot

Return JSON:
{{"assessment": "<your reasoning over the evidence>", "rulings": [...], "action": "<one action>", ...its arguments..., "reason": "<why this action>"}}"""

RULINGS_TEXT = """
Checks to rule on: {needs}. Give one ruling per need in "rulings": [{{"need_id": "<need>", "ruling": "keep" or "reject", "why": "<what decides it>"}}].
"""

REASON_END_PROMPT = """Question: {question}

Hypothesis: {hypothesis}
Answer form: {answer_form}
Answer slots with the part of the answer filled in for each:
{slots}

Needs and evidence:
{needs}

Proposed answer: {answer}
{method_check}
Review the answer against the question:
- Does it answer what was asked, in the requested form, with every slot filled and nothing the question did not ask for?
- Is each part supported by found evidence read where every qualifier of its need holds?
Return the answer only when a slot is missing, a part was not asked for, a part is not supported by the evidence, the evidence was read in the wrong place, the computation does not follow from the question, or the answer does not respond to the question. Wording, units shown and number formatting are not reasons.

Return JSON:
{{"assessment": "<your reasoning>", "verdict": "accept" or "return", "problem": "<what is wrong, empty when accepted>", "instruction": "<what to establish next, empty when accepted>"}}"""

METHOD_CHECK_TEXT = """
A Checker reviewed how the answer was computed and found it {verdict}: {problem}
Weigh this on the question and the evidence; it is a reason to return the answer only when they support it.
"""

# ---------------------------------------------------------------------------- inspection

STRATEGY_TEXTS = {
    "search_sheet": (
        "Find each label of the need with `sheet.find(label)`, then look at the candidates with `sheet.describe(ref)` "
        "and the rows or columns around them, and pick the cell whose context matches the need."
    ),
    "structure": (
        "Start from structure.yaml: the data range of the header and group that match the need locate the column "
        "and the rows; confirm with `sheet.read` and `sheet.describe` on the raw cells."
    ),
    "window": (
        "Read the window named in the instruction with `sheet.read`, including the header rows above it and the "
        "label columns to its left, and read the value from it."
    ),
}

SHEETS_TOOL = "- `sheet.sheets()` -> sheet names with their sizes"
FIND_TOOL = "- `sheet.find(text, sheet=\"\")` -> [(ref, value), ...] cells containing the text as whole words (case and spacing ignored)"
READ_TOOL = "- `sheet.read(\"B3:F9\", sheet=\"\")` -> rows of values; a ref may name its sheet, e.g. \"'Data'!B3\""
AROUND_TOOL = "- `sheet.around(\"D8\", rows=2, cols=2, sheet=\"\")` -> the filled cells around one cell, a line per row with their refs (the cell marked *)"
FRAME_TOOL = "- `sheet.frame(sheet=\"\")` -> the used range as a DataFrame (index = Excel row numbers, columns = column letters)"
HEADERS_TOOL = """- `sheet.headers("D8", sheet="")` -> what the cell means, found by walking up and left from it: "row_labels", "keys" (labels written once above and carried down), "column_headers" with "headers_from" (structure.yaml or the layout), "sections" (label-only rows above, its own first, then its parents, each with its row span and the structure.yaml group naming it), "siblings" (the other sections at its level), "block" (the run of rows shaped like its row, and why it starts and ends: a blank or label row, a total by formula, a differently formatted row), "structure" (structure.yaml entries holding it), and "uncertain" (why the walk may be wrong; then "vision" holds a vision model's reading of an image of the area)
- `sheet.describe("D8", sheet="")` -> the same in one line, without the vision model"""
STRUCTURE_TOOL = """- `sheet.structure_at(row, col, sheet="")` -> structure.yaml headers and groups whose data range holds the cell
- `sheet.structure_find(text, sheet="")` -> structure.yaml headers and groups whose label path holds the words of `text`, then those whose description does: [{"kind", "path", "sheet", "label_range", "data_range", "description", "matched"}, ...]; finds an entry the structure map left out or names differently (a range may be off by a row: confirm it on the cells)"""
LOOK_TOOL = "- `sheet.look(\"B3:H20\", \"<what to look for>\", sheet=\"\")` -> a vision model's answer about an image of the range (at most 60 rows x 30 columns; one model call each). Use it for what the values do not show: tick marks or crosses drawn as symbols, shading, how merged headers span the columns below; read values with `sheet.read`."

SHEET_TOOLS = "The `sheet` object reads the workbook (merged cells read as their visible value):\n" + "\n".join(
    (SHEETS_TOOL, FIND_TOOL, READ_TOOL, HEADERS_TOOL, AROUND_TOOL, FRAME_TOOL, STRUCTURE_TOOL)
)

NOTEBOOK_IMPORTS = """Imports: pandas, numpy, openpyxl, math, statistics, datetime, re, json, collections, itertools, functools, operator. System and file modules are not available. Variables persist between turns."""

INSPECTOR_SYSTEM_PROMPT = f"""You are an Inspector: you establish ONE need from a workbook by writing Python in a persistent notebook. An inspection takes several turns; each turn you write one code cell and then see what it printed. Look first, then extract.

{SHEET_TOOLS}
{NOTEBOOK_IMPORTS}

How to work:
- First turn: look, do not extract. Start from structure.yaml when it names the need; otherwise find the labels of the need. Print the candidates with `sheet.describe`.
- Extract only once you have seen where the value is. A label that occurs in several places belongs to different sections: choose by the context, not by the first match.
- For a need over a set of rows (all items to rank, count, sum or take the maximum of), check with `sheet.headers` which block and sections the rows belong to and which sibling sections exist, and keep only the rows the need means; leave out total rows.
- Keep output small: print only relevant cells, never whole tables.
- When the value is found, or cannot be found, set the evidence variable in code and set "done" to true in the same reply.

Reply with one JSON object, then this turn's Python in one ```python block after it, and no other text:
{{"thought": "<what you have seen and what you do next>", "done": false}}
```python
<Python for this turn>
```"""

INSPECTOR_SYSTEM_PROMPT_WITH_LOOK = INSPECTOR_SYSTEM_PROMPT.replace(SHEET_TOOLS, f"{SHEET_TOOLS}\n{LOOK_TOOL}")

INSPECTOR_PROMPT = """Question: {question}
Working hypothesis: {hypothesis}
Need to establish ({need_id}): {need}
Strategy ({strategy}): {strategy_text}
Instruction: {instruction}
{earlier}
Sheets: {sheets}

structure.yaml (a rough map; confirm against the cells):
{structure}

Evidence contract: set `{variable}` to a dict {{"value": ..., "cells": ["D8", ...], "where": "...", "note": "..."}}
- "value": the need exactly as described, read from the cells; when the need is several values, all of them as a list or dict, not a result computed from them (the answer is composed later).
- "cells": the cells the value was read from; for a list or dict value, the cell of every element in order (a range like "B3:B9" is allowed); prefix the sheet ("'Data'!D8") when the workbook has several sheets.
- "where": the labels that place those cells, as written in the sheet: section or group, row label, column header (e.g. "South > Alpha | 2019").
- "note": what the reasoning should know: other places you considered and why you did not choose them, units, assumptions.
- When the need cannot be found, set "value" to None, "cells" to [] and say why in "note".

{transcript}Turn {turn} of {max_turns}.{last}"""

# ---------------------------------------------------------------------------------- checks

REVERSE_SYSTEM_PROMPT = """You are a Checker. An Inspector read one need of a spreadsheet question by going from the need's meaning to cells; you check it the other way round, from the cells to their meaning. The runner has walked from each cell up to its column headers and left to its row labels, and lists what that walk found.
Reply with exactly one JSON object and no other text."""

REVERSE_PROMPT = """Question: {question}
Need ({need_id}): {need}

The Inspector's evidence:
- value: {value}
- cells: {cells}
- where (as the Inspector reported it): {where}
- note: {note}

What the runner found from the cells:
{facts}

Check:
1. Do the cells mean what the need describes? Compare their row labels, column headers, sections, structure.yaml entries and descriptions with every qualifier of the need (entity, group, section, measure, period, unit).
2. Do the cells hold the reported values?
3. For a set of rows: are rows that the need means left out, or are rows it does not mean (totals, other sections) included?
Refute only for a concrete reason the facts show, naming the cells; otherwise confirm.

Return JSON:
{{"verdict": "confirmed" or "refuted", "reason": "<what the facts show>", "cells": ["<cells that show it>"]}}"""

CHECK_WORK_RULES = """Reply with one JSON object, then this turn's Python in one ```python block after it, and no other text:
{"thought": "<what you have seen and what you do next>", "done": false}
```python
<Python for this turn>
```"""

INDEPENDENT_SYSTEM_PROMPT = """You are a Checker: an Inspector reported the value of one need of a spreadsheet question, and you test it by re-deriving the need yourself in Python, in your own notebook, with other tools than the Inspector's. Each turn you write one code cell and then see what it printed.

{tools}
{claim}{imports}

How to work:
- Locate the need yourself from its meaning{own_cells}.
- For a single value: check that its row, column and section match every qualifier of the need, and that no other cell fits better.
- For several values or a set of rows (a list, a ranking, a count, a maximum, an average): enumerate every row that qualifies yourself: leave out totals, subtotals and other sections.
{finish}

{work_rules}"""

INDEPENDENT_CLAIM = "`claim` holds the Inspector's evidence (value, cells, where, note).\n"
INDEPENDENT_SEES_FINISH = """- Refute only with concrete counter-evidence: the cells and values that show the claim is wrong. When your re-derivation agrees, confirm.
When you have enough, set "done" to true and add "verdict" ("confirmed" or "refuted"), "reason" and "cells" (the cells that show it); the code block may then be left out. The last turn is for the verdict only."""
INDEPENDENT_BLIND_FINISH = """- When you have the value, set `found = {"value": ..., "cells": [...]}` in code (the need as described: one value, or all values as a list or dict) and set "done" to true in the same reply, with "reason" saying where you read it. When the need cannot be found, set `found = {"value": None, "cells": []}`."""

FORMATS_TOOL = "- `sheet.formats(\"A3:D20\", sheet=\"\")` -> the filled cells of the range, a line each, with their format (bold, fill, indent) and formula: bold or filled label rows often mark sections, a formula like =SUM(...) marks a total"

VIEW_TOOLS = {
    "grid": "The `sheet` object reads raw cells only; structure.yaml is not available:\n" + "\n".join(
        (SHEETS_TOOL, FIND_TOOL, READ_TOOL, AROUND_TOOL, FRAME_TOOL, FORMATS_TOOL)
    ),
    "image": "The `sheet` object finds cells and looks at images of the sheet; cell values are not read directly:\n" + "\n".join(
        (SHEETS_TOOL, FIND_TOOL, LOOK_TOOL)
    ),
    "frame": "`tables` maps each sheet name to its used range as a pandas DataFrame (index = Excel row numbers, columns = column letters); there is no other access to the workbook.",
}

INDEPENDENT_PROMPT = """Question: {question}
Need ({need_id}): {need}
{preview}{claim}
{transcript}Turn {turn} of {max_turns}.{last}"""

DEBATE_SYSTEM_PROMPT = f"""You are a Checker arguing against an Inspector: it reported the value of one need of a spreadsheet question, and your task is to find the strongest reasons that this evidence is wrong, by exploring the workbook in Python, in your own notebook. Each turn you write one code cell and then see what it printed.

{SHEET_TOOLS}
`claim` holds the Inspector's evidence (value, cells, where, note).
{NOTEBOOK_IMPORTS}

Look for:
- another cell that fits the need better: the same label in another section, a neighbouring column or period, a different unit;
- for a set of rows: rows the need means that are left out, totals or rows of other sections that are included;
- cells that do not hold the reported values.
Every objection must name the cells that show it, read by your code; an objection without cells is dropped. When the evidence holds, raise no objection.

{CHECK_WORK_RULES}
When you have enough, set "done" to true and add "objections": [{{"objection": "<what is wrong>", "cells": ["<cells that show it>"]}}] (at most 3, the strongest first; an empty list when the evidence holds); the code block may then be left out. The last turn is for the objections only."""

DEBATE_PROMPT = """Question: {question}
Need ({need_id}): {need}

The Inspector's evidence (also in `claim`):
{claim}

{transcript}Turn {turn} of {max_turns}.{last}"""

METHOD_SYSTEM_PROMPT = """You are a Checker of how the answer of a spreadsheet question was computed from evidence already read. You do not check the evidence itself; you check that the computation does what the question asks.
Reply with exactly one JSON object and no other text."""

METHOD_PROMPT = """Question: {question}
Answer form: {answer_form}
Answer slots, in order:
{slots}

Evidence variables:
{evidence}

The code that computed the answer parts:
```python
{code}
```
Answer: {answer}

Check:
1. Does each operation follow the wording of the question: list versus sum, count, average, ratio or difference; which items are compared; the direction of a comparison?
2. Does it use every evidence variable that a slot needs, and apply every filter or condition the question states?
3. Does each part have the form, unit and precision its slot asks for?
Report a flaw only for a concrete mismatch with the question.

Return JSON:
{{"verdict": "sound" or "flawed", "problem": "<the mismatch, empty when sound>"}}"""

# ------------------------------------------------------------------------------- vision

VISION_SYSTEM_PROMPT = "You look at an image of a spreadsheet range, rendered with its row numbers and column letters, and answer the question about it precisely, naming cells by their coordinates."

VISION_PROMPT = """The image shows the range {area} of sheet "{sheet}".
{question}"""

HEADERS_LOOK_QUESTION = (
    "Cell {cell} is the bottom-right cell of the image. Which column headers stand above it, which row label "
    "stands left of it in its row, and under which section, group or title rows does its row fall? Use formatting "
    "(bold, fill, indent, merged spans, blank rows) to tell levels apart. Name each label with its cell."
)

# ----------------------------------------------------------------------------- synthesis

SYNTHESIZER_SYSTEM_PROMPT = """You compose the answer of a spreadsheet question from evidence already read, by writing Python in a notebook where each evidence variable is a dict with "value", "cells", "where" and "note".
Imports: pandas, numpy, math, statistics, datetime, re, json, collections, itertools, functools, operator. Do not read the workbook again.
Your code must set `answer_parts`: a list with exactly one entry per answer slot, in slot order.
- An entry is the value for its slot as it should appear in the answer; keep its unit (days, a percent sign, a currency) when it has one.
- A slot that asks for several items gets a list of them; a yes/no slot gets True or False.
- Round a number to the precision the question states; otherwise round non-integers to two decimals.
- A difference whose direction the question does not state is reported as its magnitude.

Reply with one JSON object, then your Python in one ```python block after it, and no other text:
{"thought": "<how the evidence gives each slot>"}
```python
<Python that sets answer_parts>
```"""

SYNTHESIZER_PROMPT = """Question: {question}
Answer form: {answer_form}
Answer slots, in order:
{slots}

Evidence variables:
{evidence}

Answer form rules:
{answer_form_rules}
{transcript}"""

# ------------------------------------------------------------------------------ fallback

DIRECT_ANSWER_SYSTEM_PROMPT = (
    "You are a table question answering agent. Use the table content and verified structure YAML to answer "
    "concisely. Output only the final answer, without explanation, steps, or introductory or concluding remarks. "
    "Never attribute a value to a neighboring or similar header, and never report a value read in one section "
    "under another section's label."
)

DIRECT_ANSWER_PROMPT = """\
Question: {question}

Verified structure.yaml:
{structure_text}

Table content:
{table_context}

Answer:"""

EVIDENCE_ANSWER_PROMPT = """Question: {question}

Composing the answer failed, but these values were read from the workbook and no check refuted them. Answer using only them; apply every constraint of the question and report only what it asks for.

{evidence}

Answer:"""
