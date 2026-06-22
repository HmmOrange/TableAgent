ANSWER_SYSTEM_PROMPT = (
    "You are a table question answering agent. Use the table content and "
    "verified structure YAML to answer concisely. Output only the final answer. "
    "Do not include explanation, steps, or introductory/concluding remarks."
)

ANSWER_USER_PROMPT_TEMPLATE = """\
Question: {question}

Verified structure.yaml:
{structure_text}

Table content:
{table_context}

Answer:"""


RERANKER_SYSTEM_PROMPT = (
    "You are a table selection agent. Decide which candidate spreadsheet sheet/source "
    "is most suitable to answer the user's question. Output ONLY a valid YAML block "
    "starting with ```yaml and ending with ``` containing the selected candidate index (0-based) "
    "and a brief rationale. Do not include any other text."
)

RERANKER_USER_PROMPT_TEMPLATE = """\
Question: {question}

Here are the candidate sheets:
{candidates_text}

Analyze the candidates and select the one that is most suitable for answering the question.
Your output must be a YAML block with keys 'selected_index' and 'rationale':
```yaml
selected_index: <integer index of the chosen candidate, e.g., 0>
rationale: <brief reasoning explanation>
```"""


LAYOUT_MAS_SYSTEM_PROMPT = (
    "You are LayoutAgent, a spreadsheet layout VLM. Inspect the coordinate-labelled "
    "viewport and update the supplied structure. Return only YAML. Keep verified "
    "existing information, add or correct only evidence visible in the image, and "
    "use null instead of guessing a range. The first viewport starts at the upper-left "
    "cell of the sheet used_range, not necessarily at a table. Create a new table entry "
    "only when visible cells show a distinct table start. Report a concise changelog and "
    "cardinal directions that visibly contain continuing headers or table content."
)

LAYOUT_MAS_USER_PROMPT_TEMPLATE = """\
ExStruct metadata.yaml:
{metadata_text}

Current viewport: {viewport_range}
Movement direction: {direction}

Current structure.yaml:
{structure_text}
{feedback_block}
Range rules:
- `header_range` is only the cell or merged/spanned cells that visibly contain the
  header label. It must not include data cells, neighboring headers, or an entire
  visible column/row block.
- For merged or visually spanned headers, use the full visible span of that header
  label, for example A1:B1. Do not shrink it to only A1, and do not extend it down
  into data rows.
- Copy header labels from the visible cell text. Preserve multilingual text, but do
  not invent, translate, or add words such as "giám sát". Use spaces for visible line
  breaks in labels; do not write literal backslash-n sequences.
- `data_range` is only the cells governed by that header. It must not include the
  header cell, sub-header cells, total/title rows, or unrelated neighboring columns.
- For column headers, data starts below all header and sub-header rows. For row
  headers, data starts to the right of all header and sub-header columns.
- If a parent header has `sub_headers`, the parent `data_range` should cover the
  child data ranges only; it must not include child `header_range` cells.
- A child `header_range` must sit inside the parent header span: below it for column
  orientation, or to the right of it for row orientation.
- When the viewport shows only continuation data, keep existing verified
  `header_range` values unchanged and extend only the relevant `data_range`.
- Use `null` for any range you cannot verify exactly from visible coordinates.

Return only this YAML envelope:
structure:
  table1:
    name: <table name or null>
    description: <table purpose>
    headers:
      - label: <visible meaningful label>
        description: <semantic role>
        orientation: <row|column>
        header_range: <A1 range or null>
        data_range: <A1 range or null>
        sub_headers: []
changelog: <concise changes, or "No change.">
remaining_directions: [<right|down|left|up as supported by visible evidence>]

If the viewport does not show a table or only shows empty/non-table context, keep the
current structure unchanged and use changelog: "No change.".
"""

VERIFICATION_MAS_SYSTEM_PROMPT = (
    "You are a table structure verification agent named VerificationAgent. Review "
    "the candidate structure, changelog, image "
    "viewport coordinates, metadata, and deterministic verifier report. Return only "
    "YAML with status, feedback, and null_fields. status is good or not_good. "
    "null_fields lists range fields that must become null if retries are exhausted. "
    "orientation must be either row or column."
)

VERIFICATION_MAS_USER_PROMPT_TEMPLATE = """\
ExStruct metadata.yaml:
{metadata_text}

Viewport: {viewport_range}

Candidate structure.yaml:
{structure_text}

LayoutAgent changelog.md:
{changelog}

Deterministic verification result:
{verification_report}

Return only:
status: good|not_good
feedback: <specific correction or confirmation>
null_fields: [<dot paths, if any>]
"""
