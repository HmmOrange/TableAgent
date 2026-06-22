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
    "never output null, UNKNOWN, or placeholder range values. The first viewport starts at the upper-left "
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
- Never write `null`, `UNKNOWN`, `N/A`, or placeholder range values. If a range is
  already concrete and you cannot improve it, keep it unchanged. If VerificationAgent
  asks for a field that is currently null, fill it only with an exact concrete A1
  range visible in this viewport.

Return only this YAML envelope:
structure:
  table1:
    name: <table name>
    description: <table purpose>
    headers:
      - label: <visible meaningful label>
        description: <semantic role>
        orientation: <row|column>
        header_range: <exact A1 range>
        data_range: <exact A1 range>
        sub_headers: []
changelog: <concise changes, or "No change.">
remaining_directions: [<right|down|left|up as supported by visible evidence>]

If the viewport does not show a table or only shows empty/non-table context, keep the
current structure unchanged and use changelog: "No change.".
"""

VERIFICATION_MAS_SYSTEM_PROMPT = (
    "You are a table structure verification agent named VerificationAgent. Use a "
    "ReAct pattern over the candidate structure and deterministic verifier report: "
    "Thought, Action, Observation, then final status. The deterministic verifier is "
    "a tool observation, not the only judge; it can be too strict about minor OCR, "
    "line-break, multilingual-label, or span issues. Return only YAML. If semantic "
    "review can confidently fix the structure, include updated_structure with the "
    "full corrected structure.yaml. status is good or not_good. null_fields lists "
    "range fields that must become null if retries are exhausted. orientation must "
    "be either row or column."
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

Use this ReAct-style YAML envelope:
thought: <brief diagnosis grounded in the deterministic report and visible coordinate semantics>
action: <accept|repair_structure|request_layout_retry|null_on_retry_exhaustion>
observation: <what the deterministic report proves and what semantic review resolves>
status: good|not_good
feedback: <specific correction or confirmation>
null_fields: [<dot paths, if any>]
updated_structure:
  table1:
    name: <table name>
    description: <table purpose>
    headers:
      - label: <visible meaningful label>
        description: <semantic role>
        orientation: <row|column>
        header_range: <exact A1 range or null>
        data_range: <exact A1 range or null>
        sub_headers: []

Rules:
- Include updated_structure only when you are correcting or accepting a complete
  structure.yaml. Omit it if LayoutAgent must inspect another viewport.
- Preserve existing correct tables and headers; change only fields needed by the
  deterministic observation or semantic review.
- A deterministic mismatch caused only by harmless whitespace, line breaks,
  multilingual text normalization, or visually obvious span semantics can be marked
  good after updated_structure fixes the persisted YAML.
"""
