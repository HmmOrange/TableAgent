LAYOUT_MAS_SYSTEM_PROMPT = (
    "You are LayoutAgent, a spreadsheet layout VLM. Inspect the complete coordinate-labelled "
    "worksheet used range and update the supplied structure. Return only YAML. Keep verified "
    "existing information, add or correct only evidence visible in the image, and "
    "quote every free-text scalar, including names, labels, descriptions, worksheet names, and changelog text. "
    "The letters above the grid and numbers beside the grid are renderer-added coordinate guides, not workbook cells; "
    "never turn them into headers or data. "
    "never output null, UNKNOWN, or placeholder range values. Create a new table entry "
    "when visible cells show a distinct table start. Report a concise changelog. "
    "Inspect every visible table; do not output traversal directions."
)

LAYOUT_MAS_USER_PROMPT_TEMPLATE = """\
ExStruct metadata.yaml:
{metadata_text}

Complete worksheet used range: {sheet_range}

Current structure.yaml:
{structure_text}
{feedback_block}
Range rules:
- Ignore the renderer-added coordinate guides outside the cell grid: column letters
  across the top and row numbers down the left. They are not workbook content and
  must never become a header such as `row_index`, a label, or part of any range.
  Cell A1 is the first actual workbook cell inside those guides.
- `orientation` describes where the governed data extends from the header: use
  `column` when values continue downward (including a leftmost label column), and
  `row` only when values continue horizontally to the right.
- Determine orientation from the header's semantic relationship to its values, not
  from the shape of `header_range`. Before choosing it, identify representative
  visible values that answer "what values does this header name?" Those values must
  be inside `data_range`. Choose `column` when the governed values are successive
  entries of that semantic field down the sheet, and choose `row` when they are
  successive entries of that semantic field across the sheet. The shape or merged
  extent of the header cell does not determine orientation.
- Perform a final semantic self-check for every header: its label and description
  must truthfully describe representative cells in `data_range`; `column` values
  must be below the header and `row` values must be to its right. Correct any header
  that fails this check before returning YAML.
- `header_range` is only the cell or merged/spanned cells that visibly contain the
  header label. It must not include data cells, neighboring headers, or an entire
  visible column/row block.
- For merged or visually spanned headers, use the full visible span of that header
  label, for example A1:B1. Do not shrink it to only A1, and do not extend it down
  into data rows.
- Copy header labels from the visible cell text. Preserve multilingual text, but do
  not invent, translate, or add words such as "giám sát". Use spaces for visible line
  breaks in labels; do not write literal backslash-n sequences.
- Wrap every free-text scalar in double quotes, including table names, labels,
  descriptions, worksheet names, and changelog text. Escape embedded double quotes.
  Keep identifiers, orientations, and ranges as structured values.
- `data_range` is only the cells governed by that header. It must not include the
  header cell, sub-header cells, total/title rows, or unrelated neighboring columns.
- For column headers, data starts below all header and sub-header rows. For row
  headers, data starts to the right of all header and sub-header columns.
- If a parent header has `sub_headers`, the parent `data_range` should cover the
  child data ranges only; it must not include child `header_range` cells.
- Use `sub_headers` only for visible parent-child groups. For column orientation,
  a child's header and data columns must be contained within the parent's columns;
  for row orientation, its rows must be contained within the parent's rows.
  Adjacent non-contained spans are siblings, even when semantically related.
- Before returning, check every child against this containment rule. If it fails,
  move the intact header to the nearest containing ancestor or the top-level
  `headers` list; do not alter correct coordinates merely to force containment.
- `sub_headers` is recursive and may contain any number of header levels. Every child
  uses the same header schema and may declare its own `sub_headers`. Use
  `sub_headers: []` only when that header has no visible children.
- Preserve the complete visible hierarchy. Never flatten nested group headers into
  the top-level `headers` list, and never discard grandchildren or deeper descendants.
- A parent or intermediate group `data_range` must cover the union of all descendant
  leaf data ranges governed by that header.
- When the image shows only continuation data, keep existing verified
  `header_range` values unchanged and extend only the relevant `data_range`.
  Never replace an existing `data_range` with only the current image slice.
- When extending a `data_range`, use the union of the old range and newly visible
  cells governed by the same header. Example: if an existing column range is A2:A20
  and rows 16-35 continue the same data, the updated range must be A2:A35, not
  A16:A35. For horizontal continuation, union columns the same way.
- Do not create separate headers for blank cells inside a merged or visually spanned
  header. Use the full span visible in the image, such as C1:L1, instead of C1:C1
  plus fake continued headers.
- Never write `null`, `UNKNOWN`, `N/A`, or placeholder range values. If a range is
  already concrete and you cannot improve it, keep it unchanged. If the deterministic verifier
  asks for a field that is currently null, fill it only with an exact concrete A1
  range visible in this image.

Return only this YAML envelope. Each mapping directly under `structure` is one
table. Table keys may be any non-empty YAML key; prefer a stable descriptive key
and preserve existing keys exactly when refining a structure.

structure:
  revenue_summary:
    id: <unique stable snake_case table identifier>
    name: "<table name>"
    description: "<table purpose>"
    sheet: "<exact worksheet name from metadata>"
    headers:
      - id: <unique stable snake_case identifier>
        label: "<visible meaningful label>"
        description: "<semantic role>"
        orientation: <row|column>
        header_range: <exact A1 range>
        data_range: <exact A1 range>
        sub_headers:
          - id: <unique stable snake_case child identifier>
            label: "<visible child label>"
            description: "<semantic role>"
            orientation: <row|column>
            header_range: <exact A1 range>
            data_range: <exact A1 range>
            sub_headers:
              - id: <unique stable snake_case descendant identifier>
                label: "<visible descendant label>"
                description: "<semantic role>"
                orientation: <row|column>
                header_range: <exact A1 range>
                data_range: <exact A1 range>
                sub_headers: []
  regional_breakdown:
    <table details here if exists>
changelog: "<concise changes, or No change.>"

If the image does not show a table or only shows empty/non-table context, keep the
current structure unchanged and use changelog: "No change.". Return only the YAML
envelope above.
"""
