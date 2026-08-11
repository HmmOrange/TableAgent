DIRECTION_SYSTEM_PROMPT = (
    "You inspect a spreadsheet image for unexplored table branches. "
    "Return valid JSON only."
)


DIRECTION_USER_PROMPT_TEMPLATE = """\
Inspect the coordinate-labelled spreadsheet viewport for unexplored table branches.

Workbook: {workbook_name}
Sheet: {sheet_name}
Current viewport: {viewport_range}
Movement direction: {direction}

Return exactly one JSON object:
{{"remaining_directions": ["right", "down"]}}

`remaining_directions` is only for unexplored perpendicular branches visible
from the current viewport. Include a direction only when the visible edge shows
a potential table header: a label-bearing, merged/spanned, or distinctly
header-formatted row or column that appears to continue beyond the viewport.
Do not include data-only continuation, worksheet bounds, blank grid cells, or
ordinary table content. Do not include the current movement axis or its
opposite. Output at most two directions and never repeat a direction. Use []
when no additional branch is visible.
"""
