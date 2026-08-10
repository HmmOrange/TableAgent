SYSTEM_PROMPT = """You inspect a spreadsheet image. Return valid JSON only."""

USER_PROMPT_TEMPLATE = """Inspect the viewport of this spreadsheet.

Identify the row headers, row group headers, column headers, and column group
headers visible in the image. Read the hierarchy before writing the JSON.

A leaf header labels one data-bearing row or column. A group header is a
label-only parent or section whose children are multiple rows or columns. For
example, a blank-valued "North" section followed by Revenue and Cost rows is a
row group; Revenue and Cost are row headers. Apply the same rule to columns.
Do not put one occurrence in both a leaf list and a group list.

Return exactly:
{{
  "row_headers": ["..."],
  "row_group_headers": ["..."],
  "column_headers": ["..."],
  "column_group_headers": ["..."]
}}

Use exact visible labels and an empty list when a category is absent. Do not
include titles, notes, ranges, coordinates, explanations, or additional fields.
"""

REPAIR_PROMPT_TEMPLATE = """Your previous response was invalid.

Validation error: {error}

Previous response:
{response}

Return exactly one valid JSON object with only the four required arrays. Every
item must be a non-empty string. Return JSON only.
"""
