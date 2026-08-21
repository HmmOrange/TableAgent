SYSTEM_PROMPT = """You inspect an entire spreadsheet image.
Return YAML only. Identify visible worksheet headers and semantic groups."""

USER_PROMPT_TEMPLATE = """Inspect the entire worksheet image.

Workbook: {workbook_name}
Worksheet: {sheet_name}
Complete used range: {viewport_range}

Return exactly these two flat lists:

headers:
  - "<exact visible header name>"
groups:
  - "<exact visible semantic group name>"

A header names what a data field measures. Include visible parent and child
header names. Do not include titles or individual record labels as headers.

A group is a visible section label whose meaning applies to a contiguous block
of multiple records or data cells. The label supplies shared context to every
record in that block. Repeated record sequences beneath peer section labels are
strong group evidence. Include the first peer when it owns a following detail
block, including an overall or total section.

Do not return table headers, parent header bands, individual records, internal
headings that only name the kind of records below them, titles, notes, footnotes,
or sources as groups.

Double-quote every name. Use an empty list when absent. Do not include ids,
ranges, descriptions, hierarchy, coordinates, explanations, Markdown fences, or
additional fields.
"""

REPAIR_PROMPT_TEMPLATE = """Your previous response was invalid.

Validation error: {error}

Previous response:
{response}

Return valid YAML with exactly flat headers and groups lists. Every item must be
a double-quoted non-empty string. Return YAML only.
"""
