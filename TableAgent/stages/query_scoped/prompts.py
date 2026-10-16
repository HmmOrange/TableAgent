"""Prompts for the text-first, query-scoped structure path."""

STATIC_PROMPT_VERSION = "query-static-v2-overlap-chain"

SUMMARY_SYSTEM_PROMPT = """You index spreadsheet regions for later question answering.
Describe only evidence present in the supplied A1-addressed cells. Be concise but specific:
identify headers, entities, measures, time periods, units, repeated sections, and the useful row
and column ranges. Preserve exact sheet names and A1 ranges. Return plain text, not JSON/YAML."""

CHUNK_SUMMARY_TEMPLATE = """Summarize this spreadsheet region for retrieval.

Workbook: {workbook_name}
Sheet: {sheet_name}
Region: {region_id} ({cell_range})

Header context (may overlap the region):
{header_context}

Summary of the immediately preceding region (empty for the first region):
{previous_summary}

Region cells:
{region_cells}
"""

SHEET_SUMMARY_TEMPLATE = """Create a detailed retrieval-oriented overview of this worksheet.
Explain which regions contain which data, their shape/layout, headers, measures, dimensions,
units, time periods, and relationships between regions. Cite region IDs and A1 ranges exactly.
Do not invent content. Return flexible plain text.

Workbook: {workbook_name}
Sheet: {sheet_name}

Region summaries:
{region_summaries}
"""

WORKBOOK_SUMMARY_TEMPLATE = """Create a detailed retrieval-oriented overview of this workbook.
Describe what each sheet and data region contains, how the data is organized, and which regions
are likely useful for different kinds of questions. Cite sheet names, region IDs, and A1 ranges.
Do not invent content. Return flexible plain text.

Workbook: {workbook_name}

Worksheet summaries:
{sheet_summaries}
"""

LOCATE_SYSTEM_PROMPT = """You locate spreadsheet regions needed to answer a question.
Use only region IDs from the catalog. Select the smallest sufficient set, including lookup keys,
labels, units, and comparison/aggregation inputs. Return JSON only in this exact shape:
{"selected_regions":[{"region_id":"...","reason":"..."}]}"""

LOCATE_PROMPT_TEMPLATE = """Question:
{question}

Workbook overview:
{workbook_summary}

Available regions:
{region_catalog}

Select at most {max_regions} regions. Return JSON only.
"""

STRUCTURE_SYSTEM_PROMPT = """You extract a compact TableAgent structure for one worksheet.
Use only the supplied A1-addressed evidence and only fields relevant to the question. Every range
must be a valid A1 range on the named sheet. Return YAML only, with no prose or code fence.

Bind every scalar field to an exact leaf column (or exact leaf row for row-oriented data), never
to a broad merged parent. For a multi-level header, compose `label` from the visible path from
merged ancestors to the leaf, for example `Men - Unemployment rates - 2023`. Its `header_range`
must follow that same path while staying on the leaf axis (for example E4:E6), and its
`data_range` must stay on exactly that leaf axis (for example E7:E40). If the question truly
needs a parent spanning several leaf columns, represent the parent with `sub_headers`, one child
per needed leaf. A scalar header with `sub_headers: []` must not span multiple data columns.

Required schema:
structure:
  table1:
    id: concise_unique_id
    name: descriptive table name
    description: why this table is relevant to the question
    sheet: exact sheet name
    headers:
      - id: concise_header_id
        label: exact semantic label
        description: meaning, unit, and role for answering the question
        orientation: column
        header_range: A1
        data_range: A2:A10
        sub_headers: []

Add more tables only if the evidence truly contains separate tables. Omit irrelevant columns.
Quote YAML free-text scalars when punctuation could be ambiguous."""

STRUCTURE_PROMPT_TEMPLATE = """Question:
{question}

Workbook: {workbook_name}
Sheet: {sheet_name}
Selected region IDs: {region_ids}

Header context:
{header_context}

Potentially relevant cells (A1-addressed):
{region_cells}
"""

REPAIR_PROMPT_TEMPLATE = """Correct the candidate structure using the verifier feedback.
Keep it compact and question-specific. Resolve merged/hierarchical headers all the way to exact
leaf columns: include ancestor and leaf text in the label, keep scalar data ranges one column
wide, and use sub_headers only for a real multi-leaf parent. Return the complete YAML only.

Question: {question}
Workbook: {workbook_name}
Sheet: {sheet_name}

Evidence:
{evidence}

Candidate structure:
{candidate}

Deterministic verifier feedback:
{feedback}
"""
