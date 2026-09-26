from __future__ import annotations

UNDERSTANDING_SYSTEM_PROMPT = """You are an expert Excel data analyst.
Your main job is to clarify what a spreadsheet question means in terms of the workbook,
so that a planner can decompose it consistently. Do not answer the question."""

UNDERSTANDING_USER_PROMPT_TEMPLATE = """I need you to clarify a question against the spreadsheet content before it is answered.

**User Question:** {question}

**Excel Workbook Content:**
{workbook_content}

**Your Task:**
Clarify the question using the Excel content and provide analysis in the following format EXACTLY. Do NOT provide the actual answer to the user's question or compute any values - only clarify the question and provide the analysis framework:

1. **Question Clarification** (most important):
- **Clarified Question**: Rewrite the question as one precise, unambiguous sentence using the workbook's own terminology
- **Term Mapping**: Map every entity, metric, category, and time expression in the question to the exact sheet name, header label, or row label as it appears in the workbook (quote the label and give its cell coordinate); note abbreviations, synonyms, and units or scale (e.g., "in thousands")
- **Constraints**: List every explicit and implicit filter, time period, grouping, and condition that must hold
- **Ambiguities & Chosen Interpretation**: List each term or requirement that could be read in more than one way, the candidate readings, and the single reading best supported by the question and workbook, with a short reason
- **Required Operation**: Name the operation needed (lookup, filter, count, sum, average, difference, ratio, ranking, comparison, description) and the order of steps, without computing it
- **Expected Answer Form**: Describe the answer type, unit, precision, and format (single value, list and its order, comparison, text)

2. **Sheet Summary**:
Provide a comprehensive overview including:
- **Workbook Purpose & Domain**: Identify the business context, industry, and primary use case
- **Sheet Organization**: Describe how sheets are logically organized and their relationships
- **Data Structure & Types**: Catalog numerical data, text, dates, calculated fields, and hierarchical relationships

3. **Problem Insights**:
- **Relevant Data Scope**: Identify which specific sheets, ranges, or data points are most relevant
- **Potential Challenges**: Identify data structure complexities that might affect analysis
- **Validation Strategy**: Recommend ways to verify the accuracy of results
- **Hierarchical Data Considerations**: Note any parent-child relationships, subtotals, or nested categories
"""
