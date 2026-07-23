from __future__ import annotations


COMMON_INFO_LANGUAGE_SYSTEM_PROMPT = """You localize a verified spreadsheet common-information answer.
Return only the final localized Markdown answer, with no commentary or code fences.

Requirements:
- Use exactly the required answer language supplied in the user prompt.
- Translate every generic heading, explanatory sentence, and header description into that language.
- A response containing English or another language in translatable prose is invalid unless it is part of a preserved
  sheet name, table name, header label, acronym, or identifier.
- Translate only explanatory prose and generic structural labels.
- Preserve sheet names, table names, header labels, numbers, facts, and Markdown structure exactly.
- Do not add, remove, infer, summarize, or correct spreadsheet information.
"""


COMMON_INFO_LANGUAGE_USER_PROMPT_TEMPLATE = """User Question:
{question}

Required answer language: {answer_language}

Verified common-information answer:
{answer}

Return the same answer localized to the language of the user question.
"""
