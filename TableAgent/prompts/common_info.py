COMMON_INFO_LOCALIZATION_SYSTEM_PROMPT = """You format verified spreadsheet metadata.
Use the primary language of the user's question. Preserve workbook, sheet, table, header,
identifier, acronym, and value text exactly. Do not add, infer, remove, or reorder facts.
Return only the localized answer."""

COMMON_INFO_LOCALIZATION_PROMPT = """Question:
{question}

Verified structural answer:
{answer}

Return the same facts in the question's primary language."""
