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


__all__ = ["ANSWER_SYSTEM_PROMPT", "ANSWER_USER_PROMPT_TEMPLATE"]
