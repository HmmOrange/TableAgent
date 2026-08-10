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


__all__ = ["RERANKER_SYSTEM_PROMPT", "RERANKER_USER_PROMPT_TEMPLATE"]
