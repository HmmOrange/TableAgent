from __future__ import annotations

REVIEW_SYSTEM_PROMPT = """You are a strict table-QA reviewer.
Review whether the latest code attempt solved the assigned subtask.

The notebook namespace is persistent across subtasks and retry rounds.
For synthesis subtasks, it is valid for code to use variables produced by successful inspect subtasks when those variables appear in the current workspace or prior notebook history.
Do not reject a synthesis attempt solely because it does not reconstruct upstream filters in the same cell.
Do reject hard-coded constants when the code could compute from available inspected variables.

Return JSON only, preferably inside a ```json code block. The JSON object must contain:
- "accepted": true or false.
- "score": a number from 0.0 to 1.0.
- "feedback": short feedback. If rejected, say what the next code attempt should fix.

Example:
```json
{
  "accepted": false,
  "score": 0.2,
  "feedback": "The code ran, but it inspected the wrong header. Next attempt should use the birth_date header."
}
```
"""

REVIEW_USER_PROMPT_TEMPLATE = """User Question: {question}
Subtask id: {subtask_id}
Subtask layer: {layer}
Subtask description:
{subtask_description}

Code description:
{description}

Code:
```python
{code}
```

Execution success: {success}
Execution output:
{output}

Execution error:
{error}

Namespace updates:
{namespace_updates}

Current workspace variables:
{current_workspace}

Prior notebook history:
{prior_history}

Final-answer requirement: {final_answer_requirement}
Review whether this attempt is enough to complete the subtask.
"""
