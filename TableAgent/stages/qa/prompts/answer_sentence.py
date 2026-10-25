from __future__ import annotations

ANSWER_SENTENCE_SYSTEM_PROMPT = """You write the final answer to a spreadsheet question from an answer that was already computed.
You never recompute, correct, or replace a value. Return only the answer text."""

ANSWER_SENTENCE_USER_PROMPT_TEMPLATE = """Question: {question}

Expected answer form from the question decisions:
{expected_form}

Computed answer: {answer}

Clarified question and term mapping from the question decisions:
{context}

Write the answer as one or two short sentences:
- State every value or name of the computed answer with the item, label, or category it belongs to and its unit
  (for example "Public Transport / Walking: 932,200 people; Cycle: 527,900 people", or "516 days").
- Keep each value exactly as computed. When a number has more than two decimal places, add it rounded to the
  precision the question states, or else to two decimal places, in parentheses: "0.0424 (about 0.04)"; add nothing
  when rounding would not change it.
- When the question asks whether something holds, start with Yes or No.
- Name each item of a list once, even when the computed answer repeats it.
- The answer contains only the values and names of the computed answer: add no other value, item, or name, and do
  not explain how it was computed.
"""
