"""Answer sentence: restate the computed `final_answer` with its labels and units.

A bare value such as `516` or `0.0424` leaves the reader to guess what it measures; one LLM call states each value
with the item and unit it belongs to, without recomputing anything. It sees only the question, the decisions that
name the terms, and the answer, so it has no other value to add. The computed answer is kept in the log.
"""
from __future__ import annotations

import re
from typing import Any

from TableAgent.stages.qa.prompts.answer_sentence import (
    ANSWER_SENTENCE_SYSTEM_PROMPT,
    ANSWER_SENTENCE_USER_PROMPT_TEMPLATE,
)

FORM_SECTION = re.compile(r"7\.\s*\**\s*Expected Answer Form\s*\**\s*:?(.*)", re.IGNORECASE | re.DOTALL)
# Sections 1-2 (clarified question, term mapping) name what each value is, without any computed value.
CONTEXT_SECTIONS = re.compile(r"1\.\s*\**\s*Clarified Question.*?(?=\n\s*(?:#+\s*)?3\.\s*\**)", re.IGNORECASE | re.DOTALL)
MAX_CONTEXT_CHARS = 2500
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def expected_form(decisions: str | None) -> str:
    """The decision sheet's Expected Answer Form section, or the whole sheet when the section is not found."""
    match = FORM_SECTION.search(decisions or "")
    return (match.group(1) if match else decisions or "").strip()


def context(decisions: str | None) -> str:
    match = CONTEXT_SECTIONS.search(decisions or "")
    return match.group(0).strip() if match else ""


def _numbers(text: str) -> list[float]:
    values = []
    for token in NUMBER.findall(text):
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return values


def numbers_supported(sentence: str, answer: str, question: str) -> bool:
    """Whether every number in the sentence comes from the computed answer (or its rounding) or the question."""
    source = _numbers(answer)
    allowed = set(source) | set(_numbers(question)) | {round(v, digits) for v in source for digits in range(4)}
    return all(any(abs(value - a) < 1e-9 for a in allowed) for value in _numbers(sentence))


class AnswerSentenceAction:
    def __init__(self, env: Any, llm_client: Any):
        self.env = env
        self.llm_client = llm_client

    def run(self, question: str, decisions: str | None, answer: str) -> str | None:
        """The answer restated as a short sentence, or None when the call fails (the computed answer then stands)."""
        prompt = ANSWER_SENTENCE_USER_PROMPT_TEMPLATE.format(
            question=question,
            expected_form=expected_form(decisions) or "(none)",
            answer=answer,
            context=context(decisions)[:MAX_CONTEXT_CHARS] or "(none)",
        )
        try:
            content = str(self.llm_client.generate(prompt, system_prompt=ANSWER_SENTENCE_SYSTEM_PROMPT).content or "")
        except Exception as exc:
            self.env.logger.log_event("answer_sentence_error", {"error": str(exc)})
            return None
        sentence = content.strip()
        if sentence and not numbers_supported(sentence, answer, question):
            # A number the computed answer does not contain was made up; keep the computed answer.
            self.env.logger.log_event("answer_sentence_rejected", {"sentence": sentence})
            return None
        return sentence or None


__all__ = ["AnswerSentenceAction", "context", "expected_form", "numbers_supported"]
