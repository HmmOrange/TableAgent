from __future__ import annotations

from dataclasses import dataclass
import re

from utils.llm.base import BaseLLM, LLMResponse


SIFLEX_FORMATTER_SYSTEM_PROMPT = """You are the SIFLEX answer-formatting agent.
Your only task is to reshape a completed spreadsheet-QA answer into the requested
benchmark presentation type.

Strict rules:
- Preserve every fact, value, unit, row relationship, and the answer language.
- Do not calculate, infer, correct, summarize, omit, or add information.
- Use the draft's meaningful user-facing labels; never expose internal field IDs.
- Return only the reformatted answer, with no commentary or code fence.
- For `table`, return a valid Markdown table.
- For `list`, return a Markdown bullet list with one answer unit per item.
- For `form`, return a clearly sectioned document; retain tables when they carry
  row or field relationships.
"""

SIFLEX_FORMATTER_USER_PROMPT = """Question:
{question}

Required SIFLEX answer type: {answer_type}

Draft answer to reformat:
{draft_answer}

Reformat the draft now. Return only the answer."""


@dataclass(frozen=True)
class SiflexFormatterResult:
    answer: str
    response: LLMResponse
    fallback_used: bool = False


class SiflexAnswerFormatterAgent:
    supported_types = frozenset({"table", "list", "form"})

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    @classmethod
    def supports(cls, answer_type: str) -> bool:
        return str(answer_type).strip().lower() in cls.supported_types

    def run(self, *, question: str, answer_type: str, draft_answer: str) -> SiflexFormatterResult:
        normalized_type = str(answer_type).strip().lower()
        try:
            response = self.llm.generate(
                prompt=SIFLEX_FORMATTER_USER_PROMPT.format(
                    question=question,
                    answer_type=normalized_type,
                    draft_answer=draft_answer,
                ),
                system_prompt=SIFLEX_FORMATTER_SYSTEM_PROMPT,
            )
        except Exception as exc:
            response = LLMResponse(content=f"ERROR: SIFLEX formatter failed: {exc}")
        formatted = self._clean_response(response.content)
        if not formatted or formatted.upper().startswith("ERROR:"):
            return SiflexFormatterResult(
                answer=draft_answer,
                response=response,
                fallback_used=True,
            )
        return SiflexFormatterResult(answer=formatted, response=response)

    @staticmethod
    def _clean_response(content: str) -> str:
        text = str(content or "").strip()
        fenced = re.fullmatch(r"```(?:markdown|md)?\s*\n?(.*?)\n?```", text, flags=re.IGNORECASE | re.DOTALL)
        return fenced.group(1).strip() if fenced else text
