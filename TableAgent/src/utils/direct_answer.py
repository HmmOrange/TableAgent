"""A single model call that answers from a prompt. Used when the exploration of the workbook failed."""

from __future__ import annotations

from TableAgent.llm import BaseLLM, LLMResponse


class DirectAnswerer:
    def __init__(self, llm: BaseLLM, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, *, prompt: str) -> LLMResponse:
        return self.llm.generate(prompt=prompt, system_prompt=self.system_prompt)
