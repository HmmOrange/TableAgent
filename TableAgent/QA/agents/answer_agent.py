from __future__ import annotations

from pathlib import Path

from TableAgent.agents.base import BaseTableAgent
from TableAgent.llm import BaseLLM, LLMResponse


class QAAgent(BaseTableAgent):
    name = "QAAgent"
    profile = "Table question answering agent"
    goal = "Produce the final answer after layout traversal is complete."

    def __init__(self, llm: BaseLLM, system_prompt: str):
        super().__init__()
        self.llm = llm
        self.system_prompt = system_prompt

    def run(
        self,
        *,
        prompt: str,
        image_path: Path | None = None,
        fallback_prompt: str | None = None,
    ) -> LLMResponse:
        generate_with_image = getattr(self.llm, "generate_with_image", None)
        if image_path is not None and callable(generate_with_image):
            return generate_with_image(prompt=prompt, image_path=image_path, system_prompt=self.system_prompt)
        return self.llm.generate(prompt=fallback_prompt or prompt, system_prompt=self.system_prompt)


__all__ = ["QAAgent"]
