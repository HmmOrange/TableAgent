"""A single model call that answers from a prompt (and an image when the model takes one).

Used when the workbook cannot be explored: no usable structure.yaml, or the exploration failed.
"""

from __future__ import annotations

from pathlib import Path

from TableAgent.llm import BaseLLM, LLMResponse


class DirectAnswerer:
    def __init__(self, llm: BaseLLM, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, *, prompt: str, image_path: Path | None = None, fallback_prompt: str | None = None) -> LLMResponse:
        """Answer with the image when one is given and the model takes images; otherwise from text."""
        generate_with_image = getattr(self.llm, "generate_with_image", None)
        if image_path is not None and callable(generate_with_image):
            return generate_with_image(prompt=prompt, image_path=image_path, system_prompt=self.system_prompt)
        return self.llm.generate(prompt=fallback_prompt or prompt, system_prompt=self.system_prompt)
