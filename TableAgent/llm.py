from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_capped: bool = False


class BaseLLM(ABC):
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        response_schema: dict | None = None,
    ) -> LLMResponse:
        """Generate a completion, optionally constrained to a JSON schema.

        `response_schema` is advisory: a backend that cannot constrain decoding ignores
        it and returns ordinary text, so callers still parse defensively.
        """
        raise NotImplementedError
