"""Compatibility exports for the public TableAgent model integration."""

from TableAgent.integrations.models import (
    OPENAI_COMPATIBLE_PROVIDERS,
    OpenAICompatibleLLM,
    create_model_client,
)

__all__ = [
    "OPENAI_COMPATIBLE_PROVIDERS",
    "OpenAICompatibleLLM",
    "create_model_client",
]
