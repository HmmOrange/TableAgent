from TableAgent.llm import LLMResponse


def token_usage(responses: list[LLMResponse]) -> dict[str, int]:
    return {
        "prompt": sum(response.prompt_tokens for response in responses),
        "completion": sum(response.completion_tokens for response in responses),
    }


__all__ = ["token_usage"]
