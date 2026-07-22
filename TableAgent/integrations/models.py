"""Public model-client integration for applications embedding TableAgent."""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import requests

from TableAgent.configs import resolve_llm_config, resolve_vlm_config
from TableAgent.llm import BaseLLM, LLMResponse


OPENAI_COMPATIBLE_PROVIDERS = {
    "gpt_oss",
    "openai",
    "openai_compatible",
    "openrouter",
}


class OpenAICompatibleLLM(BaseLLM):
    """Minimal synchronous client for OpenAI-compatible model APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout_seconds: float = 180,
        max_retries: int = 2,
        retry_delay_seconds: float = 1,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        session: requests.Session | None = None,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_delay_seconds = max(0, retry_delay_seconds)
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.session = session or requests.Session()

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        messages = self._messages(prompt, system_prompt=system_prompt)
        return self._complete(messages)

    def generate_with_image(
        self,
        prompt: str,
        image_path: str | Path,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            },
        ]
        return self._complete(self._messages(content, system_prompt=system_prompt))

    @staticmethod
    def _messages(prompt: Any, *, system_prompt: str | None) -> list[dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.extra_body)

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if (
                    response.status_code not in {408, 409, 429}
                    and response.status_code < 500
                ):
                    break
                response.raise_for_status()
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay_seconds * (attempt + 1))
        if response is None:
            raise RuntimeError("Model request did not produce a response")
        response.raise_for_status()
        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Model response is missing choices[0].message") from exc
        content = _content_text(
            message.get("content")
            or message.get("reasoning")
            or message.get("reasoning_content")
        )
        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        token_capped = str(choice.get("finish_reason") or "").lower() == "length" or (
            self.max_tokens is not None and completion_tokens >= self.max_tokens
        )
        return LLMResponse(
            content=content,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=completion_tokens,
            token_capped=token_capped,
        )

    def close(self) -> None:
        self.session.close()


def create_model_client(
    config: dict[str, Any],
    *,
    kind: str,
    profile: str = "table_agent",
    session: requests.Session | None = None,
) -> OpenAICompatibleLLM:
    """Construct a public model client from TableAgent-style configuration."""

    if kind == "llm":
        profile_name, model_config = resolve_llm_config(config, profile)
    elif kind == "vlm":
        profile_name, model_config = resolve_vlm_config(config, profile)
    else:
        raise ValueError("kind must be 'llm' or 'vlm'")

    provider = str(model_config.get("provider", "openai_compatible")).lower()
    if provider not in OPENAI_COMPATIBLE_PROVIDERS:
        raise ValueError(
            f"Model profile '{profile_name}' uses unsupported service provider '{provider}'. "
            "Use an OpenAI-compatible endpoint or inject a custom client."
        )
    base_url = model_config.get("base_url") or model_config.get("endpoint")
    if not base_url:
        if provider == "openai":
            base_url = "https://api.openai.com/v1"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        else:
            raise ValueError(f"Model profile '{profile_name}' is missing base_url")
    model_name = model_config.get("model") or model_config.get("model_name")
    if not model_name:
        raise ValueError(
            f"Model profile '{profile_name}' is missing model or model_name"
        )

    return OpenAICompatibleLLM(
        base_url=str(base_url),
        model_name=str(model_name),
        api_key=model_config.get("api_key"),
        temperature=float(model_config.get("temperature", 0.0)),
        max_tokens=_optional_int(model_config.get("max_tokens")),
        timeout_seconds=float(model_config.get("timeout_seconds", 180)),
        max_retries=int(model_config.get("max_retries", 2)),
        retry_delay_seconds=float(model_config.get("retry_delay_seconds", 1)),
        extra_headers=model_config.get("headers"),
        extra_body=model_config.get("extra_body"),
        session=session,
    )


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "OPENAI_COMPATIBLE_PROVIDERS",
    "OpenAICompatibleLLM",
    "create_model_client",
]
