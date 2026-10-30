"""Model calls of the QA loop: every call is counted and logged; replies are JSON objects."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from TableAgent.stages.qa.utils.trace import RunLog

LOGGED_REPLY_CHARS = 4000


class ReplyError(Exception):
    """A model reply that could not be used, even after one retry."""


class CountingLLM:
    """Wraps an LLM client (and optionally a vision client): counts calls and tokens, and logs each call."""

    def __init__(self, client: Any, log: RunLog, vision_client: Any = None):
        self.client = client
        self.vision_client = vision_client
        self.log = log
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, system_prompt: str) -> str:
        return self._call(lambda: self.client.generate(prompt, system_prompt=system_prompt))

    def generate_with_image(self, prompt: str, image_path: Path, system_prompt: str) -> str:
        return self._call(
            lambda: self.vision_client.generate_with_image(prompt, image_path, system_prompt=system_prompt),
            image=str(image_path),
        )

    def _call(self, request: Callable[[], Any], **details: Any) -> str:
        started = time.perf_counter()
        call: dict[str, Any] = {"index": len(self.calls) + 1, "success": False, **details}
        try:
            response = request()
        except Exception as exc:
            call["error_type"] = type(exc).__name__
            raise
        else:
            call.update({
                "success": True,
                "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(response, "completion_tokens", 0) or 0),
                "token_capped": bool(getattr(response, "token_capped", False)),
            })
            return str(response.content or "")
        finally:
            call["duration_ms"] = round((time.perf_counter() - started) * 1000)
            self.calls.append(call)
            self.log.event("llm_call", **call)

    @property
    def used(self) -> int:
        return len(self.calls)

    def token_usage(self) -> dict[str, int]:
        return {
            "prompt": sum(call.get("prompt_tokens", 0) for call in self.calls),
            "completion": sum(call.get("completion_tokens", 0) for call in self.calls),
        }


def ask_json(
    llm: CountingLLM,
    kind: str,
    system_prompt: str,
    prompt: str,
    check: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    """Ask for one JSON object and validate it with `check`; a bad reply gets one retry quoting the problem."""
    llm.log.event(f"{kind}_prompt", prompt=prompt)
    for attempt in (1, 2):
        try:
            content = llm.generate(prompt, system_prompt)
        except Exception as exc:
            raise ReplyError(f"{kind} call failed: {exc}") from exc
        try:
            reply = check(parse_json(content))
        except ValueError as exc:
            llm.log.event(f"{kind}_reply_rejected", attempt=attempt, error=str(exc), content=content[:LOGGED_REPLY_CHARS])
            prompt = f"{prompt}\n\nYour previous reply was rejected: {exc}\nReply again in the requested format."
            continue
        llm.log.event(f"{kind}_reply", **reply)
        return reply
    raise ReplyError(f"{kind} reply was invalid twice")


def parse_json(content: str) -> Any:
    """The JSON object in a reply: a ```json block, the whole text, or the first {...} in it.

    A ```python block outside the object becomes its "code", so code needs no JSON escaping.
    Raw line breaks inside JSON strings are accepted.
    """
    code = re.search(r"```python[ \t]*\n(.*?)```", content, re.DOTALL)
    if code:
        content = content[:code.start()] + content[code.end():]
    data = _json_object(content)
    if code and isinstance(data, dict):
        data["code"] = code.group(1).strip()
    return data


def _json_object(content: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    text = fenced.group(1) if fenced else content.strip()
    try:
        return json.loads(text, strict=False)
    except ValueError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    decoder = json.JSONDecoder(strict=False)
    try:
        return decoder.raw_decode(text[start:])[0]
    except ValueError as exc:
        try:
            return decoder.raw_decode(_escape_inner_quotes(text[start:]))[0]
        except ValueError:
            raise ValueError(f"invalid JSON: {exc}") from exc


def _escape_inner_quotes(text: str) -> str:
    """Escape a quote inside a string unless what follows it can only follow a closing quote."""
    out, in_string, escaped = [], False, False
    for index, char in enumerate(text):
        if in_string and escaped:
            escaped = False
        elif in_string and char == "\\":
            escaped = True
        elif char == '"':
            if not in_string:
                in_string = True
            elif text[index + 1:].lstrip()[:1] in {",", "}", "]", ":", ""}:
                in_string = False
            else:
                out.append("\\")
        out.append(char)
    return "".join(out)


def require_dict(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("the reply must be a JSON object")
    return data
