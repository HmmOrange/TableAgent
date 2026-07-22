from __future__ import annotations

from pathlib import Path

from TableAgent.integrations.models import OpenAICompatibleLLM, create_model_client


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }


class FakeSession:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or FakeResponse()

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    def close(self):
        return None


def test_openai_compatible_client_supports_text_and_images(tmp_path: Path):
    session = FakeSession()
    client = OpenAICompatibleLLM(
        base_url="http://model.test/v1/",
        model_name="model-a",
        api_key="secret",
        max_tokens=123,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        session=session,
    )

    text_result = client.generate("question", system_prompt="system")
    image_path = tmp_path / "sheet.png"
    image_path.write_bytes(b"png-data")
    image_result = client.generate_with_image(
        "inspect", image_path, system_prompt="layout"
    )

    assert text_result.content == "done"
    assert text_result.prompt_tokens == 7
    assert image_result.completion_tokens == 3
    assert session.calls[0][0] == "http://model.test/v1/chat/completions"
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer secret"
    assert session.calls[0][1]["json"]["max_tokens"] == 123
    assert session.calls[0][1]["json"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    image_content = session.calls[1][1]["json"]["messages"][1]["content"]
    assert image_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_create_model_client_resolves_public_config():
    config = {
        "llm": {"provider": "answer"},
        "models": {
            "answer": {
                "provider": "openai_compatible",
                "base_url": "http://localhost:9000/v1",
                "model": "answer-model",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
        },
    }

    client = create_model_client(config, kind="llm")

    assert client.base_url == "http://localhost:9000/v1"
    assert client.model_name == "answer-model"
    assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_openai_compatible_client_handles_reasoning_only_response():
    class ReasoningResponse(FakeResponse):
        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": None, "reasoning": "reasoning output"},
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 123},
            }

    client = OpenAICompatibleLLM(
        base_url="http://model.test/v1",
        model_name="model-a",
        max_tokens=123,
        session=FakeSession(ReasoningResponse()),
    )

    result = client.generate("question")

    assert result.content == "reasoning output"
    assert result.token_capped is True
