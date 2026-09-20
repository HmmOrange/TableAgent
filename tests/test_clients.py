from __future__ import annotations

from pathlib import Path

from service.clients import OpenAICompatibleLLM, create_model_client


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
    image_result = client.generate_with_image("inspect", image_path, system_prompt="layout")

    assert text_result.content == "done"
    assert text_result.prompt_tokens == 7
    assert image_result.completion_tokens == 3
    assert session.calls[0][0] == "http://model.test/v1/chat/completions"
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer secret"
    assert session.calls[0][1]["json"]["max_tokens"] == 123
    assert session.calls[0][1]["json"]["chat_template_kwargs"] == {"enable_thinking": False}
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


class SequencedSession:
    """Answers each POST from a queue so a rejected request can be followed by a retry."""

    def __init__(self, responses):
        self.calls = []
        self.responses = list(responses)

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        return None


class RejectingResponse:
    status_code = 400

    def raise_for_status(self):
        raise AssertionError("a rejected constrained request must not be raised to the caller")

    def json(self):
        return {"error": "guided_json is not supported"}


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def test_a_schema_is_sent_the_openai_standard_way_first():
    session = FakeSession()
    client = OpenAICompatibleLLM(
        base_url="http://model.test/v1", model_name="model-a", session=session
    )

    client.generate("question", response_schema=SCHEMA)

    body = session.calls[0][1]["json"]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "guided_json" not in body


def test_a_plain_call_carries_no_constraint():
    session = FakeSession()
    client = OpenAICompatibleLLM(
        base_url="http://model.test/v1", model_name="model-a", session=session
    )

    client.generate("question")

    body = session.calls[0][1]["json"]
    assert "response_format" not in body and "guided_json" not in body


def test_the_schema_steps_down_to_guided_json_then_off():
    """Servers disagree on the spelling; each rejection costs one call, not one per call."""
    session = SequencedSession(
        [RejectingResponse(), RejectingResponse(), FakeResponse(), FakeResponse()]
    )
    client = OpenAICompatibleLLM(
        base_url="http://model.test/v1", model_name="model-a", session=session
    )

    result = client.generate("question", response_schema=SCHEMA)

    assert result.content == "done"
    assert "response_format" in session.calls[0][1]["json"]
    assert "guided_json" in session.calls[1][1]["json"]
    third = session.calls[2][1]["json"]
    assert "response_format" not in third and "guided_json" not in third

    # The run records that it lost its constraint, instead of degrading silently.
    assert client.structured_output_mode == "off"
    assert client.structured_output_downgrades == [
        "response_format rejected -> guided_json",
        "guided_json rejected -> off",
    ]

    # And the next call does not pay for either rejection again.
    client.generate("another", response_schema=SCHEMA)
    assert len(session.calls) == 4


def test_a_server_that_only_understands_guided_json_settles_there():
    session = SequencedSession([RejectingResponse(), FakeResponse(), FakeResponse()])
    client = OpenAICompatibleLLM(
        base_url="http://model.test/v1", model_name="model-a", session=session
    )

    client.generate("question", response_schema=SCHEMA)
    assert client.structured_output_mode == "guided_json"

    client.generate("another", response_schema=SCHEMA)
    assert "guided_json" in session.calls[2][1]["json"]
    assert len(session.calls) == 3


def test_an_explicit_extra_body_constraint_is_left_alone():
    session = FakeSession()
    custom = {"type": "object"}
    client = OpenAICompatibleLLM(
        base_url="http://model.test/v1",
        model_name="model-a",
        extra_body={"response_format": custom},
        session=session,
    )

    client.generate("question", response_schema=SCHEMA)

    assert session.calls[0][1]["json"]["response_format"] == custom
