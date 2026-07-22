from __future__ import annotations

from service import server


def test_server_parser_accepts_model_profile_overrides():
    args = server.build_parser().parse_args(
        [
            "--config",
            "private.yaml",
            "--llm",
            "alternate_answer",
            "--vlm",
            "alternate_layout",
        ]
    )

    assert args.config == "private.yaml"
    assert args.llm == "alternate_answer"
    assert args.vlm == "alternate_layout"


def test_server_parser_defaults_model_profiles_to_config():
    args = server.build_parser().parse_args([])

    assert args.llm is None
    assert args.vlm is None


def test_server_main_passes_model_profile_overrides(monkeypatch):
    captured = {}
    fake_service = object()
    fake_app = object()

    class FakeTableAgentService:
        @staticmethod
        def from_config(path, **kwargs):
            captured["config"] = path
            captured.update(kwargs)
            return fake_service

    monkeypatch.setattr(server, "TableAgentService", FakeTableAgentService)
    monkeypatch.setattr(server, "create_app", lambda service: fake_app if service is fake_service else None)
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs))

    result = server.main(
        [
            "--config",
            "private.yaml",
            "--llm",
            "alternate_answer",
            "--vlm",
            "alternate_layout",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
        ]
    )

    assert result == 0
    assert captured == {
        "config": "private.yaml",
        "llm_profile": "alternate_answer",
        "vlm_profile": "alternate_layout",
        "app": fake_app,
        "host": "0.0.0.0",
        "port": 9000,
        "log_level": "info",
    }
