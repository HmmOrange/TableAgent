import pytest

from configs import load_config, resolve_llm_config, resolve_vlm_config


def test_config_example_loads_modular_files():
    config = load_config("configs/config.example.yaml")

    assert "datasets" in config
    assert "gpt_oss" in config["models"]
    assert "gemma4_vlm" in config["vlm_models"]
    # Check that resolving layout VLM for table_agent yields gemma4_vlm via top-level fallback
    provider_name, _ = resolve_vlm_config(config, "table_agent")
    assert provider_name == "gemma4_vlm"


def test_legacy_root_config_path_falls_back_to_configs():
    config = load_config("config.yaml")

    # Check that resolving layout VLM for table_agent yields qwen_local_vlm via top-level fallback
    provider_name, _ = resolve_vlm_config(config, "table_agent")
    assert provider_name == "qwen_local_vlm"


def test_resolve_vlm_provider_from_vlm_models(monkeypatch):
    monkeypatch.setenv("GEMMA4_VLM_BASE_URL", "https://vlm.local/v1")
    config = load_config("configs/config.example.yaml")

    provider_name, vlm_config = resolve_vlm_config(config, "table_agent")

    assert provider_name == "gemma4_vlm"
    assert vlm_config["base_url"] == "https://vlm.local/v1"


def test_resolve_llm_provider_from_pipeline_env(monkeypatch):
    monkeypatch.setenv("GPT_OSS_BASE_URL", "https://llm.local/v1")
    config = load_config("configs/config.example.yaml")

    provider_name, llm_config = resolve_llm_config(config, "table_agent")

    assert provider_name == "gpt_oss"
    assert llm_config["base_url"] == "https://llm.local/v1"


def test_resolve_vlm_config_direct_lookup():
    config = load_config("configs/config.example.yaml")
    provider_name, vlm_config = resolve_vlm_config(config, "gemma4_vlm")
    assert provider_name == "gemma4_vlm"


def test_resolve_llm_fallback_to_top_level():
    config = {
        "llm": {"provider": "gemini_flash"},
        "models": {
            "gemini_flash": {"provider": "gemini", "model_name": "gemini-2.5-flash"}
        }
    }
    provider_name, llm_config = resolve_llm_config(config, "some_pipeline")
    assert provider_name == "gemini_flash"
    assert llm_config["model_name"] == "gemini-2.5-flash"


def test_resolve_llm_vlm_pipeline_fallback_to_vlm_provider():
    config = {
        "vlm": {"provider": "qwen_local_vlm"},
        "llm": {"provider": "gemini_flash"},
        "models": {
            "gemini_flash": {"provider": "gemini", "model_name": "gemini-2.5-flash"}
        },
        "vlm_models": {
            "qwen_local_vlm": {
                "provider": "openai_compatible",
                "base_url": "http://localhost:8010/v1",
                "model": "Qwen/Qwen3.6-35B-A3B-FP8",
                "api_key": "EMPTY",
                "temperature": 0.0,
                "max_tokens": 2048,
                "timeout_seconds": 180,
                "max_retries": 2,
                "retry_delay_seconds": 2
            }
        }
    }
    provider_name, llm_config = resolve_llm_config(config, "table2img_vlm")
    assert provider_name == "qwen_local_vlm"
    assert llm_config["base_url"] == "http://localhost:8010/v1"
    assert llm_config["model"] == "Qwen/Qwen3.6-35B-A3B-FP8"


def test_resolve_vlm_fallback_to_top_level():
    config = {
        "vlm": {"provider": "gemma4_vlm"},
        "vlm_models": {
            "gemma4_vlm": {"provider": "openai", "model": "llm"}
        }
    }
    provider_name, vlm_config = resolve_vlm_config(config, "some_pipeline")
    assert provider_name == "gemma4_vlm"
    assert vlm_config["model"] == "llm"


def test_resolve_llm_and_vlm_direct_lookup():
    config = {
        "models": {
            "gemini_flash": {"provider": "gemini", "model_name": "gemini-2.5-flash"}
        },
        "vlm_models": {
            "gemma4_vlm": {"provider": "openai", "model": "llm"}
        }
    }
    # Direct lookup on LLM
    provider_name, llm_config = resolve_llm_config(config, "gemini_flash")
    assert provider_name == "gemini_flash"
    
    # Direct lookup on VLM
    provider_name, vlm_config = resolve_vlm_config(config, "gemma4_vlm")
    assert provider_name == "gemma4_vlm"


def test_resolve_unknown_provider_raises_value_error():
    config = {
        "llm": {"provider": "non_existent"},
        "vlm": {"provider": "non_existent"},
        "models": {},
        "vlm_models": {}
    }
    with pytest.raises(ValueError, match="is unknown or not defined"):
        resolve_llm_config(config, "some_pipeline")
        
    with pytest.raises(ValueError, match="is unknown or not defined"):
        resolve_vlm_config(config, "some_pipeline")


def test_temperature_validation():
    from configs.models_config import normalize_model_config

    normalize_model_config({"temperature": 0.5})
    normalize_model_config({"temperature": 0.0})
    normalize_model_config({"temperature": 2.0})

    with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
        normalize_model_config({"temperature": 2.1})
    with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
        normalize_model_config({"temperature": -0.1})
    with pytest.raises(ValueError, match="Invalid temperature value"):
        normalize_model_config({"temperature": "invalid"})


def test_provider_validation():
    from configs.models_config import normalize_model_config

    normalize_model_config({"provider": "gemini"})
    normalize_model_config({"provider": "openai"})
    normalize_model_config({"provider": "openai_compatible"})
    normalize_model_config({"provider": "gpt_oss"})
    normalize_model_config({"provider": "openrouter"})

    with pytest.raises(ValueError, match="Unsupported provider"):
        normalize_model_config({"provider": "unsupported_llm"})


def test_duplicate_model_definition():
    from configs.models_config import available_models

    bad_config = {
        "models": {"gpt_oss": {"provider": "openai"}},
        "vlm_models": {"gpt_oss": {"provider": "openai"}},
    }
    with pytest.raises(ValueError, match="Duplicate model definition 'gpt_oss'"):
        available_models(bad_config)


def test_non_dict_group_definition():
    from configs.models_config import available_models

    bad_config = {
        "models": ["not-a-dict"],
    }
    with pytest.raises(ValueError, match="must be a mapping"):
        available_models(bad_config)


def test_non_dict_model_definition():
    from configs.models_config import available_models

    bad_config = {
        "models": {"gpt_oss": "not-a-dict"},
    }
    with pytest.raises(ValueError, match="Model definition 'gpt_oss'.*must be a mapping"):
        available_models(bad_config)


def test_resolve_env_field_empty_string_falls_back(monkeypatch):
    monkeypatch.setenv("TEST_EMPTY_ENV", "")

    from configs.models_config import _resolve_env_field

    res = _resolve_env_field(
        {"api_key": "fallback-key", "api_key_env": "TEST_EMPTY_ENV"},
        "api_key",
        "api_key_env",
    )
    assert res == "fallback-key"
