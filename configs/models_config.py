from __future__ import annotations

import os
from typing import Any


MODEL_GROUPS = ("models", "vlm_models", "llm_providers")
VALID_PROVIDERS = {"gemini", "openai", "openai_compatible", "gpt_oss", "openrouter"}


def available_models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for group in MODEL_GROUPS:
        group_config = config.get(group)
        if group_config is not None:
            if not isinstance(group_config, dict):
                raise ValueError(
                    f"Configuration group '{group}' must be a mapping, "
                    f"got {type(group_config).__name__}"
                )
            for name, model_config in group_config.items():
                if not isinstance(model_config, dict):
                    raise ValueError(
                        f"Model definition '{name}' in group '{group}' must be a mapping, "
                        f"got {type(model_config).__name__}"
                    )
                if name in models:
                    raise ValueError(f"Duplicate model definition '{name}' found across model groups")
                models[name] = dict(model_config)
    return models


def resolve_model_config(config: dict[str, Any], name_or_pipeline: str) -> tuple[str, dict[str, Any]]:
    models = available_models(config)
    if name_or_pipeline in models:
        return name_or_pipeline, normalize_model_config(models[name_or_pipeline])

    if name_or_pipeline not in config:
        raise ValueError(f"Pipeline or model provider '{name_or_pipeline}' not found in config")

    pipeline_config = config[name_or_pipeline]
    if not isinstance(pipeline_config, dict) or "llm_provider" not in pipeline_config:
        raise ValueError(f"Pipeline '{name_or_pipeline}' does not have 'llm_provider' configured")

    provider_name = str(pipeline_config["llm_provider"])
    if provider_name not in models:
        raise ValueError(f"LLM provider '{provider_name}' not found in model config")
    return provider_name, normalize_model_config(models[provider_name])


def normalize_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(model_config)

    temperature = config.get("temperature")
    if temperature is not None:
        try:
            temp_val = float(temperature)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid temperature value: {temperature}")
        if not (0.0 <= temp_val <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")

    provider = config.get("provider")
    if provider is not None:
        if str(provider).lower() not in VALID_PROVIDERS:
            valid = ", ".join(sorted(VALID_PROVIDERS))
            raise ValueError(f"Unsupported provider '{provider}'. Valid providers are: {valid}")

    config["api_key"] = _resolve_env_field(config, "api_key", "api_key_env")
    endpoint = _resolve_env_field(config, "endpoint", "endpoint_env")
    base_url = _resolve_env_field(config, "base_url", "base_url_env")
    if endpoint is not None:
        config["endpoint"] = endpoint
    if base_url is not None:
        config["base_url"] = base_url
    return config


def _resolve_env_field(config: dict[str, Any], value_key: str, env_key: str) -> Any:
    value = config.get(value_key)
    env_name = config.get(env_key)
    if env_name:
        env_value = os.environ.get(str(env_name))
        if env_value:
            return env_value
    return value
