from __future__ import annotations

from typing import Any

from configs.models_config import available_models, normalize_model_config


def resolve_embedding_config(
    config: dict[str, Any],
    pipeline_name: str,
    provider_key: str = "embedding_provider",
) -> tuple[str, dict[str, Any]]:
    models = available_models(config)
    
    # a) if the argument is a direct model key, use it
    if pipeline_name in models:
        return pipeline_name, normalize_model_config(models[pipeline_name])
    
    # b) pipeline-specific provider_key remains an override for compatibility
    provider_name = None
    if pipeline_name in config:
        pipeline_config = config[pipeline_name]
        if isinstance(pipeline_config, dict):
            provider_name = (
                pipeline_config.get(provider_key) or
                pipeline_config.get("embedding_provider")
            )
            
    # c) otherwise fall back to top-level embedding.provider
    if not provider_name:
        embedding_section = config.get("embedding")
        if isinstance(embedding_section, dict):
            provider_name = embedding_section.get("provider")
            
    # d) raise a clear ValueError if no selection exists or the provider key is unknown
    if not provider_name:
        raise ValueError(
            f"No embedding provider selection exists for '{pipeline_name}' and "
            "no top-level default 'embedding.provider' is configured."
        )
        
    provider_name_str = str(provider_name)
    if provider_name_str not in models:
        raise ValueError(f"Embedding provider '{provider_name_str}' is unknown or not defined in model configurations.")
        
    return provider_name_str, normalize_model_config(models[provider_name_str])
