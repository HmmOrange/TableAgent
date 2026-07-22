from __future__ import annotations

from typing import Any

from configs.models_config import available_models, normalize_model_config


def resolve_vlm_config(
    config: dict[str, Any],
    pipeline_name: str,
    provider_key: str = "layout_vlm_provider",
) -> tuple[str, dict[str, Any]]:
    models = available_models(config)
    
    # a) if the argument is a direct model key, use it
    if pipeline_name in models:
        return pipeline_name, normalize_model_config(models[pipeline_name])
        
    # b) pipeline-specific provider_key/layout_vlm_provider/vlm_provider remains an override for compatibility
    provider_name = None
    if pipeline_name in config:
        pipeline_config = config[pipeline_name]
        if isinstance(pipeline_config, dict):
            provider_name = (
                pipeline_config.get(provider_key) or
                pipeline_config.get("layout_vlm_provider") or
                pipeline_config.get("vlm_provider")
            )
            
    # c) otherwise fall back to top-level vlm.provider
    if not provider_name:
        vlm_section = config.get("vlm")
        if isinstance(vlm_section, dict):
            provider_name = vlm_section.get("provider")
            
    # d) raise a clear ValueError if no selection exists or the provider key is unknown
    if not provider_name:
        raise ValueError(
            f"No VLM provider selection exists for '{pipeline_name}' and "
            "no top-level default 'vlm.provider' is configured."
        )
        
    provider_name_str = str(provider_name)
    if provider_name_str not in models:
        raise ValueError(f"VLM provider '{provider_name_str}' is unknown or not defined in model configurations.")
        
    return provider_name_str, normalize_model_config(models[provider_name_str])

