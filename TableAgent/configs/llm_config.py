from __future__ import annotations

from typing import Any

from TableAgent.configs.models_config import available_models, normalize_model_config


def resolve_llm_config(config: dict[str, Any], name_or_pipeline: str) -> tuple[str, dict[str, Any]]:
    models = available_models(config)
    
    # a) if the argument is a direct model key, use it
    if name_or_pipeline in models:
        return name_or_pipeline, normalize_model_config(models[name_or_pipeline])
    
    # b) if the named pipeline has llm_provider, that remains a pipeline-specific override for compatibility
    provider_name = None
    if name_or_pipeline in config:
        pipeline_config = config[name_or_pipeline]
        if isinstance(pipeline_config, dict):
            provider_name = pipeline_config.get("llm_provider")
            
    # c) otherwise fall back to top-level llm.provider (or vlm.provider for VLM pipelines)
    if not provider_name:
        if name_or_pipeline.endswith("_vlm"):
            vlm_section = config.get("vlm")
            if isinstance(vlm_section, dict):
                provider_name = vlm_section.get("provider")
        else:
            llm_section = config.get("llm")
            if isinstance(llm_section, dict):
                provider_name = llm_section.get("provider")
            
    # d) raise a clear ValueError if no selection exists or the provider key is unknown
    if not provider_name:
        if name_or_pipeline.endswith("_vlm"):
            raise ValueError(
                f"No VLM provider selection exists for VLM pipeline '{name_or_pipeline}' and "
                "no top-level default 'vlm.provider' is configured."
            )
        else:
            raise ValueError(
                f"No LLM provider selection exists for '{name_or_pipeline}' and "
                "no top-level default 'llm.provider' is configured."
            )
    
    provider_name_str = str(provider_name)
    if provider_name_str not in models:
        raise ValueError(f"LLM provider '{provider_name_str}' is unknown or not defined in model configurations.")
        
    return provider_name_str, normalize_model_config(models[provider_name_str])
