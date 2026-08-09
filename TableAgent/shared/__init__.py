"""Cross-stage helpers shared by the pipeline stages."""

from importlib import import_module

_EXPORTS = {
    "AgentMemory": (".agents", "AgentMemory"),
    "AgentMessage": (".agents", "AgentMessage"),
    "BaseTableAgent": (".agents", "BaseTableAgent"),
    "PromptBuilder": (".prompting", "PromptBuilder"),
    "prepared_verification": (".artifacts", "prepared_verification"),
    "SourceCandidate": (".pipeline", "SourceCandidate"),
    "display_path": (".pipeline", "display_path"),
    "has_workbook_sources": (".pipeline", "has_workbook_sources"),
    "read_image_tiles": (".pipeline", "read_image_tiles"),
    "safe_name": (".pipeline", "safe_name"),
    "token_usage": (".pipeline", "token_usage"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    return getattr(import_module(f"{__name__}{module_name}"), attribute)


__all__ = list(_EXPORTS)
