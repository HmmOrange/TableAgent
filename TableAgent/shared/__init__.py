"""Cross-stage helpers shared by the pipeline stages."""

from importlib import import_module

_EXPORTS = {
    "AgentMemory": (".agents", "AgentMemory"),
    "AgentMessage": (".agents", "AgentMessage"),
    "BaseTableAgent": (".agents", "BaseTableAgent"),
    "prepared_verification": (".artifacts", "prepared_verification"),
    "read_image_tiles": (".artifacts", "read_image_tiles"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    return getattr(import_module(f"{__name__}{module_name}"), attribute)


__all__ = list(_EXPORTS)
