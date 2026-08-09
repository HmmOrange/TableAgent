import importlib

def __getattr__(name: str):
    if name == "extract_relations":
        mod = importlib.import_module("TableAgent.perception.relations.extract")
        return getattr(mod, "extract_relations")
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = ["extract_relations"]

