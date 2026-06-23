from TableAgent.perception.structure import (
    _extract_yaml_text,
    _is_valid_structure,
    _parse_yaml_mapping,
    extract_strict_structure,
)

def __getattr__(name: str):
    if name == "extract_relations":
        from TableAgent.perception.relations import extract_relations
        return extract_relations
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "_extract_yaml_text",
    "_is_valid_structure",
    "_parse_yaml_mapping",
    "extract_strict_structure",
    "extract_relations",
]
