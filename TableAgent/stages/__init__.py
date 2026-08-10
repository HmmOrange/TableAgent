"""Explicit pipeline stage boundaries.

Import individual stage packages so their cross-cutting dependencies stay lazy.
"""

from importlib import import_module


_EXPORTS = {
    "QAInput": "TableAgent.stages.qa",
    "QAOutput": "TableAgent.stages.qa",
    "QAStage": "TableAgent.stages.qa",
    "RetrievalInput": "TableAgent.stages.retrieval",
    "RetrievalOutput": "TableAgent.stages.retrieval",
    "RetrievalStage": "TableAgent.stages.retrieval",
    "StructureInput": "TableAgent.stages.structure",
    "StructureOutput": "TableAgent.stages.structure",
    "StructureStage": "TableAgent.stages.structure",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


__all__ = list(_EXPORTS)
