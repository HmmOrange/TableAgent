from importlib import import_module

__all__ = ["LayoutAgent", "LayoutResult", "LayoutWorkflowResult", "TableLayoutWorkflow"]


def __getattr__(name: str):
    modules = {
        "LayoutAgent": "TableAgent.stages.structure.layout.agent",
        "LayoutResult": "TableAgent.stages.structure.layout.agent",
        "LayoutWorkflowResult": "TableAgent.stages.structure.layout.workflow",
        "TableLayoutWorkflow": "TableAgent.stages.structure.layout.workflow",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
