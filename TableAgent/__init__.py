from importlib import import_module

__all__ = [
    "TableAgentPipeline",
    "TableAgentConfig",
    "TableAgentSettings",
    "answer_question",
]


def __getattr__(name: str):
    modules = {
        "TableAgentPipeline": "TableAgent.pipeline",
        "TableAgentConfig": "TableAgent.configs",
        "TableAgentSettings": "TableAgent.configs",
        "answer_question": "TableAgent.stages.qa",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
