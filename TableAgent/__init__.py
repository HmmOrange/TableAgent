from importlib import import_module

__all__ = [
    "TableAgentPipeline",
    "TableAgentConfig",
    "TableAgentSettings",
    "QAEnvironment",
    "TableQARunner",
]


def __getattr__(name: str):
    modules = {
        "TableAgentPipeline": "TableAgent.pipeline",
        "TableAgentConfig": "TableAgent.configs",
        "TableAgentSettings": "TableAgent.configs",
        "QAEnvironment": "TableAgent.stages.qa.environment.qa_env",
        "TableQARunner": "TableAgent.stages.qa.runner",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name]), name)
    globals()[name] = value
    return value
