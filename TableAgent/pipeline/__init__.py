from importlib import import_module

from TableAgent.pipeline.contracts import PipelineOutput, PipelineStages
from TableAgent.pipeline.sample import EvalSample

__all__ = ["EvalSample", "PipelineOutput", "PipelineStages", "TableAgentPipeline"]


def __getattr__(name: str):
    if name != "TableAgentPipeline":
        raise AttributeError(name)
    value = import_module("TableAgent.pipeline.table_agent_pipeline").TableAgentPipeline
    globals()[name] = value
    return value
