"""QA stage: planning, verified execution, and answer synthesis."""

from importlib import import_module

_EXPORTS = {
    "BaseAction": (".actions.base_action", "BaseAction"),
    "BaseCodeExecutionAction": (".actions.base_action", "BaseCodeExecutionAction"),
    "BaseCodeGenerationAction": (".actions.base_action", "BaseCodeGenerationAction"),
    "BasePlanAction": (".actions.base_action", "BasePlanAction"),
    "BaseReActAgent": (".agents.base_agent", "BaseReActAgent"),
    "BaseReviewAction": (".actions.base_action", "BaseReviewAction"),
    "AgentOutput": (".models.results", "AgentOutput"),
    "CodeExecutionRequest": (".actions.base_action", "CodeExecutionRequest"),
    "CodeExecutionResult": (".actions.base_action", "CodeExecutionResult"),
    "CodeGenerationRequest": (".actions.base_action", "CodeGenerationRequest"),
    "CodeGenerationResult": (".actions.base_action", "CodeGenerationResult"),
    "ExecuteNotebookCodeAction": (".actions.execute_notebook", "ExecuteNotebookCodeAction"),
    "ExperiencePool": (".experience", "ExperiencePool"),
    "ExperienceRecord": (".experience", "ExperienceRecord"),
    "LLMCodeGenerationAction": (".actions.llm_code_generation", "LLMCodeGenerationAction"),
    "PlanGenerationRequest": (".actions.base_action", "PlanGenerationRequest"),
    "PlanGenerationResult": (".actions.base_action", "PlanGenerationResult"),
    "QAInput": (".contracts", "QAInput"),
    "QAOutput": (".contracts", "QAOutput"),
    "QAResult": (".models.results", "QAResult"),
    "QAStage": (".stage", "QAStage"),
    "SourceQAPipeline": (".source_pipeline", "SourceQAPipeline"),
    "SubTask": (".models.subtask", "SubTask"),
    "VerifiedQAPipeline": (".pipeline", "VerifiedQAPipeline"),
    "ReviewRequest": (".actions.base_action", "ReviewRequest"),
    "ReviewResult": (".actions.base_action", "ReviewResult"),
    "ReviewSubtaskAction": (".actions.review", "ReviewSubtaskAction"),
    "TableQAAgent": (".agents.react_agent", "TableQAAgent"),
    "TableQAPlanner": (".agents.planner", "TableQAPlanner"),
    "TableQARunner": (".runner", "TableQARunner"),
    "TableQASynthesisAgent": (".agents.synthesis_agent", "TableQASynthesisAgent"),
    "WriteQAPlanAction": (".actions.write_plan", "WriteQAPlanAction"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    return getattr(import_module(f"{__name__}{module_name}"), attribute)


__all__ = list(_EXPORTS)
