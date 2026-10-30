"""QA stage: answer a question by exploring the workbook (see orchestrator.py for the loop)."""

from importlib import import_module

_EXPORTS = {
    "AnswerResult": (".orchestrator", "AnswerResult"),
    "DirectAnswerer": (".utils.direct_answer", "DirectAnswerer"),
    "QAInput": (".stage", "QAInput"),
    "QAOutput": (".stage", "QAOutput"),
    "QAStage": (".stage", "QAStage"),
    "SheetTools": (".tools.sheet", "SheetTools"),
    "SourceQAPipeline": (".utils.source_pipeline", "SourceQAPipeline"),
    "WorkbookQA": (".workbook_qa", "WorkbookQA"),
    "answer_question": (".orchestrator", "answer_question"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    return getattr(import_module(f"{__name__}{module_name}"), attribute)


__all__ = list(_EXPORTS)
