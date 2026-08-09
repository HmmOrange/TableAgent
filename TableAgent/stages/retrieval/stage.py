from __future__ import annotations

from .contracts import RetrievalInput, RetrievalOutput


class RetrievalStage:
    """Select prepared corpus artifacts; candidate embeddings are never generated here."""

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, stage_input: RetrievalInput) -> RetrievalOutput:
        if stage_input.sample is None:
            raise ValueError("RetrievalInput.sample is required for prepared-source retrieval")
        candidate = (
            self.retriever.select_perfect(stage_input.sample)
            if stage_input.perfect
            else self.retriever.select(
                stage_input.sample,
                stage_input.responses,
                stage_input.fit_context,
            )
        )
        return RetrievalOutput(candidate=candidate)

    def run_indexed(self, stage_input: RetrievalInput) -> RetrievalOutput:
        candidate = self.retriever.select_indexed(
            question=stage_input.question,
            artifacts=stage_input.artifacts,
            workbook_paths=stage_input.workbook_paths,
            responses=stage_input.responses,
            fit_context=stage_input.fit_context,
        )
        return RetrievalOutput(candidate=candidate)
