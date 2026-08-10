from __future__ import annotations

from TableAgent.schema import EvalSample
from TableAgent.pipeline.component import RuntimeComponent
from TableAgent.pipeline.contracts import PipelineRuntimeContract


class RetrievalPipeline(RuntimeComponent):
    """Expose retrieval-stage orchestration through an explicit runtime."""

    def __init__(self, runtime: PipelineRuntimeContract):
        super().__init__(runtime)

    def filter_samples(self, samples: list[EvalSample]) -> list[EvalSample]:
        """Skip samples whose perfect-retrieval source is unavailable."""
        if not self.settings.perfect_retrieval:
            return samples

        filtered = []
        for sample in samples:
            try:
                self.source_retriever.select_perfect(sample)
            except RuntimeError as exc:
                if "Perfect retrieval excludes sheet" in str(exc):
                    continue
                raise
            filtered.append(sample)
        return filtered
