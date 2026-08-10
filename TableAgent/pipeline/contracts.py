"""Contracts shared by the pipeline orchestrator and its composed runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, TYPE_CHECKING

from TableAgent.pipeline.sample import EvalSample

if TYPE_CHECKING:
    from TableAgent.stages.qa.contracts import QAInput, QAOutput
    from TableAgent.stages.retrieval.contracts import RetrievalInput, RetrievalOutput
    from TableAgent.stages.structure.contracts import StructureInput, StructureOutput


@dataclass
class PipelineOutput:
    """Stable result returned by a complete TableAgent run."""

    sample_id: str
    structured_table: Any | None = None
    predicted_answer: str = ""
    latency: float = 0.0
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"prompt": 0, "completion": 0}
    )
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureStageContract(Protocol):
    def run(self, stage_input: StructureInput) -> StructureOutput:
        ...


class RetrievalStageContract(Protocol):
    def run(self, stage_input: RetrievalInput) -> RetrievalOutput:
        ...

    def run_indexed(self, stage_input: RetrievalInput) -> RetrievalOutput:
        ...


class QAStageContract(Protocol):
    def run(self, stage_input: QAInput) -> QAOutput:
        ...


class PipelineRuntimeContract(Protocol):
    """Dependencies exposed to one composed pipeline component.

    Components receive this narrow runtime object instead of inheriting all
    behavior from ``TableAgentPipeline``. The protocol documents the intentional
    compatibility surface while allowing the orchestrator to remain the owner of
    mutable run state.
    """

    settings: Any
    llm: Any
    layout_vlm: Any
    qa_agent: Any
    table_retriever: Any
    prompts: Any
    source_preparer: Any
    source_retriever: Any
    structure_cache: Any
    layout_workflow: Any
    artifact_dir: Path
    _artifact_dir: Path
    structure_stage: StructureStageContract
    retrieval_stage: RetrievalStageContract
    qa_stage: QAStageContract
    _verified_samples: dict[str, Any]
    _prepared_source_samples: set[str]
    _progress_callback: Callable[[str], None] | None

    def _progress(self, stage: str, **fields: Any) -> None:
        ...

    def _fit_context(self, text: str) -> str:
        ...

    def _qa_sample_dir(self, sample: EvalSample) -> Path:
        ...

    def _run_verified_qa(self, **kwargs: Any) -> Any:
        ...

    def _run_prepared_source(self, *args: Any, **kwargs: Any) -> PipelineOutput:
        ...

    def _run_cached_qa(self, *args: Any, **kwargs: Any) -> PipelineOutput:
        ...


@dataclass(frozen=True)
class PipelineStages:
    """Concrete stage bundle assembled by the pipeline composition root."""

    structure: StructureStageContract
    retrieval: RetrievalStageContract
    qa: QAStageContract
