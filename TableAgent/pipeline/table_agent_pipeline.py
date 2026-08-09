from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from TableAgent.stages.qa.prompts.answer import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
)
from TableAgent.stages.retrieval.retrieval_prompts import (
    RERANKER_SYSTEM_PROMPT,
    RERANKER_USER_PROMPT_TEMPLATE,
)

from TableAgent.configs import TableAgentConfig
from TableAgent.configs.models_config import available_models
from TableAgent.llm import BaseLLM
from TableAgent.pipeline.base import BasePipeline
from TableAgent.stages.qa.agents.answer_agent import QAAgent
from TableAgent.stages.qa.runner import TableQARunner
from TableAgent.run_logging import Logger
from TableAgent.schema import EvalSample
from TableAgent.stages.structure.layout.agent import LayoutAgent
from TableAgent.shared.pipeline import has_workbook_sources
from TableAgent.shared.prompting import PromptBuilder
from TableAgent.stages.retrieval import SourceRetriever
from TableAgent.stages.retrieval.embeddings import (
    MockEmbeddingModel,
    OpenAICompatibleEmbeddingClient,
)
from TableAgent.stages.qa.pipeline import PipelineQAMixin
from TableAgent.pipeline.pipeline_run import PipelineRunMixin
from TableAgent.stages.qa.source_pipeline import PipelineSourceQAMixin
from TableAgent.stages.structure.source_preparer import SourcePreparer
from TableAgent.stages.structure.layout.workflow import TableLayoutWorkflow
from TableAgent.stages.structure.cache import StructureCache, StructureCacheRecord
from TableAgent.rendering.workbook import WorkbookRenderer
from TableAgent.stages.structure.verification import DeterministicVerifier
from TableAgent.stages.qa import QAStage
from TableAgent.stages.retrieval import RetrievalStage
from TableAgent.stages.retrieval.pipeline import RetrievalPipelineMixin
from TableAgent.stages.structure import StructureStage
from TableAgent.stages.structure.pipeline import StructurePipelineMixin

if TYPE_CHECKING:
    from TableAgent.stages.retrieval import TableRetrieverContract

logger = Logger(__name__)


class TableAgentPipeline(
    StructurePipelineMixin,
    RetrievalPipelineMixin,
    PipelineRunMixin,
    PipelineSourceQAMixin,
    PipelineQAMixin,
    BasePipeline,
):
    name = "table_agent"
    prepare_samples_before_run = True
    answer_system_prompt = ANSWER_SYSTEM_PROMPT
    answer_user_prompt_template = ANSWER_USER_PROMPT_TEMPLATE
    reranker_system_prompt = RERANKER_SYSTEM_PROMPT
    reranker_user_prompt_template = RERANKER_USER_PROMPT_TEMPLATE

    def __init__(
        self,
        llm_client: BaseLLM | None,
        layout_vlm_client: BaseLLM | None,
        config: dict[str, Any] | None = None,
        table_retriever: TableRetrieverContract | None = None,
        embedding_client: Any | None = None,
    ):
        self.llm = llm_client
        self.layout_vlm = layout_vlm_client
        self.settings = TableAgentConfig.from_config(config)
        self._artifact_dir = self.settings.artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.settings.phase in {"qa", "all"} and self.llm is None:
            raise ValueError(f"TableAgent phase '{self.settings.phase}' requires an answer LLM client")
        if self.settings.phase in {"structure", "all"} and self.layout_vlm is None:
            raise ValueError(f"TableAgent phase '{self.settings.phase}' requires a layout VLM client")
        self.prompts = PromptBuilder(self.settings, self)
        self.workbook_renderer = WorkbookRenderer(self.settings, logger)
        self.layout_agent = LayoutAgent(self.layout_vlm) if self.layout_vlm is not None else None
        self.verifier = DeterministicVerifier(
            data_only=self.settings.structure_data_only,
        )
        self.qa_agent = QAAgent(self.llm, self.answer_system_prompt) if self.llm is not None else None
        self.table_retriever = table_retriever
        self.layout_workflow = (
            TableLayoutWorkflow(
                self.settings,
                self.workbook_renderer,
                self.layout_agent,
                self.verifier,
                progress_callback=self._progress,
            )
            if self.layout_agent is not None
            else None
        )
        configured_models = available_models(config or {})
        if (
            embedding_client is None
            and (
                self.settings.prepare_retrieval_embeddings
                or (
                    self.settings.phase == "all"
                    and self.settings.routing.retrieval.use_embeddings
                )
            )
            and self.settings.retrieval_embedding_provider == "mock"
        ):
            embedding_client = MockEmbeddingModel()
        if (
            embedding_client is None
            and self.settings.retrieval_embedding_provider not in {None, "mock"}
            and self.settings.retrieval_embedding_provider in configured_models
        ):
            embedding_client = OpenAICompatibleEmbeddingClient.from_config(
                config or {},
                self.settings.retrieval_embedding_provider,
            )
        embedding_model = (
            "mock-hash-embedding"
            if isinstance(embedding_client, MockEmbeddingModel)
            else str(getattr(embedding_client, "model", "") or "")
        )
        if (
            self.settings.embed_retrieval_cards
            and self.settings.retrieval_embedding_provider == "mock"
        ):
            raise ValueError(
                "Embedding export requires a configured real embedding provider and model"
            )
        if (
            self.settings.embed_retrieval_cards
            or self.settings.prepare_retrieval_embeddings
        ) and (
            embedding_client is None or not embedding_model
        ):
            raise ValueError(
                "Retrieval embedding preparation requires a configured real embedding provider and model"
            )
        prepare_corpus_embeddings = bool(
            self.settings.embed_retrieval_cards
            or self.settings.prepare_retrieval_embeddings
            or (
                self.settings.phase == "all"
                and self.settings.routing.retrieval.use_embeddings
                and embedding_client is not None
                and embedding_model
            )
        )
        self.source_preparer = SourcePreparer(
            self.settings,
            self._analyze_source_sheet,
            progress_callback=self._progress,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
            include_embeddings=prepare_corpus_embeddings,
        )
        self.source_retriever = SourceRetriever(
            self.settings,
            self.llm,
            self,
            self.prompts,
            embedding_client=embedding_client,
        )
        self.retrieval_stage = RetrievalStage(self.source_retriever)
        self.structure_cache = StructureCache(
            self.settings,
            self.layout_workflow,
            self._metadata_for_workbook_sheet,
        )
        self._verified_samples: dict[str, StructureCacheRecord] = {}
        self._prepared_source_samples: set[str] = set()
        self._progress_callback: Callable[[str], None] | None = None
        self.structure_stage = StructureStage(self._verify_samples_impl)
        self.qa_stage = QAStage(lambda **kwargs: self._run_verified_qa(**kwargs))
        self._apply_generation_cap()

    def prepare_samples(self, samples: list[EvalSample], logger: Any | None = None) -> None:
        if self.settings.phase == "qa":
            missing = []
            for sample in samples:
                if (
                    self.settings.should_retrieve(sample)
                    and has_workbook_sources(sample)
                    and (
                        self.source_retriever.load_perfect_candidates(sample)
                        if self.settings.perfect_retrieval
                        else self.source_retriever.load_candidates(sample)
                    )
                ):
                    continue
                record = self.structure_cache.load(sample)
                if record is None or not record.valid:
                    missing.append(sample.sample_id)
            if missing:
                raise RuntimeError(
                    "Missing or stale TableAgent structure caches for: "
                    + ", ".join(missing[:20])
                    + ". Run with --table-agent-phase structure or all first."
                )
            return
        records = self.verify_samples(samples, force=self.settings.phase == "all")
        failed = [record for record in records if not record.valid]
        if failed:
            raise RuntimeError(f"TableAgent verification failed for {len(failed)} cache entries")
        if self.settings.phase == "all":
            self._prepared_source_samples.update(
                sample.sample_id
                for sample in samples
                if has_workbook_sources(sample) and self.settings.should_retrieve(sample)
            )

    def set_progress_callback(self, callback: Callable[[str], None] | None) -> None:
        self._progress_callback = callback

    def _progress(self, stage: str, **fields: Any) -> None:
        if self._progress_callback is None:
            return
        labels = {
            "prepare": "prepare",
            "prepare_extract": "prepare:extract",
            "prepare_metadata": "prepare:metadata",
            "prepare_cached": "prepare:cached",
            "prepare_error": "prepare:error",
            "prepare_layout": "prepare:layout",
            "prepare_done": "prepare:done",
            "retrieval": "retrieve",
            "rerank": "rerank",
            "render": "render",
            "layout": "layout",
            "verify": "verify",
            "structure_done": "structure:done",
            "qa": "qa",
            "answer": "answer",
            "done": "done",
        }
        parts = [labels.get(stage, stage)]
        if stage in {"prepare_layout", "prepare_done", "render", "layout", "verify"}:
            ordered_fields = [
                ("range", "range"),
                ("iteration", "iter"),
                ("direction", "dir"),
                ("workbook", "book"),
                ("sheet", "sheet"),
                ("sample", "sample"),
            ]
        else:
            ordered_fields = [
                ("sample", "sample"),
                ("workbook", "book"),
                ("sheet", "sheet"),
                ("table", "table"),
                ("range", "range"),
                ("iteration", "iter"),
                ("direction", "dir"),
            ]
        for key, label in ordered_fields:
            value = fields.get(key)
            if value is None or value == "":
                continue
            text = str(value)
            parts.append(f"{label}={text}")
        self._progress_callback(" | ".join(parts))

    def set_run_id(self, run_id: int) -> Path:
        if run_id < 1:
            raise ValueError("run_id must be at least 1")
        if self.settings.run_artifact_dir is not None:
            repeat_dir = self.settings.repeat_dir_template.format(run_id=run_id)
            self._artifact_dir = self.settings.run_artifact_dir / repeat_dir
        else:
            self._artifact_dir = self.settings.artifact_dir
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        return self._artifact_dir

    def _apply_generation_cap(self) -> None:
        if self.settings.generation_max_tokens is None:
            return
        if hasattr(self.llm, "max_tokens"):
            self.llm.max_tokens = self.settings.generation_max_tokens
        if hasattr(self.layout_vlm, "max_tokens"):
            self.layout_vlm.max_tokens = self.settings.generation_max_tokens

    @staticmethod
    def _client_config(client: Any) -> dict[str, Any]:
        return {
            "model_name": getattr(client, "model_name", None),
            "temperature": getattr(client, "temperature", None),
            "max_tokens": getattr(client, "max_tokens", None),
        }
