# Pipeline Stages

The stage packages are the canonical home for phase-specific business logic.

```text
stages/
  structure/
    layout/                 VLM layout extraction
    verification/           deterministic structure checks
    metadata.py             workbook and sheet metadata extraction
    card_builders.py        workbook/sheet/table retrieval-card preparation
    retrieval_artifacts.py persisted corpus cards and embeddings
    source_preparer.py      workbook ingestion and structure artifacts
    cache.py                non-workbook structure cache
    traversal.py            workbook viewport traversal
    pipeline.py             structure-stage pipeline hooks
  retrieval/
    contracts.py            retrieval inputs, outputs, and table contracts
    candidate_loading.py    loading prepared structure artifacts
    ranking.py              lexical/BM25/query-vector ranking
    source_retriever.py     source selection orchestration
    embeddings.py           query embedding clients
    ...                      guards, reranking, scoring
  qa/
    prompts/                 QA prompt templates
    actions/                verified QA actions
    agents/                 QA agents
    operators/              workbook/table operators
    environment/            notebook execution environment
    runner*.py              QA execution and artifact handling
    pipeline.py             verified QA stage hooks
    source_pipeline.py      retrieval-to-QA handoff
```

The top-level `TableAgent/pipeline/` package is intentionally limited to pipeline
composition. Cross-stage helpers live under
`TableAgent/shared/`; rendering, schemas, configuration, integrations, and other
cross-cutting concerns retain their own top-level packages.

## Adding a New Stage

Add one package under `TableAgent/stages/` and keep all business logic specific
to that phase inside the package. For example, a normalization stage would use:

```text
stages/
  normalization/
    __init__.py       public stage exports
    contracts.py      NormalizationInput and NormalizationOutput
    stage.py          NormalizationStage.run(input) -> output
    ...               normalization-specific implementation modules
```

Follow these steps:

1. Define explicit input and output dataclasses in `contracts.py`. A stage should
   consume the previous stage's output, or a deliberately small handoff derived
   from it, and return everything the next stage needs.
2. Implement one stage entry point in `stage.py` with a typed
   `run(stage_input) -> stage_output` method. Keep detailed phase logic behind
   this entry point and inside the same stage package.
3. Export the stage and its contracts from the package `__init__.py` and, when a
   project-level import is useful, from `TableAgent/stages/__init__.py`.
4. Construct the stage in `TableAgent/pipeline/table_agent_pipeline.py` and call
   it at the intended boundary in the pipeline execution flow. The orchestrator
   should only prepare the handoff and compose stages; it should not implement
   the new stage's business logic.
5. Add configuration only when the stage has meaningful optional behavior. If
   the stage must be independently runnable, extend the existing phase/CLI
   selection without changing current command behavior.
6. Add focused tests for the stage contract and behavior, plus an integration
   test confirming the surrounding stages receive the expected handoff.

For example, inserting normalization should result in the following boundaries:

```text
StructureOutput
      -> NormalizationInput
      -> NormalizationOutput
      -> RetrievalInput
```

Do not move cross-cutting concerns into the new package merely because the stage
uses them. Reuse `TableAgent/shared/`, configuration, schemas, rendering, and
integrations when the same functionality is shared across multiple stages.
Avoid registries, factories, or dependency-injection frameworks unless multiple
real stages require them; explicit composition is the default.
