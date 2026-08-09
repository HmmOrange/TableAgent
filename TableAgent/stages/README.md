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
