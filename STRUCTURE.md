# Repository Structure

TableAgent extracts verified table structure from Excel workbooks, retrieves the
most relevant workbook context, and answers natural-language questions. The code
is organized around three business stages: structure, retrieval, and QA.

## Runtime Flow

```text
CLI or HTTP API (`service/`)
    -> configuration and model clients (`TableAgent/configs/`, `TableAgent/llm.py`)
    -> pipeline composition (`TableAgent/pipeline/`)
    -> structure stage (`TableAgent/stages/structure/`)
    -> retrieval stage (`TableAgent/stages/retrieval/`)
    -> QA stage (`TableAgent/stages/qa/`)
    -> run results and artifacts
```

`TableAgent/pipeline/table_agent_pipeline.py` is the main composition root. It
constructs the stage implementations and coordinates their handoffs. Business
logic that belongs to one phase should remain in that phase's package rather than
being added to the orchestrator.

## Top-Level Directories

```text
TableAgent/
|- TableAgent/       Core Python library
|- service/          CLI and HTTP service entry points
|- web/              React/Vite browser interface
|- tests/            Automated Python tests
|- sample/           Example workbooks, structures, and sample scripts
|- data/             Local benchmark or development datasets
|- outputs/          Generated run results and persisted artifacts
|- logs/             Generated runtime and QA logs
|- docker/           Local Docker-related environment configuration
|- .codegraph/       Generated CodeGraph index for repository navigation
|- .venv/            Local Python virtual environment
|- .pytest_cache/    Generated pytest cache
`- table_agent.egg-info/ Generated Python package metadata
```

### `TableAgent/`

The installable application library. It owns the pipeline, stage logic, schemas,
rendering, configuration, artifact conventions, and shared infrastructure.

### `service/`

The user-facing Python entry layer.

- `cli.py` implements the `table-agent` command.
- `server.py` starts the `table-agent-api` service.
- `api.py` defines the FastAPI endpoints and request handling.
- `runtime.py` builds and runs pipelines for CLI and API jobs.
- `clients.py` constructs configured LLM, VLM, and embedding clients.

This package should translate external requests into pipeline inputs. Core table
processing behavior belongs under `TableAgent/`.

### `web/`

The React 19 and Vite frontend used to submit workbooks and display results.

- `src/` contains the application component, browser entry point, and styles.
- `index.html` is the Vite HTML entry point.
- `vite.config.js` configures the development server and API proxy.
- `package.json` defines frontend dependencies and development/build commands.

### `tests/`

The pytest suite. Tests cover the service layer, configuration, rendering,
structure extraction, retrieval, QA orchestration, and end-to-end pipeline behavior.
`mock_policy.py` contains shared test policy/helpers for model mocking.

### `sample/`

Small example workbooks and expected YAML artifacts used for manual runs,
development, and demonstrations. This is not production application code.

### `data/`

Local benchmark and development datasets, currently including RealHiTBench and
SiFlex data. Treat this directory as input data rather than source code.

### `outputs/`

Generated run directories. A run can contain `run.json`, workbook schemas,
metadata, verified sheet structures, retrieval cards, embeddings, and QA execution
artifacts. Code should not depend on a specific existing run in this directory.

### `logs/`

Generated pipeline and QA event logs. These files are operational output and are
not part of the application architecture.

### `docker/`

Local Docker-related environment configuration. At present it contains an
environment file rather than a complete image or Compose definition.

## Core Library: `TableAgent/`

```text
TableAgent/
|- artifacts/       Artifact paths, metadata, and output schemas
|- configs/         Configuration loading and typed settings
|- pipeline/        Cross-stage composition and run coordination
|- rendering/       Workbook-to-image conversion and image handling
|- schema/          Shared domain models and value objects
|- shared/          Helpers reused by multiple stages
|- stages/          Phase-specific business logic
|- utils/           General Excel, structure, and text utilities
|- llm.py           LLM interface and response types
`- run_logging.py   Run/event logging support
```

### `TableAgent/artifacts/`

Defines stable artifact layout and serialization conventions.

- `layout.py` builds workbook, sheet, and run artifact paths.
- `metadata.py` reads and writes artifact metadata.
- `schema.py` defines artifact-facing record shapes.

### `TableAgent/configs/`

Loads YAML configuration, expands environment values, validates model profiles,
and exposes typed pipeline settings.

- `config.py` is the general configuration loader.
- `table_agent.py` owns TableAgent pipeline settings.
- `routing.py` owns retrieval and QA routing settings.
- `llm_config.py`, `vlm_config.py`, `embedding_config.py`, and
  `models_config.py` resolve model-specific configuration.

### `TableAgent/pipeline/`

Composes stages without owning phase-specific business logic.

- `table_agent_pipeline.py` constructs structure, retrieval, and QA components and
  coordinates the main workflow without inheriting stage mixins.
- `contracts.py` defines `PipelineOutput`, stage protocols, the runtime dependency
  contract, and the composed `PipelineStages` bundle.
- `component.py` provides the small runtime-backed base used by composed runners.
- `pipeline_run.py` contains `PipelineRunner`, which routes prepared-source and
  cached runs and assembles run results.
- `progress.py` formats the stable progress-event messages consumed by the service.
- `base.py` contains the `BasePipeline` interface and re-exports `PipelineOutput`.

The pipeline is a composition root. It owns mutable run state and constructs the
stage objects, while `StructurePipeline`, `RetrievalPipeline`, `VerifiedQAPipeline`,
and `SourceQAPipeline` receive that state through an explicit runtime contract.
The orchestrator uses composition rather than pipeline-level mixin inheritance.

### `TableAgent/rendering/`

Converts workbook content into images used by layout extraction.

- `converter.py` handles workbook conversion.
- `workbook.py` coordinates workbook and worksheet rendering.
- `image_utils.py` provides image sizing, tiling, and related helpers.
- `pdfium_worker.py` isolates PDFium rendering work.

### `TableAgent/domain/`

Shared spreadsheet concepts independent of pipeline stages. `ranges.py` models
cells and selections, while `structure.py` models header trees. Pipeline samples
live in `TableAgent/pipeline/sample.py`, and QA-only models remain under
`TableAgent/stages/qa/`.

### `TableAgent/shared/`

Cross-stage infrastructure that has more than one real consumer.

- `agents.py` contains shared agent/message behavior.
- `artifacts.py` contains reusable artifact helpers.
- `pipeline.py` contains shared pipeline utilities.
- `prompting.py` contains prompt construction helpers used across boundaries.

### `TableAgent/utils/`

Small general-purpose helpers for Excel operations, structure manipulation, and
table-to-text conversion. Phase-specific helpers should stay with their stage.

## Stage Packages: `TableAgent/stages/`

Each stage exposes an explicit entry point in `stage.py` and typed handoff objects
in `contracts.py`. The stage package is the canonical owner of its business logic.

### `TableAgent/stages/structure/`

Ingests workbooks and produces verified structures and retrieval-ready corpus
artifacts.

- `layout/` runs the layout VLM, parses its YAML, and traverses rendered viewports.
- `verification/` performs deterministic checks and verification worker execution.
- `relations/` scans, classifies, normalizes, assigns, and writes formula/table
  relationships.
- `source_preparer.py` coordinates workbook and sheet preparation.
- `metadata.py` extracts workbook and worksheet metadata.
- `card_builders.py` builds workbook, sheet, and table retrieval cards.
- `retrieval_artifacts.py` writes corpus cards and optional embeddings.
- `cache.py` handles structure records used by non-workbook/sample flows.
- `traversal.py` controls viewport movement across worksheets.
- `structure_prompts.py` contains layout/structure prompt templates.
- `pipeline.py` contains the composed structure coordinator, while `stage.py` and
  `contracts.py` define the public stage boundary.

### `TableAgent/stages/retrieval/`

Selects relevant prepared workbook, sheet, and table context for a question.

- `candidate_loading.py` reads candidates from structure-stage artifacts.
- `ranking.py`, `scoring.py`, and `embeddings.py` implement lexical, BM25,
  embedding, entity, and hybrid scoring support.
- `reranking.py` applies optional model-assisted reranking.
- `indexed_guards.py` validates indexed retrieval inputs.
- `perfect.py` supports explicit/perfect candidate selection.
- `source_retriever.py` orchestrates prepared-source selection.
- `retrieval_prompts.py` contains retrieval and reranking prompts.
- `pipeline.py` contains retrieval filtering coordination, while `stage.py` and
  `contracts.py` define the stage boundary and table-retriever extension contract.

Retrieval chooses context; it does not calculate the final answer.

### `TableAgent/stages/qa/`

Plans and executes question answering against verified workbook structures.

- `actions/` implements bounded operations such as planning, code generation,
  notebook execution, review, and final-answer review.
- `agents/` contains planner, ReAct, answer, and synthesis agents.
- `environment/` owns the persistent notebook/runtime environment and QA logging.
- `operators/` exposes deterministic workbook, table, range, filter, relation,
  multi-tab, and routing operations to generated code.
- `prompts/` contains QA prompt templates grouped by agent/action.
- `runner.py` and `runner_*.py` coordinate plan execution, support code, and
  artifact persistence.
- `header_hints.py` prepares verified header context for QA.
- `source_pipeline.py` converts retrieval results into QA inputs.
- `pipeline.py` contains verified-QA coordination and fallback handling, while
  `stage.py` and `contracts.py` define the stage boundary.

See `TableAgent/stages/qa/PIPELINE.md` for the detailed planning and execution
flow.

## Root Files

- `README.md` documents installation, configuration, CLI/API usage, and outputs.
- `pyproject.toml` defines the Python package, dependencies, and console scripts.
- `uv.lock` pins Python dependency versions.
- `config.example.yaml` is the checked-in configuration template.
- `config.yaml` is the local runtime configuration and may contain secrets.
- `.gitignore` defines files and generated directories excluded from version control.

## Refactor Ownership Rules

- Put phase-specific behavior in its package under `TableAgent/stages/`.
- Keep `TableAgent/pipeline/` limited to constructing stages and coordinating
  explicit handoffs.
- Put code in `TableAgent/shared/` only when multiple stages genuinely use it.
- Keep external transport and request parsing in `service/`.
- Keep shared spreadsheet types in `TableAgent/domain/`; keep pipeline samples in
  `TableAgent/pipeline/` and stage-only models and contracts with their stage.
- Treat `outputs/`, `logs/`, local datasets, caches, and package metadata as
  generated or operational state, not application source.
