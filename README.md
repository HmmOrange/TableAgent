# TableAgent

TableAgent extracts verified structure from Excel workbooks and answers natural-language
questions over one or more sheets. The project combines workbook rendering, layout-model
structure extraction, deterministic range verification, table retrieval, and
notebook-backed QA.

The repository is split into two clear layers:

- `TableAgent/` contains the reusable extraction and QA library.
- `service/` contains the model clients, application service, FastAPI routes, and server
  entry point.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- LibreOffice for workbook-to-image rendering
- OpenAI-compatible chat-completions endpoints for the answer LLM and layout VLM, unless
  custom clients are injected

LibreOffice must be available as `soffice`/`libreoffice` on `PATH`, or configured with
`table_agent.libreoffice_path`.

## Installation

```bash
uv sync
```

Create a private configuration file from the checked-in example:

```bash
cp config.example.yaml config.yaml
```

PowerShell equivalent:

```powershell
Copy-Item config.example.yaml config.yaml
```

`config.yaml` is ignored by git. Do not commit model API keys or private endpoints.

## Configuration

The default configuration reads model settings from environment variables:

| Variable | Purpose |
| --- | --- |
| `TABLE_AGENT_LLM_BASE_URL` | Answer-model OpenAI-compatible base URL |
| `TABLE_AGENT_LLM_MODEL` | Answer-model name |
| `TABLE_AGENT_LLM_API_KEY` | Answer-model API key |
| `TABLE_AGENT_VLM_BASE_URL` | Layout-model OpenAI-compatible base URL |
| `TABLE_AGENT_VLM_MODEL` | Layout-model name |
| `TABLE_AGENT_VLM_API_KEY` | Layout-model API key |
| `TABLE_AGENT_SERVICE_API_KEY` | Optional key required by every `/v1/*` endpoint |

Model URLs should include the API prefix, such as `http://localhost:8000/v1`. TableAgent
appends `/chat/completions` when making requests.

Important configuration groups:

- `models` and `vlm_models`: model endpoints, names, timeouts, and generation settings.
- `table_agent`: cache, rendering, retrieval, verification, and QA settings.
- `service`: API storage, worker count, upload limits, authentication, and local-path
  policy.

See [`config.example.yaml`](config.example.yaml) for all supported settings.

## Run The API

Start the HTTP service from the repository root:

```bash
uv run table-agent-api --config config.yaml --host 127.0.0.1 --port 8000
```

The top-level `llm.provider` and `vlm.provider` values are used by default. Override
either selection with a configured profile name when starting the server:

```bash
uv run table-agent-api --config config.yaml --llm answer_model --vlm layout_model
```

For an installation without the generated console script:

```bash
python -m service.server --config config.yaml
```

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `GET` | `/health/ready` | Storage readiness check |
| `GET` | `/v1/status` | Worker and job counts |
| `POST` | `/v1/jobs/upload` | Submit uploaded workbooks |
| `POST` | `/v1/jobs` | Submit trusted server-side paths when enabled |
| `GET` | `/v1/jobs/{job_id}` | Read job status and results |
| `GET` | `/v1/jobs/{job_id}/artifacts` | List generated artifacts |
| `GET` | `/v1/jobs/{job_id}/artifacts/{path}` | Download an artifact |

Health endpoints are public. When `service.api_key` is configured, requests to `/v1/*`
must include `X-API-Key`.

### Upload A Workbook

The upload endpoint accepts a JSON `payload` form field and one or more `files` fields:

```bash
curl -X POST "http://127.0.0.1:8000/v1/jobs/upload?wait=true" \
  -H "X-API-Key: your-service-key" \
  -F 'payload={"stage":"all","queries":["What is the total revenue?"]}' \
  -F "files=@sample/QA_sample.xlsx"
```

Without `wait=true`, the endpoint returns immediately. Poll `/v1/jobs/{job_id}` until
the status becomes `succeeded` or `failed`.

Server-side workbook paths are disabled by default. To enable `/v1/jobs`, set
`service.allow_local_paths: true` and restrict access with `service.allowed_input_roots`.

## Use The Python Service

The service layer can be called without HTTP:

```python
from service import TableAgentService

service = TableAgentService.from_config("config.yaml")
result = service.run(
    stage="all",
    workbooks=["./sales.xlsx", "./costs.xlsx"],
    queries=[
        "Which quarter had the largest revenue increase?",
        "List the largest cost outliers.",
    ],
)

for item in result["answers"]:
    print(item["answer"])
```

Custom model clients can be injected into `TableAgentService`. The answer client must
provide `generate()`. The layout client must provide `generate_with_image()`.

## Processing Stages

| Stage | Behavior |
| --- | --- |
| `structure` | Render workbooks, extract structure, verify ranges, and populate the cache |
| `qa` | Reuse valid cached structures and answer the supplied queries |
| `all` | Generate or refresh structure first, then answer the queries |

`qa` fails when a workbook has no valid structure cache. Run `structure` or `all` first.

Supported input extensions are `.xls`, `.xlsx`, `.xlsm`, `.xltx`, and `.xltm`. Inputs
are normalized into content-addressed `.xlsx` files under the configured service root.

## Repository Layout

```text
TableAgent/
  configs/              Configuration loading and pipeline settings
  pipeline/             Structure extraction, retrieval, and QA orchestration
  rendering/            Workbook conversion and image rendering
  structure/            Layout extraction and deterministic verification
  QA/                   Notebook environment, agents, actions, and operators
  schema/                Shared data contracts
service/
  api.py                 FastAPI routes and asynchronous job manager
  runtime.py             Workbook/query application service
  clients.py             OpenAI-compatible LLM and VLM client
  server.py              `table-agent-api` command entry point
sample/                  Example workbooks, structures, and a direct QA script
tests/                   Unit and integration tests
config.example.yaml      Public configuration template
```

The old root-level `configs`, `datasets`, `pipelines`, and `utils` packages now live
under `TableAgent`. The inactive standalone `table2img` package and CLI were removed;
workbook rendering remains under `TableAgent/rendering/`. Import core functionality
from `TableAgent` and deployment functionality from `service`.

## Artifacts And Cache

By default, service data is written under `outputs/table_agent/service/`:

```text
inputs/                  Content-addressed normalized workbooks
jobs/<job_id>/           Status, result metadata, and run artifacts
structure/               Reusable prepared structures
structure/cache/         Content-addressed structure cache
uploads/                 Temporary upload staging, removed after each job
```

Change `service.root_dir` to relocate all service-owned data. Pipeline artifact and
cache paths can also be configured independently under `table_agent`.

## Development

Run the complete test suite:

```bash
python -m pytest -q
```

The current suite covers configuration, rendering, structure verification, retrieval,
multi-table operators, QA, the service runtime, and the HTTP API.

For retrieval-specific extension guidance, see
[`TableAgent/pipeline/retrieval/README.md`](TableAgent/pipeline/retrieval/README.md).
