# TableAgent

TableAgent extracts verified spreadsheet structure and answers natural-language
questions against one or more workbooks. It includes workbook rendering, deterministic
range verification, multi-sheet retrieval, and notebook-backed QA operators.

## Setup

```bash
uv sync
Copy-Item config.example.yaml config.yaml
# Set TABLE_AGENT_LLM_* and TABLE_AGENT_VLM_* for your model server.
```

`config.yaml` is intentionally ignored by git. It contains local model endpoints and
keys; only `config.example.yaml` belongs in the public repository. Set
`TABLE_AGENT_SERVICE_API_KEY` to require `X-API-Key` on every `/v1/*` endpoint.

## Python API

The service API handles one or more Excel workbooks and any number of queries. A
`structure` run only generates verified `structure.yaml` files, `qa` reuses existing
structures, and `all` performs both stages before answering.

```python
from TableAgent import TableAgentService

service = TableAgentService.from_config("config.yaml")
result = service.run(
    stage="all",
    workbooks=["./sales.xlsx", "./costs.xlsx"],
    queries=["What was the largest quarterly increase?", "List the cost outliers."],
)
for answer in result["answers"]:
    print(answer["answer"])
```

Custom model clients can be injected with `TableAgentService(..., llm_client=..., layout_vlm_client=...)`.
The answer client implements `generate()`; the layout client implements
`generate_with_image()`.

## HTTP API

Start the service with a private config file:

```bash
uv run table-agent-api --config config.yaml --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health` and `GET /health/ready` for liveness/readiness.
- `GET /v1/status` for worker and job counts.
- `POST /v1/jobs/upload` for multipart workbook uploads. Pass a `payload` form field
  containing `{"stage":"all","queries":["..."]}` and one or more `files` fields.
- `POST /v1/jobs` for trusted server-side workbook paths (disabled by default).
- `GET /v1/jobs/{job_id}` for asynchronous status and results; add `?wait=true` to
  either create endpoint when a synchronous response is preferred.
- `GET /v1/jobs/{job_id}/artifacts` and `/artifacts/{path}` to list/download generated
  structures and QA artifacts.

Example upload:

```bash
curl -X POST http://localhost:8000/v1/jobs/upload \
  -H 'X-API-Key: your-service-key' \
  -F 'payload={"stage":"all","queries":["How many rows are present?"]}' \
  -F 'files=@sample/QA_sample.xlsx'
```

## Tests

```bash
uv run pytest -q
```
