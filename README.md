# TableAgent

TableAgent extracts verified table structure from Excel workbooks and answers
natural-language questions over one or more sheets.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- LibreOffice for workbook rendering
- OpenAI-compatible chat-completions endpoints for the answer LLM and layout VLM

LibreOffice must be available as `soffice` or `libreoffice` on `PATH`. Otherwise,
set `table_agent.libreoffice_path` in the configuration.

## Setup

Install the dependencies:

```bash
uv sync
```

Create the local configuration:

```bash
cp config.example.yaml config.yaml
```

PowerShell:

```powershell
Copy-Item config.example.yaml config.yaml
```

### Fill In The Configuration

Edit `config.yaml` before running TableAgent. The important sections are:

| Configuration | What to fill in |
| --- | --- |
| `models.answer_model` | Answer LLM `base_url`, `model`, and `api_key` |
| `vlm_models.layout_model` | Vision/layout model `base_url`, `model`, and `api_key` |
| `llm.provider` | Name of the default profile under `models` |
| `vlm.provider` | Name of the default profile under `vlm_models` |
| `table_agent.libreoffice_path` | LibreOffice executable path when it is not on `PATH` |
| `service.root_dir` | Where jobs, generated structures, and cached inputs are stored |

The profile names must match. For example, `llm.provider: answer_model` selects
`models.answer_model`, while `vlm.provider: layout_model` selects
`vlm_models.layout_model`.

The checked-in example reads model values from these environment variables, so you
can either keep those references and set the variables or replace them with values
directly in `config.yaml`:

| Variable | Value |
| --- | --- |
| `TABLE_AGENT_LLM_BASE_URL` | Answer LLM API base URL |
| `TABLE_AGENT_LLM_MODEL` | Answer LLM model name |
| `TABLE_AGENT_LLM_API_KEY` | Answer LLM API key |
| `TABLE_AGENT_VLM_BASE_URL` | Layout VLM API base URL |
| `TABLE_AGENT_VLM_MODEL` | Layout VLM model name |
| `TABLE_AGENT_VLM_API_KEY` | Layout VLM API key |
| `TABLE_AGENT_SERVICE_API_KEY` | Optional API key for the HTTP service |

Model base URLs should include the API prefix, for example
`http://localhost:8000/v1`. TableAgent appends `/chat/completions`. The VLM endpoint
must accept images in OpenAI-compatible message content.

See [`config.example.yaml`](config.example.yaml) for the remaining pipeline,
retrieval, rendering, and service settings.

## Run From The CLI

Run commands from the repository root. The CLI prints a JSON result containing the
job ID, answers, and generated artifact paths.

### Ingestion: Produce `structure.yaml`

Run the `structure` stage to render the workbook, detect its tables, verify their
ranges, and cache one `structure.yaml` per worksheet:

```bash
uv run table-agent --config config.yaml --stage structure --workbook sample/QA_sample.xlsx
```

This stage uses the VLM but does not use the answer LLM.

With the default `service.root_dir`, the canonical files are stored at:

```text
outputs/table_agent/service/structure/sources/<source-id>/structure.yaml
```

The run also exports readable copies to:

```text
outputs/table_agent/service/jobs/<job-id>/structures/*.yaml
```

### Run QA Against Existing Structures

After ingestion, run the `qa` stage with the same workbook. It reuses the cached
structures and calls only the answer LLM:

```bash
uv run table-agent --config config.yaml --stage qa --workbook sample/QA_sample.xlsx --query "What is the total revenue?"
```

If the workbook is new or has changed, `qa` reports a missing or stale structure
cache. Run `structure` or `all` first.

### Run End-To-End

Run the `all` stage to ensure structures exist and then answer the question in one
command:

```bash
uv run table-agent --config config.yaml --stage all --workbook sample/QA_sample.xlsx --query "What is the total revenue?"
```

This stage requires both the layout VLM and answer LLM.

Repeat `--workbook` to process multiple workbooks and repeat `--query` to ask
multiple questions. Select a different configured model profile with `--llm NAME`
or `--vlm NAME`. Run `uv run table-agent --help` for the complete CLI reference.

Supported workbook extensions are `.xls`, `.xlsx`, `.xlsm`, `.xltx`, and `.xltm`.
Each run is saved under `service.root_dir/jobs/<job-id>/`, including `run.json`.

## Serve The API

Start the HTTP service after configuring the same LLM and VLM profiles:

```bash
uv run table-agent-api --config config.yaml --host 127.0.0.1 --port 8000
```

Submit a workbook and wait for an end-to-end result:

```bash
curl -X POST "http://127.0.0.1:8000/v1/jobs/upload?wait=true" \
  -H "X-API-Key: your-service-key" \
  -F 'payload={"stage":"all","queries":["What is the total revenue?"]}' \
  -F "files=@sample/QA_sample.xlsx"
```

Omit `X-API-Key` when `service.api_key` is empty. Without `wait=true`, poll
`GET /v1/jobs/{job_id}`. Generated files are available through
`GET /v1/jobs/{job_id}/artifacts`, and the interactive API documentation is at
`http://127.0.0.1:8000/docs`.

Server-side workbook paths are disabled by default. To enable `POST /v1/jobs`, set
`service.allow_local_paths: true` and restrict access with
`service.allowed_input_roots`.

## Development

Run the test suite:

```bash
uv run pytest -q
```
