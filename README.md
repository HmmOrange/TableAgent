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
job ID, processing results, answers, and generated artifact paths. Each run is saved
under `service.root_dir/jobs/<job-id>/`, including the complete result in `run.json`.
When no job ID is supplied programmatically, the generated ID is a UTC timestamp such
as `2026-07-22T16-05-54.123456Z`.

### Ingestion

#### Description and requirements

Use the `structure` stage to ingest one or more workbooks. TableAgent renders each
selected worksheet, detects and verifies its table ranges, and caches a
`structure.yaml` file for later QA requests.

By default, valid cached structures are reused. Add `--force` to regenerate the
selected worksheet structures with the layout VLM.

- At least one `--workbook` is required. Supported extensions are `.xls`, `.xlsx`,
  `.xlsm`, `.xltx`, and `.xltm`.
- Structure generation requires LibreOffice and the configured layout VLM.
- Schema generation also requires the configured answer LLM to describe the
  workbook and its worksheets.

#### Example: One workbook

Ingestion always generates `schema.yaml`, `metadata.json`, and retrieval card
artifacts:

```bash
uv run table-agent --config config.yaml --stage structure --workbook sample/QA_sample.xlsx
```

With the default `service.root_dir`, canonical artifacts are stored at:

```text
outputs/table_agent/service/structure/sources/<workbook-id>/
  schema.yaml
  metadata.json
  <sheet-name>/
    structure.yaml
    metadata.yaml
    metadata.json
    sheet_text.txt
    table.png
    table.metadata.json
    retrieval_cards.jsonl
    retrieval_cards.csv
  retrieval_cards.jsonl
  retrieval_cards.csv
```

Each job also exports readable copies using the original workbook and worksheet
names:

```text
outputs/table_agent/service/jobs/<job-id>/workbooks/<workbook-name>/
  schema.yaml
  metadata.json
  retrieval_cards.jsonl
  retrieval_cards.csv
  <sheet-name>/
    structure.yaml
    retrieval_cards.jsonl
    retrieval_cards.csv
    ...
```

`retrieval_cards.jsonl` and `retrieval_cards.csv` contain the card text used by
retrieval. Records are labeled with `retrieval_type` (`data` or `metadata`) and
`retrieval_level` (`workbook`, `sheet`, or `table`). The card text intentionally
omits layout-only noise such as used ranges and merged ranges.

Add `--embed` when you want ingestion to also export `retrieval_cards.pkl`. The
pickle stores the same records plus an `embedding` field with the embedding model,
dimension, and vector values.

The CLI result includes per-sheet records in `structures` and workbook-level paths
in `schema_artifacts` and `metadata_artifacts`. A schema embeds the parsed structure
for each selected worksheet:

```yaml
Summary:
  id: summary
  description: Concise LLM-generated description of the worksheet.
  structure:
    table1:
      name: Revenue summary
      headers: []
```

`metadata.json` uses the following stable keys:

```json
{
  "name": "book.xlsx",
  "description": "Workbook summary generated from schema.yaml.",
  "sheet_names": ["Summary", "Detail"],
  "author": null,
  "date_created": null,
  "date_modified": null,
  "size_bytes": 12345
}
```

#### Available flags

| Flag | Description |
| --- | --- |
| `--config PATH` | Configuration file to load. Defaults to `config.yaml`. |
| `--stage structure` | Runs ingestion only. |
| `--workbook PATH` | Workbook to ingest. Repeat the flag to ingest multiple workbooks. |
| `--embed` | Also writes `retrieval_cards.pkl` with retrieval card embeddings. |
| `--force` | Regenerates cached worksheet structures. Valid only with `--stage structure` or `--stage all`. |
| `--sheet NAME[,NAME...]` | Processes only the named worksheets. Repeat the flag or separate names with commas. |
| `--llm NAME` | Overrides the configured answer LLM profile used for descriptions. |
| `--vlm NAME` | Overrides the configured layout VLM profile used for structure detection. |

Worksheet matching is exact and case-sensitive. When multiple workbooks are
provided, every requested worksheet must exist in every workbook. Metadata always
lists every worksheet in the workbook, even when `--sheet` limits structure and
schema processing.

For example, the following command ingests `Summary`, `Detail`, and `Archive` only:

```bash
uv run table-agent --config config.yaml --stage structure \
  --workbook sample/QA_sample.xlsx \
  --sheet "Summary,Detail" --sheet Archive
```

To rebuild the cached structures for the selected worksheets, add `--force`:

```bash
uv run table-agent --config config.yaml --stage structure \
  --workbook sample/QA_sample.xlsx --force
```

### QA

#### Description and requirements

Use the `qa` stage to answer natural-language questions using previously generated
worksheet structures. This stage reuses the cached structures and calls the answer
LLM; it does not call the layout VLM.

- At least one `--workbook` and one non-empty `--query` are required.
- The workbook must already have a valid structure cache created by the `structure`
  or `all` stage.
- If the workbook is new or has changed, QA reports a missing or stale cache. Run
  ingestion again before retrying.

#### Example: One workbook and one query

```bash
uv run table-agent --config config.yaml --stage qa \
  --workbook sample/QA_sample.xlsx \
  --query "What is the total revenue?"
```

#### Output

The CLI prints a JSON result with the answer and supporting execution details:

```json
{
  "job_id": "<job-id>",
  "stage": "qa",
  "workbooks": ["QA_sample.xlsx"],
  "structures": [
    {
      "workbook": "QA_sample.xlsx",
      "sheet": "Summary",
      "status": "good",
      "cache_hit": true,
      "structure": "<cached structure>",
      "artifact": "workbooks/QA_sample.xlsx/Summary/structure.yaml"
    }
  ],
  "schema_artifacts": [
    {
      "workbook": "QA_sample.xlsx",
      "artifact": "workbooks/QA_sample.xlsx/schema.yaml"
    }
  ],
  "metadata_artifacts": [
    {
      "workbook": "QA_sample.xlsx",
      "artifact": "workbooks/QA_sample.xlsx/metadata.json"
    }
  ],
  "answers": [
    {
      "query": "What is the total revenue?",
      "answer": "<answer>",
      "latency": 0.0,
      "token_usage": {},
      "workbook": "QA_sample.xlsx",
      "sheets": ["Summary"],
      "verification": {},
      "retrieval": {},
      "qa": {}
    }
  ],
  "artifacts": [
    "workbooks/QA_sample.xlsx/Summary/structure.yaml",
    "workbooks/QA_sample.xlsx/schema.yaml",
    "workbooks/QA_sample.xlsx/metadata.json"
  ]
}
```

The exact values depend on the workbook, retrieval result, and model response.
Cached per-sheet structures and the selected workbook artifacts are copied into the
job directory.

#### Available flags

| Flag | Description |
| --- | --- |
| `--config PATH` | Configuration file to load. Defaults to `config.yaml`. |
| `--stage qa` | Runs QA against cached structures. |
| `--workbook PATH` | Workbook to query. Repeat the flag to query multiple workbooks together. |
| `--query TEXT` | Question to answer. Repeat the flag to ask multiple questions. |
| `--sheet NAME[,NAME...]` | Limits retrieval to the named worksheets. |
| `--llm NAME` | Overrides the configured answer LLM profile. |

The `--vlm` option is accepted by the CLI but is not used during the `qa` stage.

### Run End-To-End

Run the `all` stage to ingest the workbook when needed and answer the question in
one command:

```bash
uv run table-agent --config config.yaml --stage all --workbook sample/QA_sample.xlsx --query "What is the total revenue?"
```

This stage requires both the layout VLM and answer LLM.

Run `uv run table-agent --help` for the complete CLI reference.

## Serve The API

Start the HTTP service after configuring the same LLM and VLM profiles:

```bash
uv run table-agent-api --config config.yaml --host 127.0.0.1 --port 8000
```

Submit a workbook and wait for an end-to-end result:

```bash
curl -X POST "http://127.0.0.1:8000/v1/jobs/upload?wait=true" \
  -H "X-API-Key: your-service-key" \
  -F 'payload={"stage":"all","queries":["What is the total revenue?"],"embed":true,"sheets":["Summary,Detail"]}' \
  -F "files=@sample/QA_sample.xlsx"
```

Both `POST /v1/jobs` and `POST /v1/jobs/upload` accept `embed` and `sheets`.
Ingestion always generates workbook schema and metadata artifacts. Sheet list
entries may contain comma-separated names.

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
