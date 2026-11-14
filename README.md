# TableAgent

TableAgent answers natural-language questions over Excel workbooks. It plans what the question needs, reads
each value by writing Python against the workbook, has a Checker test every value it read, and computes the
answer in code. [`TableAgent/src/README.md`](TableAgent/src/README.md) describes the loop.

## Requirements

- Python 3.13 or newer and [uv](https://docs.astral.sh/uv/)
- OpenAI-compatible chat-completions endpoints for the answer model and, optionally, a vision model
- LibreOffice, when a vision model is configured (cell ranges are rendered for it to look at)

## Setup

```bash
uv sync
cp config.example.yaml config.yaml
```

Fill in `config.yaml`:

| Key | What to set |
| --- | --- |
| `llm` | Name of the answer model's profile under `models` |
| `vlm` | Name of the vision model's profile under `models` (leave it out to answer from cell values only) |
| `models.<name>` | `base_url` (with the API prefix, e.g. `http://localhost:8000/v1`), `model`, `api_key` |
| `table_agent` | How questions are answered: Checkers, generation cap, rendering ([`TableAgent/settings.py`](TableAgent/settings.py)) |
| `service` | Job directory, workers, upload limit, API key |

The example reads the endpoints from `TABLE_AGENT_LLM_BASE_URL`, `TABLE_AGENT_LLM_MODEL`,
`TABLE_AGENT_LLM_API_KEY`, `TABLE_AGENT_VLM_BASE_URL`, `TABLE_AGENT_VLM_MODEL`, `TABLE_AGENT_VLM_API_KEY` and
`TABLE_AGENT_SERVICE_API_KEY`; `${NAME:-default}` in any value reads an environment variable.

## Ask From The Command Line

```bash
uv run table-agent --config config.yaml --workbook sample/QA_sample.xlsx \
  --question "What is the total revenue in 2024?" --question "Which region grew most?"
```

The command prints the job as JSON: one entry per question with its answer, token usage, the path of its
`trace.md`, and its QA metadata. `--llm` / `--vlm` pick other profiles, `--output-dir` another job root. Each job
is a directory under `service.root_dir`:

```text
outputs/<job-id>/
  QA_sample.xlsx          the workbook, copied
  qa/q1/<run>/            one directory per question: trace.md, ledger.json, events.jsonl, events.log,
                          result.json, cells/ (every notebook cell), images/ (rendered ranges)
```

Workbooks must be `.xlsx`, `.xlsm`, `.xltx` or `.xltm`.

## Serve The API

```bash
uv run table-agent-api --config config.yaml --host 127.0.0.1 --port 3636
```

| Method and path | What it does |
| --- | --- |
| `GET /health`, `/health/live`, `/health/ready` | Liveness and readiness |
| `GET /v1/status` | Version and workers |
| `POST /v1/jobs/upload` | Multipart form: `file` (the workbook) and `payload` (`{"questions": ["..."]}`) |
| `POST /v1/jobs` | JSON `{"workbook": "<path>", "questions": ["..."]}`; only with `service.allow_local_paths`, and only under `service.allowed_input_roots` when set |

With `service.api_key` set, every `/v1/` request needs the header `X-API-Key`.

```bash
curl -X POST http://127.0.0.1:3636/v1/jobs/upload \
  -H "X-API-Key: $TABLE_AGENT_SERVICE_API_KEY" \
  -F 'payload={"questions": ["What is the total revenue in 2024?"]}' \
  -F "file=@sample/QA_sample.xlsx"
```

The web page in [`web/`](web/README.md) is a chat over one uploaded workbook that calls this API.

## Use As A Library

```python
from pathlib import Path

from TableAgent import QASettings, WorkbookQA

agent = WorkbookQA(llm, QASettings(checker=("independent",), checker_view="frame"), vision_llm)
response, qa = agent.run(
    question="What is the total revenue in 2024?",
    workbook_path=Path("book.xlsx"),
    qa_artifact_dir=Path("runs/q1"),
    fallback_prompt="Question: ...\n\nTable content:\n...",  # used only when the exploration fails
)
```

`llm` is any object with `generate(prompt, system_prompt)` returning a `TableAgent.llm.LLMResponse`; the vision
model also needs `generate_with_image(prompt, image_path, system_prompt)`. `service/clients.py` has one for
OpenAI-compatible endpoints.

## Development

```bash
uv run pytest
```
