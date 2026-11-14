# Repository Structure

TableAgent answers questions over Excel workbooks. One question runs one exploration
([`TableAgent/src/README.md`](TableAgent/src/README.md)); everything else is the way in.

## Runtime Flow

```text
table-agent CLI / HTTP API (service/)          ise-table benchmark adapter (adapters/table_agent/ in ise-table)
        \                                                  /
         WorkbookQA (TableAgent/src/workbook_qa.py): settings, models, renderer, fallback answer
                  |
         answer_question (orchestrator.py): plan -> inspect / check each need -> synthesize -> review
```

## Top-Level Directories

```text
TableAgent/
|- TableAgent/   the library
|- service/      CLI and HTTP API
|- web/          React/Vite page: a chat over one uploaded workbook
|- tests/        pytest suite
`- sample/       example workbooks
```

## The Library: `TableAgent/`

```text
TableAgent/
|- __init__.py        exports WorkbookQA and QASettings
|- settings.py        QASettings: Checkers, generation cap, rendering; read from the `table_agent` config section
|- llm.py             BaseLLM and LLMResponse, the model interface
|- rendering/         workbook.py renders a cell range with LibreOffice for the vision model (pdfium_worker.py rasterizes)
|- utils/excel_utils.py   A1 addresses and CellRange
`- src/               the QA agent
   |- workbook_qa.py      WorkbookQA: run one question, fall back to one direct answer when the exploration fails
   |- orchestrator.py     the reasoning loop (reason_start, reason_step, reason_end)
   |- ledger.py           needs, evidence, checks, steps; the limits per question
   |- inspector.py        establishes one need in a notebook
   |- checker.py          tests a need's evidence (reverse, independent, debate) and the synthesis method
   |- synthesizer.py      computes the answer parts in code
   |- prompts.py          every prompt
   |- tools/              what the notebooks use: sheet.py (the `sheet` object), headers.py (header walk),
   |                      formats.py (formats and legends found in code), look.py (vision), notebook.py (sandbox)
   `- utils/              model calls (llm.py), workbook preview (workbook_view.py), run artifacts (trace.py),
                          the fallback model call (direct_answer.py)
```

## `service/`

- `config.py` loads the YAML config (with `${NAME:-default}` environment values) and picks model profiles.
- `clients.py` is the OpenAI-compatible model client.
- `runtime.py` holds `TableAgentService`: a job is one workbook and its questions, each answered in its own
  directory, `service.max_workers` at a time.
- `api.py` is the FastAPI app, `server.py` starts it (`table-agent-api`), `cli.py` is `table-agent`.

## Root Files

- `config.example.yaml`: the configuration to copy to `config.yaml`.
- `pyproject.toml`: package metadata, dependencies and the two console scripts.
