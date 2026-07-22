# TableAgent

TableAgent is an image-assisted table-structure and question-answering pipeline.
This repository contains the extracted core used by `ise-table`, including structure
generation, deterministic verification, retrieval, workbook rendering, and QA agents.

## Setup

```bash
uv sync
```

## Library usage

TableAgent keeps model clients injectable. The answer client must implement
`generate()`, while the layout client must implement `generate_with_image()`.

```python
from TableAgent import TableAgentPipeline

pipeline = TableAgentPipeline(
    llm_client=answer_client,
    layout_vlm_client=layout_client,
    config=table_agent_config,
)
```

See `TableAgent/README.md` for architecture, configuration, lifecycle, and artifact
details.

## Tests

```bash
uv run pytest -q
```

