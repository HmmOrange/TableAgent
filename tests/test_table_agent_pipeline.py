import json
from pathlib import Path

import yaml

from datasets.base import EvalSample
from pipelines.table_agent_pipeline import TableAgentPipeline
from table2img.core import RenderResult
from utils.llm.base import LLMResponse


class FakeLLM:
    model_name = "fake"
    temperature = 0.0

    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.calls.append((prompt, system_prompt))
        if system_prompt and "layout agent" in system_prompt:
            return LLMResponse(
                content=yaml.safe_dump(
                    {
                        "structure": {
                            "table1": {
                                "name": "Revenue",
                                "description": "Company revenue values by year",
                                "headers": [
                                    {
                                        "label": "Revenue",
                                        "description": "Company revenue values by year",
                                        "orientation": "column",
                                        "header_range": "B1:B1",
                                        "data_range": "B2:B2",
                                        "sub_headers": [],
                                    }
                                ],
                            }
                        },
                        "changelog": "Added Revenue header.",
                        "remaining_directions": [],
                    },
                    sort_keys=False,
                ),
                prompt_tokens=10,
                completion_tokens=5,
            )
        if system_prompt and "verification agent" in system_prompt:
            return LLMResponse(
                content="status: good\nfeedback: Structure uses meaningful detected labels.\n",
                prompt_tokens=4,
                completion_tokens=2,
            )
        return LLMResponse(content="100", prompt_tokens=6, completion_tokens=1)


class SuccessfulQALLM(FakeLLM):
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.calls.append((prompt, system_prompt))
        if system_prompt and "verification agent" in system_prompt:
            return LLMResponse(
                content="status: good\nfeedback: Structure uses meaningful detected labels.\n",
                prompt_tokens=4,
                completion_tokens=2,
            )
        if system_prompt and "expert spreadsheet data planner" in system_prompt:
            return LLMResponse(
                content=json.dumps(
                    {
                        "subtasks": [
                            {
                                "id": "inspect_table",
                                "layer": "inspect",
                                "depends_on": [],
                                "description": "Inspect the available table.",
                            },
                            {
                                "id": "synthesize_answer",
                                "layer": "synthesis",
                                "depends_on": ["inspect_table"],
                                "description": "Compute the final answer.",
                            },
                        ]
                    },
                    sort_keys=False,
                ),
                prompt_tokens=7,
                completion_tokens=3,
            )
        if system_prompt and "spreadsheet analysis ReAct agent" in system_prompt:
            return LLMResponse(
                content=json.dumps(
                    {
                        "reasoning": "Inspect a compact preloaded table summary.",
                        "code": "inspected_shape = table_df.shape\nprint(inspected_shape)",
                        "description": "Stores and prints the table shape.",
                    }
                ),
                prompt_tokens=11,
                completion_tokens=4,
            )
        if system_prompt and "spreadsheet synthesis agent" in system_prompt:
            return LLMResponse(
                content=json.dumps(
                    {
                        "reasoning": "The sample answer is directly computed for this fixture.",
                        "code": "final_answer = '100'\nprint(final_answer)",
                        "description": "Sets the final answer.",
                    }
                ),
                prompt_tokens=13,
                completion_tokens=4,
            )
        if system_prompt and "strict table-QA reviewer" in system_prompt:
            return LLMResponse(
                content=json.dumps(
                    {
                        "accepted": True,
                        "score": 1.0,
                        "feedback": "Accepted.",
                    }
                ),
                prompt_tokens=5,
                completion_tokens=2,
            )
        return super().generate(prompt, system_prompt=system_prompt)


class FakeLayoutVLM:
    model_name = "fake-vlm"
    temperature = 0.0

    def __init__(self):
        self.calls = []

    def generate_with_image(self, prompt: str, image_path: Path, system_prompt: str | None = None) -> LLMResponse:
        self.calls.append((prompt, Path(image_path), system_prompt))
        structure = yaml.safe_dump(
            {
                "structure": {
                    "table1": {
                        "name": "Revenue",
                        "description": "Company revenue values by year",
                        "headers": [
                            {
                                "label": "Revenue",
                                "description": "Company revenue values by year",
                                "orientation": "column",
                                "header_range": "B1:B1",
                                "data_range": "B2:B2",
                                "sub_headers": [],
                            }
                        ],
                    }
                },
                "changelog": "Added Revenue header.",
                "remaining_directions": [],
            },
            sort_keys=False,
        )
        return LLMResponse(
            content=f"```yaml\n{structure}```",
            prompt_tokens=10,
            completion_tokens=5,
        )


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def __call__(self, document, image_path, **kwargs):
        self.calls.append((document, Path(image_path), kwargs))
        image_path = Path(image_path)
        image_path.write_bytes(b"fake-image")
        html_path = image_path.with_suffix(".html")
        html_path.write_text(document.html, encoding="utf-8")
        return RenderResult(
            image_path=image_path,
            html_path=html_path,
            width=100,
            height=80,
            browser_path=Path("fake-browser"),
        )


def test_table_agent_writes_verified_structure(tmp_path: Path):
    sample = EvalSample(
        index=0,
        sample_id="sample/1",
        table_id="table-1",
        table_content="Year | Revenue\n2024 | 100",
        question="What is the 2024 revenue?",
        answer=["100"],
    )
    llm = FakeLLM()
    layout_vlm = FakeLayoutVLM()
    renderer = FakeRenderer()
    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=renderer,
    )

    output = pipeline.run(sample)

    assert output.predicted_answer == "100"
    assert output.metadata["verification"]["status"] == "good"
    structure_path = Path(output.metadata["structure_path"])
    assert structure_path.is_file()
    structure = yaml.safe_load(structure_path.read_text(encoding="utf-8"))
    assert structure["table1"]["headers"][0]["label"] == "Revenue"
    assert Path(output.metadata["workbook_path"]).is_file()
    assert Path(output.metadata["image_path"]).is_file()
    assert Path(output.metadata["html_path"]).is_file()
    assert Path(output.metadata["changelog_path"]).is_file()
    assert Path(output.metadata["events_path"]).is_file()
    assert output.metadata["workbook_sheets"] == ["table-1"]
    assert len(renderer.calls) == 1
    assert len(layout_vlm.calls) == 1
    assert layout_vlm.calls[0][1].name == "viewport.png"
    assert output.metadata["qa"]["token_usage"] == {"prompt": 12, "completion": 2}
    assert output.token_usage == {"prompt": 32, "completion": 10}


def test_table_agent_counts_successful_qa_runner_tokens(tmp_path: Path):
    sample = EvalSample(
        index=0,
        sample_id="sample/1",
        table_id="table-1",
        table_content="Year | Revenue\n2024 | 100",
        question="What is the 2024 revenue?",
        answer=["100"],
    )
    pipeline = TableAgentPipeline(
        llm_client=SuccessfulQALLM(),
        layout_vlm_client=FakeLayoutVLM(),
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=FakeRenderer(),
    )

    output = pipeline.run(sample)

    assert output.predicted_answer == "100"
    assert output.metadata["qa"]["success"] is True
    assert output.metadata["qa"]["fallback_used"] is False
    assert output.metadata["qa"]["token_usage"] == {"prompt": 41, "completion": 15}
    assert output.token_usage == {"prompt": 55, "completion": 22}


def test_table_agent_siflex_retrieval(tmp_path: Path):
    import openpyxl
    # Create two dummy workbooks
    wb1 = openpyxl.Workbook()
    ws1 = wb1.active
    ws1.title = "SheetA"
    ws1["A1"] = "Apple"
    ws1["B1"] = "Banana"
    path_a = tmp_path / "doc_a.xlsx"
    wb1.save(path_a)

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "SheetB"
    ws2["A1"] = "Orange"
    ws2["B1"] = "Grape"
    path_b = tmp_path / "doc_b.xlsx"
    wb2.save(path_b)

    sample = EvalSample(
        index=0,
        sample_id="siflex/1",
        table_id="siflex-table",
        table_content="Apple | Banana\nOrange | Grape",
        question="Which sheet has the Apple and Banana?",
        answer=["SheetA"],
        sample_path="data/SiFlex/golden_tests/compiled/golden_cases.json:cases[0]",
        table_path=f"{path_a};{path_b}",
    )

    llm = FakeLLM()
    # Mock generate_with_image on LLM for XLSX answering flow
    llm.generate_with_image = lambda prompt, image_path, system_prompt=None: LLMResponse(
        content="SheetA", prompt_tokens=15, completion_tokens=3
    )

    layout_vlm = FakeLayoutVLM()
    renderer = FakeRenderer()

    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=renderer,
    )

    # 1. Pre-encode sheets
    pipeline.prepare_samples([sample])

    # Verify that structures/sources are generated
    sources_dir = tmp_path / "sources"
    assert sources_dir.is_dir()
    assert (sources_dir / "doc_a.xlsx_SheetA" / "structure.yaml").is_file()
    assert (sources_dir / "doc_a.xlsx_SheetA" / "sheet_text.txt").is_file()
    assert (sources_dir / "doc_b.xlsx_SheetB" / "structure.yaml").is_file()
    assert (sources_dir / "doc_b.xlsx_SheetB" / "sheet_text.txt").is_file()

    # 2. Run the question answering flow
    output = pipeline.run(sample)

    assert output.predicted_answer == "SheetA"
    assert output.metadata["workbook_path"] == str(path_a.resolve())
    assert output.metadata["workbook_sheets"] == ["SheetA"]
    assert "doc_a.xlsx_SheetA/table.png" in output.metadata["image_path"]


def test_table_agent_run_prepares_siflex_source_lazily(tmp_path: Path):
    import openpyxl

    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "SheetA"
    worksheet["A1"] = "Apple"
    workbook_path = tmp_path / "doc_a.xlsx"
    workbook.save(workbook_path)

    sample = EvalSample(
        index=0,
        sample_id="siflex/lazy",
        table_id="siflex-table",
        table_content="Apple",
        question="Which sheet has Apple?",
        answer=["SheetA"],
        sample_path="data/SiFlex/golden_tests/compiled/golden_cases.json:cases[0]",
        table_path=str(workbook_path),
    )

    llm = FakeLLM()
    llm.generate_with_image = lambda prompt, image_path, system_prompt=None: LLMResponse(
        content="SheetA", prompt_tokens=15, completion_tokens=3
    )
    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=FakeLayoutVLM(),
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=FakeRenderer(),
    )

    assert pipeline.prepare_samples_before_run is False
    assert not (tmp_path / "sources").exists()

    output = pipeline.run(sample)

    assert output.predicted_answer == "SheetA"
    assert (tmp_path / "sources" / "doc_a.xlsx_SheetA" / "structure.yaml").is_file()
    assert output.metadata["workbook_path"] == str(workbook_path.resolve())


def test_table_agent_prepare_samples_regenerates_invalid_structure(tmp_path: Path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SheetX"
    ws["A1"] = "Data"
    path_x = tmp_path / "doc_x.xlsx"
    wb.save(path_x)

    sample = EvalSample(
        index=0,
        sample_id="siflex/invalid_test",
        table_id="siflex-table-invalid",
        table_content="Data",
        question="What is Data?",
        answer=["Data"],
        sample_path="data/SiFlex/golden_tests/compiled/golden_cases.json:cases[0]",
        table_path=str(path_x),
    )

    llm = FakeLLM()
    layout_vlm = FakeLayoutVLM()
    renderer = FakeRenderer()

    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=renderer,
    )

    # Pre-create the directory and put an invalid structure.yaml in there
    sheet_dir = tmp_path / "sources" / "doc_x.xlsx_SheetX"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    invalid_structure_path = sheet_dir / "structure.yaml"
    invalid_structure_path.write_text("ERROR: Connection error.", encoding="utf-8")

    # Run prepare_samples
    pipeline.prepare_samples([sample])

    # Assert that structure.yaml was regenerated and contains valid headers YAML
    assert invalid_structure_path.is_file()
    content = invalid_structure_path.read_text(encoding="utf-8")
    assert "ERROR" not in content
    assert "headers" in content


def test_table_agent_run_ignores_invalid_structures(tmp_path: Path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SheetY"
    ws["A1"] = "Apple"
    path_y = tmp_path / "doc_y.xlsx"
    wb.save(path_y)

    sample = EvalSample(
        index=0,
        sample_id="siflex/ignore_test",
        table_id="siflex-table-ignore",
        table_content="Apple",
        question="Which sheet?",
        answer=["SheetY"],
        sample_path="data/SiFlex/golden_tests/compiled/golden_cases.json:cases[0]",
        table_path=str(path_y),
    )

    llm = FakeLLM()
    layout_vlm = FakeLayoutVLM()
    renderer = FakeRenderer()

    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=renderer,
    )

    # 1. Pre-encode sheets
    pipeline.prepare_samples([sample])

    # 2. Overwrite structure.yaml with connection error
    structure_path = tmp_path / "sources" / "doc_y.xlsx_SheetY" / "structure.yaml"
    assert structure_path.is_file()
    structure_path.write_text("ERROR: Connection error.", encoding="utf-8")

    # 3. Running should ignore the candidate directory because of the invalid structure.yaml,
    # and fall back. Since the fallback path executes, we verify it falls through.
    output = pipeline.run(sample)

    assert output.metadata["verification"]["feedback"] != "Retrieved from encoded source"


def test_table_agent_fallback_invalid_structure_marked_not_good(tmp_path: Path):
    sample = EvalSample(
        index=0,
        sample_id="siflex/fallback_error",
        table_id="siflex-table-err",
        table_content="Apple",
        question="Which sheet?",
        answer=["SheetY"],
    )

    class ErrorLLM:
        model_name = "error-llm"
        temperature = 0.0
        def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
            return LLMResponse(content="ERROR: Connection error.", prompt_tokens=0, completion_tokens=0)

    class ErrorLayoutVLM:
        model_name = "error-vlm"
        temperature = 0.0
        def generate_with_image(self, prompt: str, image_path: Path, system_prompt: str | None = None) -> LLMResponse:
            return LLMResponse(content="ERROR: Connection error.", prompt_tokens=0, completion_tokens=0)

    renderer = FakeRenderer()
    pipeline = TableAgentPipeline(
        llm_client=ErrorLLM(),
        layout_vlm_client=ErrorLayoutVLM(),
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=renderer,
    )

    output = pipeline.run(sample)
    assert output.metadata["verification"]["status"] == "not_good"
    assert "invalid" in output.metadata["verification"]["feedback"].lower() or "empty" in output.metadata["verification"]["feedback"].lower()


def test_table_agent_siflex_answer_prompt_formatting(tmp_path: Path):
    pipeline = TableAgentPipeline(
        llm_client=FakeLLM(),
        layout_vlm_client=FakeLayoutVLM(),
        config={"artifact_dir": str(tmp_path)},
        renderer=FakeRenderer(),
    )

    # 1. Non-SiFlex sample
    non_siflex_sample = EvalSample(
        index=0,
        sample_id="hitab/1",
        table_id="t1",
        table_content="Content",
        question="What is the answer?",
        answer=["Ans"],
        sample_path="data/HiTab/split.json:cases[0]",
    )
    prompt = pipeline._answer_prompt(non_siflex_sample, "Content", "headers: []")
    assert "FORMAT INSTRUCTIONS" not in prompt
    assert "Verified structure.yaml" in prompt

    # 2. SiFlex table sample
    siflex_table_sample = EvalSample(
        index=0,
        sample_id="siflex/1",
        table_id="t1",
        table_content="Content",
        question="What is the answer?",
        answer=["Ans"],
        sample_path="data/SiFlex/compiled/golden_cases.json:cases[0]",
        raw={"answer_type": "table"},
    )
    prompt = pipeline._answer_prompt(siflex_table_sample, "Content", "headers: []")
    assert "FORMAT INSTRUCTIONS" in prompt
    assert "CRITICAL EXPECTED FORMAT: TABLE" in prompt
    assert "Format your final answer as a markdown table" in prompt

    # 3. SiFlex list sample
    siflex_list_sample = EvalSample(
        index=0,
        sample_id="siflex/2",
        table_id="t1",
        table_content="Content",
        question="What is the answer?",
        answer=["Ans"],
        sample_path="data/SiFlex/compiled/golden_cases.json:cases[1]",
        raw={"answer_type": "list"},
    )
    prompt = pipeline._answer_prompt(siflex_list_sample, "Content", "headers: []")
    assert "FORMAT INSTRUCTIONS" in prompt
    assert "CRITICAL EXPECTED FORMAT: LIST" in prompt
    assert "Format your final answer as a bulleted list" in prompt

    # 4. SiFlex form sample
    siflex_form_sample = EvalSample(
        index=0,
        sample_id="siflex/3",
        table_id="t1",
        table_content="Content",
        question="What is the answer?",
        answer=["Ans"],
        sample_path="data/SiFlex/compiled/golden_cases.json:cases[2]",
        raw={"answer_type": "form"},
    )
    prompt = pipeline._answer_prompt(siflex_form_sample, "Content", "headers: []")
    assert "FORMAT INSTRUCTIONS" in prompt
    assert "CRITICAL EXPECTED FORMAT: FORM/DOCUMENT" in prompt
    assert "Organize your final answer in a clear document structure" in prompt


def test_table_agent_default_applies_generation_cap_and_early_breaks(tmp_path: Path):
    llm = FakeLLM()
    llm.max_tokens = None
    layout_vlm = FakeLayoutVLM()
    layout_vlm.max_tokens = 4096
    
    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={"artifact_dir": str(tmp_path)},
        renderer=FakeRenderer(),
    )
    assert llm.max_tokens == 2048
    assert layout_vlm.max_tokens == 2048

    # Test fit_context truncation
    long_content = "X" * 70000
    fitted = pipeline._fit_context(long_content)
    assert len(fitted) <= 60000 + 50
    assert "...TRUNCATED..." in fitted


def test_table_agent_max_tokens_config_driven(tmp_path: Path):
    llm = FakeLLM()
    llm.max_tokens = 100
    layout_vlm = FakeLayoutVLM()
    layout_vlm.max_tokens = 200
    
    # Config overrides default cap
    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={"artifact_dir": str(tmp_path), "generation_max_tokens": 1024},
        renderer=FakeRenderer(),
    )
    assert llm.max_tokens == 1024
    assert layout_vlm.max_tokens == 1024
    assert pipeline.get_config()["agent"]["generation_max_tokens"] == 1024

    # Config None disables capping/mutating
    llm.max_tokens = 500
    layout_vlm.max_tokens = 600
    pipeline_none = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={"artifact_dir": str(tmp_path), "generation_max_tokens": None},
        renderer=FakeRenderer(),
    )
    assert llm.max_tokens == 500
    assert layout_vlm.max_tokens == 600
    assert pipeline_none.get_config()["agent"]["generation_max_tokens"] is None


def test_is_valid_structure_rules():
    from pipelines.table_agent_pipeline import _is_valid_structure
    
    # 1. Reject structures with 'error' key
    assert not _is_valid_structure("headers: []\nerror: Failed to generate structure")
    assert not _is_valid_structure("error: Failed to generate structure")
    
    # 2. Reject empty headers
    assert not _is_valid_structure("headers: []")
    
    # 3. Reject structures with only placeholder/vague headers
    assert not _is_valid_structure("headers:\n  - Column 1\n  - Column 2")
    assert not _is_valid_structure("headers:\n  - col1\n  - col 2\n  - Placeholder")
    assert not _is_valid_structure("headers:\n  - -")
    assert not _is_valid_structure("headers:\n  - Empty\n  - None")
    
    # 4. Accept if at least one header is not placeholder
    assert _is_valid_structure("headers:\n  - Column 1\n  - Revenue\n  - Column 2")
    assert _is_valid_structure("headers:\n  - label: Column 1\n  - label: Total Revenue")
    assert _is_valid_structure("headers:\n  - name: Column 1\n  - name: Net Profit")


def test_table_agent_mas_prompts_assign_nulling_to_verifier():
    from TableAgent.prompts import (
        LAYOUT_MAS_SYSTEM_PROMPT,
        LAYOUT_MAS_USER_PROMPT_TEMPLATE,
        VERIFICATION_MAS_SYSTEM_PROMPT,
        VERIFICATION_MAS_USER_PROMPT_TEMPLATE,
    )

    assert "never output null, UNKNOWN, or placeholder range values" in LAYOUT_MAS_SYSTEM_PROMPT
    assert "header_range` is only the cell or merged/spanned cells" in LAYOUT_MAS_USER_PROMPT_TEMPLATE
    assert "data starts below all header and sub-header rows" in LAYOUT_MAS_USER_PROMPT_TEMPLATE
    assert "must not include child `header_range` cells" in LAYOUT_MAS_USER_PROMPT_TEMPLATE
    assert "Never write `null`, `UNKNOWN`, `N/A`, or placeholder range values" in LAYOUT_MAS_USER_PROMPT_TEMPLATE
    assert "use the union of the old range and newly visible" in LAYOUT_MAS_USER_PROMPT_TEMPLATE
    assert "Do not create separate headers for blank cells inside a merged" in LAYOUT_MAS_USER_PROMPT_TEMPLATE
    assert "null_fields" in VERIFICATION_MAS_SYSTEM_PROMPT
    assert "ReAct pattern" in VERIFICATION_MAS_SYSTEM_PROMPT
    assert "updated_structure" in VERIFICATION_MAS_SYSTEM_PROMPT
    assert "tool failure" in VERIFICATION_MAS_USER_PROMPT_TEMPLATE


def test_strict_structure_normalizes_uncertain_ranges_to_null():
    from pipelines.table_agent_pipeline import extract_strict_structure

    structure_text, _ = extract_strict_structure(
        "headers:\n"
        "- label: Revenue\n"
        "  range: UNKNOWN\n"
        "  sub_headers:\n"
        "  - label: 2024\n"
        "    range: uncertain\n"
    )
    structure = yaml.safe_load(structure_text)

    assert structure["headers"][0]["range"] is None
    assert structure["headers"][0]["sub_headers"][0]["range"] is None


def test_strict_structure_extraction_discards_reasoning_and_extra_keys():
    from pipelines.table_agent_pipeline import extract_strict_structure

    content = """I analyzed the table before producing the result.
```yaml
reasoning: this must never be persisted
headers:
  - label: Revenue
    description: Annual revenue
    orientation: column
    range: B1:B3
    confidence: 0.9
    sub_headers:
      - label: "2024"
        description: Fiscal year 2024
        orientation: column
        range: B2
```
This trailing explanation is also logging-only."""

    structure_text, discarded = extract_strict_structure(content)
    structure = yaml.safe_load(structure_text)

    assert set(structure) == {"headers"}
    assert set(structure["headers"][0]) == {"label", "description", "orientation", "range", "sub_headers"}
    assert set(structure["headers"][0]["sub_headers"][0]) == {"label", "description", "orientation", "range"}
    assert "reasoning" not in structure_text
    assert "confidence" not in structure_text
    assert "analyzed the table" in discarded
    assert "trailing explanation" in discarded
    assert "reasoning" in discarded
    assert "confidence" in discarded


def test_table_agent_persists_only_strict_structure(tmp_path: Path):
    class ReasoningLayoutVLM(FakeLayoutVLM):
        def generate_with_image(self, prompt: str, image_path: Path, system_prompt: str | None = None) -> LLMResponse:
            return LLMResponse(
                content="""Analysis that belongs in logs.
```yaml
structure:
  table1:
    name: Revenue
    description: Annual revenue
    headers:
      - label: Revenue
        description: Annual revenue
        orientation: column
        header_range: B1:B1
        data_range: B2:B2
        sub_headers: []
changelog: Added revenue.
remaining_directions: []
```""",
                prompt_tokens=10,
                completion_tokens=5,
            )

    sample = EvalSample(
        index=0,
        sample_id="strict/1",
        table_id="table-1",
        table_content="Year | Revenue\n2024 | 100",
        question="What is the revenue?",
        answer=["100"],
    )
    pipeline = TableAgentPipeline(
        llm_client=FakeLLM(),
        layout_vlm_client=ReasoningLayoutVLM(),
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 0},
        renderer=FakeRenderer(),
    )

    output = pipeline.run(sample)
    persisted = Path(output.metadata["structure_path"]).read_text(encoding="utf-8")

    assert persisted.startswith("table1:")
    assert "Analysis that belongs in logs" not in persisted


def test_table_agent_separates_artifacts_by_benchmark_repeat(tmp_path: Path):
    from TableAgent.config import run_scoped_table_agent_config

    scoped_config = run_scoped_table_agent_config(
        {
            "table_agent": {
                "artifact_root": str(tmp_path),
                "run_dir_template": "{run_name}",
                "repeat_dir_template": "repeat_{run_id}",
                "shared_dir_name": "shared",
            }
        },
        "hitab-table_agent-20260621_163851",
    )
    pipeline = TableAgentPipeline(
        llm_client=FakeLLM(),
        layout_vlm_client=FakeLayoutVLM(),
        config=scoped_config,
        renderer=FakeRenderer(),
    )
    sample = EvalSample(
        index=0,
        sample_id="sample/1",
        table_id="table-1",
        table_content="Year | Revenue\n2024 | 100",
        question="What is the revenue?",
        answer=["100"],
    )

    pipeline.set_run_id(1)
    first_output = pipeline.run(sample)
    pipeline.set_run_id(3)
    third_output = pipeline.run(sample)

    assert "repeat_1" in Path(first_output.metadata["structure_path"]).parts
    assert "repeat_3" in Path(third_output.metadata["structure_path"]).parts
    assert Path(first_output.metadata["structure_path"]) != Path(third_output.metadata["structure_path"])
    assert pipeline.settings.source_artifact_dir == tmp_path / "hitab-table_agent-20260621_163851" / "shared"


def test_table_agent_default_outputs_and_logs_are_scoped_under_table_agent():
    from TableAgent.config import resolve_table_agent_run_roots

    config = {
        "table_agent": {
            "evaluation_output_dir": "TableAgent/outputs/evaluations",
            "log_dir": "TableAgent/outputs/logs",
        }
    }

    output_dir, log_dir = resolve_table_agent_run_roots("table_agent", "outputs", config)

    assert output_dir == Path("TableAgent/outputs/evaluations")
    assert log_dir == Path("TableAgent/outputs/logs")
    assert resolve_table_agent_run_roots("table_agent", "custom", config)[0] == Path("custom")
    assert resolve_table_agent_run_roots("graphotter", "outputs", config) == (Path("outputs"), Path("logs"))


def test_table_agent_retrieval_rejects_placeholder_structures(tmp_path: Path):
    import openpyxl
    # Create two dummy workbooks
    wb1 = openpyxl.Workbook()
    ws1 = wb1.active
    ws1.title = "SheetA"
    ws1["A1"] = "Apple"
    path_a = tmp_path / "doc_a.xlsx"
    wb1.save(path_a)

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "SheetB"
    ws2["A1"] = "Orange"
    path_b = tmp_path / "doc_b.xlsx"
    wb2.save(path_b)

    sample = EvalSample(
        index=0,
        sample_id="siflex/1",
        table_id="siflex-table",
        table_content="Apple | Banana\nOrange | Grape",
        question="Which sheet has the Apple and Banana?",
        answer=["SheetA"],
        sample_path="data/SiFlex/golden_tests/compiled/golden_cases.json:cases[0]",
        table_path=f"{path_a};{path_b}",
    )

    llm = FakeLLM()
    llm.generate_with_image = lambda prompt, image_path, system_prompt=None: LLMResponse(
        content="SheetA", prompt_tokens=15, completion_tokens=3
    )

    layout_vlm = FakeLayoutVLM()
    renderer = FakeRenderer()

    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=renderer,
    )

    # Pre-encode sheets
    pipeline.prepare_samples([sample])

    # Now manually overwrite structure.yaml of SheetA and SheetB with placeholder/error/invalid structures
    sources_dir = tmp_path / "sources"
    struct_a_path = sources_dir / "doc_a.xlsx_SheetA" / "structure.yaml"
    struct_b_path = sources_dir / "doc_b.xlsx_SheetB" / "structure.yaml"
    
    # Overwrite both structure.yaml with placeholder structures
    struct_a_path.write_text("headers: []\nerror: Failed to generate structure", encoding="utf-8")
    struct_b_path.write_text("headers: []\nerror: Failed to generate structure", encoding="utf-8")

    # Run the pipeline - it should skip both since their structures are invalid (rejected)
    output = pipeline.run(sample)
    
    # Verify that it didn't retrieve either (status won't be good or verification metadata won't say retrieved)
    assert output.metadata["verification"]["feedback"] != "Retrieved from encoded source"


def test_table_agent_prepare_samples_uses_error_sidecar(tmp_path: Path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SheetZ"
    ws["A1"] = "Data"
    path_z = tmp_path / "doc_z.xlsx"
    wb.save(path_z)

    sample = EvalSample(
        index=0,
        sample_id="siflex/sidecar_test",
        table_id="siflex-table-sidecar",
        table_content="Data",
        question="What is Data?",
        answer=["Data"],
        sample_path="data/SiFlex/golden_tests/compiled/golden_cases.json:cases[0]",
        table_path=str(path_z),
    )

    # Mock VLM layout client that returns invalid/empty layout to simulate failure
    class ErrorLayoutVLM:
        model_name = "error-vlm"
        temperature = 0.0
        def generate_with_image(self, prompt: str, image_path: Path, system_prompt: str | None = None) -> LLMResponse:
            return LLMResponse(content="", prompt_tokens=0, completion_tokens=0)

    llm = FakeLLM()
    renderer = FakeRenderer()

    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=ErrorLayoutVLM(),
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=renderer,
    )

    # First prepare_samples call: should attempt generation, fail (VLM returns empty),
    # write structure.error, and not write structure.yaml
    pipeline.prepare_samples([sample])

    sheet_dir = tmp_path / "sources" / "doc_z.xlsx_SheetZ"
    error_path = sheet_dir / "structure.error"
    struct_path = sheet_dir / "structure.yaml"

    assert error_path.is_file()
    assert not struct_path.is_file()

    # Now let's mock a layout VLM that succeeds
    class SuccessLayoutVLM:
        model_name = "success-vlm"
        temperature = 0.0
        def generate_with_image(self, prompt: str, image_path: Path, system_prompt: str | None = None) -> LLMResponse:
            # Return a valid structure
            return LLMResponse(
                content="headers:\n  - label: SuccessHeader",
                prompt_tokens=5,
                completion_tokens=5,
            )

    pipeline_success = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=SuccessLayoutVLM(),
        config={"artifact_dir": str(tmp_path), "max_refinement_rounds": 1},
        renderer=renderer,
    )

    # Run prepare_samples again. Since structure.error exists, it should NOT regenerate,
    # and structure.yaml should still NOT exist.
    pipeline_success.prepare_samples([sample])
    assert not struct_path.is_file()

    # If we delete structure.error, then running prepare_samples should regenerate and succeed
    error_path.unlink()
    pipeline_success.prepare_samples([sample])
    assert struct_path.is_file()
    assert "SuccessHeader" in struct_path.read_text(encoding="utf-8")


def test_table_agent_image_dimension_config_and_render_kwargs(tmp_path: Path):
    sample = EvalSample(
        index=0,
        sample_id="sample/1",
        table_id="table-1",
        table_content="Year | Revenue\n2024 | 100",
        question="What is the 2024 revenue?",
        answer=["100"],
    )
    llm = FakeLLM()
    layout_vlm = FakeLayoutVLM()
    renderer = FakeRenderer()
    
    # Configure with small max_image_dimension to trigger scaling down
    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={
            "artifact_dir": str(tmp_path),
            "max_refinement_rounds": 1,
            "image_scale": 2.0,
            "max_image_dimension": 200,
            "max_viewport_width": 1000,
            "max_viewport_height": 800,
        },
        renderer=renderer,
    )
    
    # Expose configs check
    cfg = pipeline.get_config()
    assert cfg["agent"]["max_image_dimension"] == 200
    assert cfg["agent"]["max_viewport_width"] == 1000
    assert cfg["agent"]["max_viewport_height"] == 800
    
    # Run the pipeline
    output = pipeline.run(sample)
    
    # Verify renderer calls parameters
    assert len(renderer.calls) == 1
    doc, path, kwargs = renderer.calls[0]
    assert kwargs["max_viewport_width"] == 1000
    assert kwargs["max_viewport_height"] == 800


def test_table_agent_image_fitting_and_tiling(tmp_path: Path):
    from PIL import Image
    from pipelines.table_agent_pipeline import (
        compute_viewport_and_scale,
        _generate_image_tiles,
        _resize_image_file_to_fit,
    )

    # 1. Test compute_viewport_and_scale
    # Case A: dimension limits
    vw, vh, scale = compute_viewport_and_scale(
        estimated_width=1000,
        estimated_height=1000,
        image_scale=2.0,
        max_viewport_width=800,
        max_viewport_height=800,
        max_image_dimension=400,
        max_image_pixels=None,
    )
    # vw/vh are limited to 800x800. max(800, 800) * 2.0 = 1600 > 400.
    # Scale should become 400 / 800 = 0.5
    assert vw == 800
    assert vh == 800
    assert abs(scale - 0.5) < 1e-5

    # Case B: pixel limits
    vw, vh, scale = compute_viewport_and_scale(
        estimated_width=1000,
        estimated_height=1000,
        image_scale=2.0,
        max_viewport_width=800,
        max_viewport_height=800,
        max_image_dimension=None,
        max_image_pixels=10000,
    )
    # vw/vh are 800x800. Total base pixels = 640000.
    # Total pixels at scale=2.0 would be 640000 * 4 = 2.56M > 10000.
    # Scale should become (10000 / 640000) ** 0.5 = (1/64) ** 0.5 = 0.125
    assert abs(scale - 0.125) < 1e-5

    # 2. Test _generate_image_tiles and _resize_image_file_to_fit on a large fake image
    large_img_path = tmp_path / "large_table.png"
    # Create a dummy image of size 1200 x 800
    img = Image.new("RGB", (1200, 800), color="white")
    img.save(large_img_path)

    # Slice into tiles of size 500 with 100 overlap
    tiles = _generate_image_tiles(large_img_path, tile_size=500, overlap=100)
    assert len(tiles) == 6
    for tile in tiles:
        tile_file = tmp_path / tile["filename"]
        assert tile_file.is_file()
        with Image.open(tile_file) as t_img:
            assert t_img.width <= 500
            assert t_img.height <= 500

    # Resize the large image to max_dim 600 and max_pixels 200000
    _resize_image_file_to_fit(large_img_path, max_dim=600, max_pixels=200000)
    with Image.open(large_img_path) as resized_img:
        assert resized_img.width <= 600
        assert resized_img.height <= 400
        assert resized_img.width * resized_img.height <= 200000

    # 3. Test decompression bomb safety
    bomb_path = tmp_path / "bomb.png"
    try:
        # Create a 10000x9000 (90MP) image in mode "1" (binary, keeps RAM tiny ~11MB)
        bomb_img = Image.new("1", (10000, 9000), color=0)
        bomb_img.save(bomb_path)
        
        # Verify opening it does not crash/throw with our bypass, and it successfully resizes to max_dim 1000
        _resize_image_file_to_fit(bomb_path, max_dim=1000)
        assert bomb_path.is_file()
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(bomb_path) as opened_bomb:
            assert opened_bomb.width == 1000
    except MemoryError:
        pass


def test_table_agent_llm_reranker(tmp_path: Path):
    import openpyxl
    # Create two dummy workbooks: doc_a has low lexical score, doc_b has high lexical score
    wb1 = openpyxl.Workbook()
    ws1 = wb1.active
    ws1.title = "LowLexical"
    ws1["A1"] = "Cherry"
    path_a = tmp_path / "doc_a.xlsx"
    wb1.save(path_a)

    wb2 = openpyxl.Workbook()
    ws2 = wb2.active
    ws2.title = "HighLexical"
    ws2["A1"] = "Apple"
    path_b = tmp_path / "doc_b.xlsx"
    wb2.save(path_b)

    sample = EvalSample(
        index=0,
        sample_id="siflex/rerank_test",
        table_id="siflex-table",
        table_content="Apple | Cherry",
        question="Which sheet has the Apple?",
        answer=["SomeSecretGoldAnswer"],  # gold answer is distinct secret string
        sample_path="data/SiFlex/golden_tests/compiled/golden_cases.json:cases[0]",
        table_path=f"{path_a};{path_b}",
    )

    class CustomFakeLLM:
        model_name = "custom-fake"
        temperature = 0.0
        def __init__(self):
            self.calls = []
            self.generate_content = "selected_index: 1\nrationale: LLM chose LowLexical"
        
        def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
            self.calls.append((prompt, system_prompt))
            if system_prompt and "selection agent" in system_prompt:
                return LLMResponse(content=self.generate_content, prompt_tokens=100, completion_tokens=10)
            return LLMResponse(content="LowLexical", prompt_tokens=15, completion_tokens=3)

    llm = CustomFakeLLM()
    # Mock generate_with_image for answering
    llm.generate_with_image = lambda prompt, image_path, system_prompt=None: LLMResponse(
        content="LowLexical", prompt_tokens=15, completion_tokens=3
    )

    layout_vlm = FakeLayoutVLM()
    renderer = FakeRenderer()

    pipeline = TableAgentPipeline(
        llm_client=llm,
        layout_vlm_client=layout_vlm,
        config={
            "artifact_dir": str(tmp_path),
            "max_refinement_rounds": 1,
            "retrieval_rerank_with_llm": True,
            "retrieval_top_k": 2,
        },
        renderer=renderer,
    )

    # 1. Pre-encode sheets
    pipeline.prepare_samples([sample])

    # 2. Run the QA flow with LLM reranking choosing the lower lexical candidate
    output = pipeline.run(sample)
    
    # Verify we selected LowLexical (from path_a)
    assert output.metadata["workbook_sheets"] == ["LowLexical"]
    assert output.metadata["workbook_path"] == str(path_a.resolve())
    assert output.metadata["retrieval_info"]["reranker_selected_index"] == 1
    assert output.metadata["retrieval_info"]["fallback_used"] is False
    assert output.metadata["retrieval_info"]["reranker_rationale"] == "LLM chose LowLexical"
    
    # Verify the prompt does not contain the gold answer "SomeSecretGoldAnswer"
    rerank_call = [call for call in llm.calls if call[1] and "selection agent" in call[1]]
    assert len(rerank_call) == 1
    rerank_prompt = rerank_call[0][0]
    assert "SomeSecretGoldAnswer" not in rerank_prompt
    assert "gold" not in rerank_prompt

    # Test 2: Invalid/empty reranker output falls back to lexical best (HighLexical/doc_b) and does not crash.
    llm.calls.clear()
    llm.generate_content = "invalid yaml here: index: 9999"
    llm.generate_with_image = lambda prompt, image_path, system_prompt=None: LLMResponse(
        content="HighLexical", prompt_tokens=15, completion_tokens=3
    )
    
    output2 = pipeline.run(sample)
    assert output2.metadata["workbook_sheets"] == ["HighLexical"]
    assert output2.metadata["workbook_path"] == str(path_b.resolve())
    assert output2.metadata["retrieval_info"]["fallback_used"] is True





