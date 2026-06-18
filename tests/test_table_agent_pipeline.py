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
                        "headers": [
                            {
                                "label": "Revenue",
                                "description": "Company revenue values by year",
                                "orientation": "column",
                                "range": "B1:B3",
                                "sub_headers": [
                                    {
                                        "label": "2024",
                                        "description": "Revenue for fiscal year 2024",
                                        "orientation": "column",
                                        "range": "B2:B2",
                                    }
                                ],
                            }
                        ]
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


class FakeLayoutVLM:
    model_name = "fake-vlm"
    temperature = 0.0

    def __init__(self):
        self.calls = []

    def generate_with_image(self, prompt: str, image_path: Path, system_prompt: str | None = None) -> LLMResponse:
        self.calls.append((prompt, Path(image_path), system_prompt))
        return LLMResponse(
            content=yaml.safe_dump(
                {
                    "headers": [
                        {
                            "label": "Revenue",
                            "description": "Company revenue values by year",
                            "orientation": "column",
                            "range": "B1:B3",
                            "sub_headers": [
                                {
                                    "label": "2024",
                                    "description": "Revenue for fiscal year 2024",
                                    "orientation": "column",
                                    "range": "B2:B2",
                                }
                            ],
                        }
                    ]
                },
                sort_keys=False,
            ),
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
    assert structure["headers"][0]["label"] == "Revenue"
    assert Path(output.metadata["workbook_path"]).is_file()
    assert Path(output.metadata["image_path"]).is_file()
    assert Path(output.metadata["html_path"]).is_file()
    assert output.metadata["workbook_sheets"] == ["table-1"]
    assert len(renderer.calls) == 1
    assert len(layout_vlm.calls) == 1
    assert layout_vlm.calls[0][1] == Path(output.metadata["image_path"])
    assert output.token_usage == {"prompt": 20, "completion": 8}


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



