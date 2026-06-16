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
