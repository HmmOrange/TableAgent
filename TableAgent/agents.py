from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prompts.table_agent import (
    LAYOUT_MAS_SYSTEM_PROMPT,
    LAYOUT_MAS_USER_PROMPT_TEMPLATE,
    VERIFICATION_MAS_SYSTEM_PROMPT,
    VERIFICATION_MAS_USER_PROMPT_TEMPLATE,
)
from utils.llm.base import BaseLLM, LLMResponse

from TableAgent.perception.structure import (
    _is_valid_structure,
    _parse_yaml_mapping,
    extract_layout_structure,
)


@dataclass(frozen=True)
class AgentMessage:
    sent_from: str
    sent_to: str
    content: str
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMemory:
    messages: list[AgentMessage] = field(default_factory=list)

    def add(self, message: AgentMessage) -> None:
        self.messages.append(message)


class BaseTableAgent:
    name = "Agent"
    profile = ""
    goal = ""

    def __init__(self):
        self.memory = AgentMemory()

    def remember(self, message: AgentMessage) -> None:
        self.memory.add(message)


@dataclass(frozen=True)
class LayoutResult:
    structure_text: str
    changelog: str
    directions: list[str]
    changed: bool
    response: LLMResponse
    discarded: str


class LayoutAgent(BaseTableAgent):
    name = "LayoutAgent"
    profile = "Spreadsheet layout VLM"
    goal = "Incrementally extract table headers and data ranges from coordinate-labelled viewports."

    def __init__(self, vlm: BaseLLM):
        super().__init__()
        self.vlm = vlm

    def run(
        self,
        *,
        metadata_text: str,
        structure_text: str,
        image_path: Path,
        viewport_range: str,
        direction: str,
        feedback: str,
        iteration: int,
        iteration_dir: Path,
    ) -> LayoutResult:
        feedback_block = f"\nVerificationAgent feedback:\n{feedback}\n" if feedback else ""
        prompt = LAYOUT_MAS_USER_PROMPT_TEMPLATE.format(
            metadata_text=metadata_text,
            viewport_range=viewport_range,
            direction=direction,
            structure_text=structure_text or "{}",
            feedback_block=feedback_block,
        )
        iteration_dir.joinpath("layout_prompt.txt").write_text(prompt, encoding="utf-8")
        response = self.vlm.generate_with_image(
            prompt=prompt,
            image_path=image_path,
            system_prompt=LAYOUT_MAS_SYSTEM_PROMPT,
        )
        iteration_dir.joinpath("layout_response.txt").write_text(response.content, encoding="utf-8")
        updated, discarded, directions, model_changelog = extract_layout_structure(response.content)
        if not _is_valid_structure(updated):
            updated = structure_text
        changed = bool(updated.strip()) and _canonical_yaml(updated) != _canonical_yaml(structure_text)
        changelog = model_changelog or ("Structure updated." if changed else "No change.")
        if not changed:
            changelog = "No change."
        self.remember(AgentMessage(
            sent_from=self.name,
            sent_to="VerificationAgent",
            content=changelog,
            iteration=iteration,
            metadata={"viewport": viewport_range, "directions": directions, "changed": changed},
        ))
        return LayoutResult(updated, changelog, directions, changed, response, discarded)


@dataclass(frozen=True)
class VerificationResult:
    status: str
    feedback: str
    null_fields: list[str]
    report: dict[str, Any]
    response: LLMResponse

    @property
    def is_good(self) -> bool:
        return self.status == "good"


class VerificationAgent(BaseTableAgent):
    name = "VerificationAgent"
    profile = "Spreadsheet structure verifier"
    goal = "Verify header and data ranges with executable checks and semantic review."

    def __init__(self, llm: BaseLLM):
        super().__init__()
        self.llm = llm

    def run(
        self,
        *,
        workbook_path: Path,
        sheet_name: str,
        metadata_text: str,
        structure_text: str,
        changelog: str,
        viewport_range: str,
        iteration: int,
        iteration_dir: Path,
    ) -> VerificationResult:
        verifier_path = iteration_dir / "verification.py"
        verifier_path.write_text(_VERIFIER_CODE, encoding="utf-8")
        report = _execute_verifier(verifier_path, workbook_path, sheet_name, iteration_dir / "structure_after.yaml")
        if not _is_valid_structure(structure_text):
            report = {
                "status": "not_good",
                "errors": ["Candidate structure is empty or invalid."],
            }
        report_text = json.dumps(report, ensure_ascii=False, indent=2)
        (iteration_dir / "verification_output.json").write_text(report_text, encoding="utf-8")
        prompt = VERIFICATION_MAS_USER_PROMPT_TEMPLATE.format(
            metadata_text=metadata_text,
            viewport_range=viewport_range,
            structure_text=structure_text,
            changelog=changelog,
            verification_report=report_text,
        )
        (iteration_dir / "verification_prompt.txt").write_text(prompt, encoding="utf-8")
        response = self.llm.generate(prompt=prompt, system_prompt=VERIFICATION_MAS_SYSTEM_PROMPT)
        (iteration_dir / "verification_response.yaml").write_text(response.content, encoding="utf-8")
        parsed = _parse_yaml_mapping(response.content)
        status = str(parsed.get("status") or "not_good").strip().lower()
        feedback = str(parsed.get("feedback") or response.content).strip()
        null_fields = parsed.get("null_fields") or []
        if not isinstance(null_fields, list):
            null_fields = []
        if report.get("status") != "good":
            status = "not_good"
            feedback = "; ".join(report.get("errors") or [feedback])
        if status not in {"good", "not_good"}:
            status = "not_good"
        self.remember(AgentMessage(
            sent_from=self.name,
            sent_to="LayoutAgent" if status == "not_good" else "orchestrator",
            content=feedback,
            iteration=iteration,
            metadata={"viewport": viewport_range, "status": status},
        ))
        return VerificationResult(status, feedback, [str(value) for value in null_fields], report, response)


class QAAgent(BaseTableAgent):
    name = "QAAgent"
    profile = "Table question answering agent"
    goal = "Produce the final answer after layout traversal is complete."

    def __init__(self, llm: BaseLLM, system_prompt: str):
        super().__init__()
        self.llm = llm
        self.system_prompt = system_prompt

    def run(self, *, prompt: str, image_path: Path | None = None, fallback_prompt: str | None = None) -> LLMResponse:
        generate_with_image = getattr(self.llm, "generate_with_image", None)
        if image_path is not None and callable(generate_with_image):
            return generate_with_image(prompt=prompt, image_path=image_path, system_prompt=self.system_prompt)
        return self.llm.generate(prompt=fallback_prompt or prompt, system_prompt=self.system_prompt)


def _canonical_yaml(text: str) -> Any:
    if not text.strip():
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text.strip()


def _execute_verifier(
    verifier_path: Path,
    workbook_path: Path,
    sheet_name: str,
    structure_path: Path,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [sys.executable, str(verifier_path), str(workbook_path), sheet_name, str(structure_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "not_good", "errors": [f"Verifier execution failed: {exc}"]}
    if result.returncode != 0:
        return {"status": "not_good", "errors": [result.stderr.strip() or "Verifier failed."]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "not_good", "errors": ["Verifier returned invalid JSON."]}


_VERIFIER_CODE = '''from __future__ import annotations

import json
import sys

import openpyxl
import yaml
from openpyxl.utils.cell import range_boundaries


def walk(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in {"range", "header_range", "data_range"} and child is not None:
                yield child_path, child
            else:
                yield from walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{path}[{index}]")


workbook_path, sheet_name, structure_path = sys.argv[1:4]
with open(structure_path, "r", encoding="utf-8") as handle:
    structure = yaml.safe_load(handle) or {}

workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
errors = []
try:
    worksheet = workbook[sheet_name]
    for path, value in walk(structure):
        try:
            min_col, min_row, max_col, max_row = range_boundaries(str(value))
        except (TypeError, ValueError):
            errors.append(f"{path} is not a valid A1 range: {value}")
            continue
        if min_row < 1 or min_col < 1 or max_row > 1048576 or max_col > 16384:
            errors.append(f"{path} is outside Excel worksheet bounds: {value}")
finally:
    workbook.close()

print(json.dumps({"status": "not_good" if errors else "good", "errors": errors}))
'''
