import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from TableAgent.llm import BaseLLM, LLMResponse
from TableAgent.stages.structure.traversal import Direction

from .direction_prompts import DIRECTION_SYSTEM_PROMPT, DIRECTION_USER_PROMPT_TEMPLATE


@dataclass(frozen=True)
class DirectionResult:
    directions: list[str]
    response: LLMResponse


class DirectionAgent:
    def __init__(self, vlm: BaseLLM):
        self.vlm = vlm

    def run(
        self,
        *,
        workbook_name: str,
        sheet_name: str,
        viewport_range: str,
        direction: str,
        image_path: Path,
        iteration_dir: Path,
    ) -> DirectionResult:
        artifact_dir = iteration_dir / "directions"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        prompt = DIRECTION_USER_PROMPT_TEMPLATE.format(
            workbook_name=workbook_name,
            sheet_name=sheet_name,
            viewport_range=viewport_range,
            direction=direction,
        )
        (artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        response = self.vlm.generate_with_image(
            prompt=prompt,
            image_path=image_path,
            system_prompt=DIRECTION_SYSTEM_PROMPT,
        )
        (artifact_dir / "response.txt").write_text(response.content, encoding="utf-8")
        directions = parse_directions(response.content, direction)
        (artifact_dir / "result.json").write_text(
            json.dumps({"remaining_directions": directions}, indent=2) + "\n",
            encoding="utf-8",
        )
        return DirectionResult(directions, response)


def parse_directions(content: str, current_direction: str) -> list[str]:
    text = str(content).strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError:
            return []
    if not isinstance(payload, dict):
        return []
    values = payload.get("remaining_directions") or payload.get("directions") or []
    if not isinstance(values, list):
        return []

    current = str(current_direction).strip().lower()
    opposites = {"right": "left", "left": "right", "down": "up", "up": "down"}
    blocked = {current, opposites.get(current)}
    directions: list[str] = []
    for value in values:
        parsed = Direction.parse(str(value))
        if parsed is None or parsed == Direction.STAY:
            continue
        name = parsed.name.lower()
        if name not in blocked and name not in directions:
            directions.append(name)
        if len(directions) == 2:
            break
    return directions


__all__ = ["DirectionAgent", "DirectionResult", "parse_directions"]
