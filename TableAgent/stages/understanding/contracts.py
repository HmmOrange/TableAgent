from dataclasses import dataclass
from pathlib import Path

from TableAgent.llm import LLMResponse


@dataclass(frozen=True)
class UnderstandingInput:
    workbook_path: Path
    sheet_name: str
    artifact_dir: Path


@dataclass(frozen=True)
class StructureUnderstanding:
    headers: tuple[str, ...]
    groups: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "headers": list(self.headers),
            "groups": list(self.groups),
        }


# Preserve the public name while callers migrate to the structure-oriented name.
HeaderUnderstanding = StructureUnderstanding


@dataclass(frozen=True)
class UnderstandingOutput:
    understanding: StructureUnderstanding
    viewport_range: str
    image_path: Path
    result_path: Path
    response: LLMResponse
