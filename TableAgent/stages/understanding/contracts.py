from dataclasses import dataclass
from pathlib import Path

from TableAgent.llm import LLMResponse


@dataclass(frozen=True)
class UnderstandingInput:
    workbook_path: Path
    sheet_name: str
    artifact_dir: Path


@dataclass(frozen=True)
class HeaderUnderstanding:
    row_headers: tuple[str, ...]
    column_headers: tuple[str, ...]
    row_group_headers: tuple[str, ...]
    column_group_headers: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "row_headers": list(self.row_headers),
            "column_headers": list(self.column_headers),
            "row_group_headers": list(self.row_group_headers),
            "column_group_headers": list(self.column_group_headers),
        }


@dataclass(frozen=True)
class UnderstandingOutput:
    understanding: HeaderUnderstanding
    viewport_range: str
    image_path: Path
    result_path: Path
    response: LLMResponse
