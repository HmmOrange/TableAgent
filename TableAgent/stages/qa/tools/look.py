"""Look at a cell range as an image: the range is rendered with LibreOffice and a vision model answers a question.

For what the cell values do not show: tick marks and shading, how merged headers span the columns below them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from TableAgent.stages.qa.ledger import MAX_LLM_CALLS
from TableAgent.stages.qa.prompts import VISION_PROMPT, VISION_SYSTEM_PROMPT
from TableAgent.stages.qa.utils.llm import CountingLLM
from TableAgent.utils.excel_utils import cell_to_a1, parse_a1_range

MAX_LOOK_ROWS = 60
MAX_LOOK_COLUMNS = 30


class Vision:
    def __init__(self, llm: CountingLLM, renderer: Any, workbook_path: Path, image_dir: Path):
        self.llm = llm
        self.renderer = renderer
        self.workbook_path = workbook_path
        self.image_dir = image_dir

    def look(self, ref: str, question: str, sheet: str) -> str:
        """The vision model's answer to `question` about an image of the range `ref` of `sheet`."""
        cells = parse_a1_range(ref, sheet)
        rows, columns = cells.end_row - cells.start_row + 1, cells.end_col - cells.start_col + 1
        if rows > MAX_LOOK_ROWS or columns > MAX_LOOK_COLUMNS:
            raise ValueError(
                f"{ref} is {rows} rows x {columns} columns; look at most {MAX_LOOK_ROWS} rows x {MAX_LOOK_COLUMNS} columns."
            )
        if self.llm.used >= MAX_LLM_CALLS:
            raise ValueError("The model call budget is used up.")
        area = f"{cell_to_a1(cells.start_row, cells.start_col)}:{cell_to_a1(cells.end_row, cells.end_col)}"
        image = self.image_dir / f"{re.sub(r'[^A-Za-z0-9]+', '_', cells.sheet)}_{area.replace(':', '_')}.png"
        if not image.exists():
            self.renderer.render_workbook_range(self.workbook_path, cells.sheet, area, image)
        prompt = VISION_PROMPT.format(area=area, sheet=cells.sheet, question=question)
        return self.llm.generate_with_image(prompt, image, VISION_SYSTEM_PROMPT)
