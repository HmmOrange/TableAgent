"""TableAgent: answers questions over Excel workbooks by exploring them in code (see src/README.md)."""

from TableAgent.settings import QASettings
from TableAgent.src.workbook_qa import WorkbookQA

__all__ = ["QASettings", "WorkbookQA"]
