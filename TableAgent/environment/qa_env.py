from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import openpyxl
import pandas as pd

from TableAgent.schema.experience import ExperiencePool
from TableAgent.schema.range import AxisSelection, Cell, CellRange
from TableAgent.schema.header import Header
from TableAgent.utils.structure_utils import load_table_structures
from TableAgent.environment.notebook import Notebook
from TableAgent.environment.logger import QALogger

class QAEnvironment:
    """Notebook-like runtime environment for QA execution."""
    def __init__(
        self,
        structure_path: str,
        workbook_path: str,
        max_experience_records: int = 5,
        log_path: Optional[str] = None,
        max_observation_chars: int = 2000,
        max_error_chars: int = 2000,
        max_value_repr_chars: int = 800,
    ):
        from TableAgent.QA.operators.table_operator import TableOperators

        self.structure_path = structure_path
        self.workbook_path = workbook_path
        
        # Load tables structures
        self.structures = load_table_structures(structure_path)
        
        # Load workbook (data_only=True reads evaluated values of formulas)
        self.workbook = openpyxl.load_workbook(workbook_path, data_only=True)
        
        # Experience pool
        self.experience_pool = ExperiencePool(max_records=max_experience_records)
        
        # Operators facade
        self.operators = TableOperators(self)
        
        # Logger
        self.logger = QALogger(log_path)
        
        # Try importing numpy
        try:
            import numpy as np
        except ImportError:
            np = None

        # Build initial namespace
        initial_ns = {
            "pd": pd,
            "openpyxl": openpyxl,
            "env": self,
            "operators": self.operators,
            "Cell": Cell,
            "CellRange": CellRange,
            "AxisSelection": AxisSelection,
            "Header": Header,
        }
        if np is not None:
            initial_ns["np"] = np
            
        # Notebook workspace
        self.notebook = Notebook(
            initial_ns,
            max_observation_chars=max_observation_chars,
            max_error_chars=max_error_chars,
            max_value_repr_chars=max_value_repr_chars,
        )
        
        # Keep execution_namespace property pointing to the notebook's namespace for backward compatibility
        self.execution_namespace = self.notebook.namespace

    def default_table_id(self) -> str:
        """Returns the first structure key or raises a clear error if none exist."""
        if not self.structures:
            raise ValueError("No table structures are loaded in the environment.")
        return next(iter(self.structures.keys()))

    def get_table_structure(self, table_id: str) -> Dict[str, Any]:
        """Get parsed structure for a table_id."""
        if table_id in self.structures:
            return self.structures[table_id]
        for t_id, struct in self.structures.items():
            if struct.get("name") == table_id:
                return struct
        return {}

    def get_sheet(self, sheet_name: str) -> Any:
        """Get openpyxl sheet object by name."""
        if sheet_name in self.workbook.sheetnames:
            return self.workbook[sheet_name]
        return None

    def get_active_sheet(self) -> Any:
        """Get openpyxl active sheet object."""
        return self.workbook.active

    def get_active_sheet_name(self) -> str:
        """Get the active sheet name."""
        return self.workbook.active.title

    def execute_code(self, code: str, cell_id: Optional[str] = None) -> Tuple[str, str, bool, Dict[str, Any]]:
        """
        Execute python code in the notebook cell.
        Captures and returns stdout, error traceback, success flag, and namespace updates.
        """
        if not cell_id:
            cell_id = f"cell_{len(self.notebook.cells) + 1}"
            
        res = self.notebook.execute_cell(cell_id, code)
        
        observation = self.notebook.observation_for_cell(res)

        # Log execution. Keep full text in notebook cells; log previews by default.
        self.logger.log_event("execute_code", {
            "cell_id": res.cell_id,
            "code": res.code,
            "success": res.success,
            "stdout_preview": observation.stdout_preview,
            "stderr_preview": observation.stderr_preview,
            "error_preview": observation.error_preview,
            "stdout_chars": len(res.stdout),
            "stderr_chars": len(res.stderr),
            "error_chars": len(res.error),
            "stdout_truncated": observation.stdout_truncated,
            "stderr_truncated": observation.stderr_truncated,
            "error_truncated": observation.error_truncated,
            "namespace_updates": observation.namespace_summary,
            "namespace_updates_keys": list(res.namespace_updates.keys()),
        })
        
        output = observation.stdout_preview
        if observation.stderr_preview:
            if output:
                output += "\n"
            output += f"Stderr:\n{observation.stderr_preview}"
            
        return output, observation.error_preview, res.success, res.namespace_updates

    def get_history(
        self,
        last_n: Optional[int] = None,
        include_output: bool = True,
        max_code_len: int = 1000,
        max_output_len: int = 800,
        only_success: Optional[bool] = None,
    ) -> str:
        return self.notebook.get_history(
            last_n=last_n,
            include_output=include_output,
            max_code_len=max_code_len,
            max_output_len=max_output_len,
            only_success=only_success,
        )

    def preview_variable(self, name: str, rows: int = 5, max_chars: Optional[int] = None) -> str:
        return self.notebook.preview_variable(name, rows=rows, max_chars=max_chars)

    def get_cell_output(self, cell_id: str, max_chars: Optional[int] = None) -> str:
        return self.notebook.get_cell_output(cell_id, max_chars=max_chars)

    def export_notebook(self, path: str | Path) -> Path:
        return self.notebook.export_ipynb(path)
