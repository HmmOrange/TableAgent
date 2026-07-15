from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from TableAgent.structure.layout.parsing import _is_valid_structure


@dataclass(frozen=True)
class DeterministicVerificationResult:
    status: str
    feedback: str
    null_fields: list[str]
    report: dict[str, Any]
    structure_text: str

    @property
    def is_good(self) -> bool:
        return self.status == "good"


class DeterministicVerifier:
    def __init__(self, timeout_seconds: float = 30):
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        *,
        workbook_path: Path,
        sheet_name: str,
        structure_text: str,
        iteration_dir: Path,
    ) -> DeterministicVerificationResult:
        structure_path = iteration_dir / "structure_after.yaml"
        report = self._execute(workbook_path, sheet_name, structure_path)
        if not _is_valid_structure(structure_text):
            report = {"status": "not_good", "errors": ["Candidate structure is empty or invalid."]}
        repaired = str(report.get("repaired_structure_yaml") or structure_text)
        if repaired != structure_text:
            structure_path.write_text(repaired, encoding="utf-8")
            (iteration_dir / "structure_repaired.yaml").write_text(repaired, encoding="utf-8")
        report_payload = {key: value for key, value in report.items() if key != "repaired_structure_yaml"}
        (iteration_dir / "verification_output.json").write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        status = str(report.get("status") or "not_good").strip().lower()
        if status not in {"good", "not_good"}:
            status = "not_good"
        null_fields = report.get("null_fields") or []
        if not isinstance(null_fields, list):
            null_fields = []
        feedback = str(report.get("feedback") or "; ".join(report.get("errors") or ["Verification failed."]))
        return DeterministicVerificationResult(status, feedback, [str(value) for value in null_fields], report, repaired)

    def _execute(self, workbook_path: Path, sheet_name: str, structure_path: Path) -> dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "TableAgent.structure.verification.worker",
                    str(workbook_path),
                    sheet_name,
                    str(structure_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._tool_error(f"Verifier execution failed: {exc}")
        if result.returncode != 0:
            return self._tool_error(result.stderr.strip() or "Verifier failed.")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return self._tool_error("Verifier returned invalid JSON.")

    @staticmethod
    def _tool_error(message: str) -> dict[str, Any]:
        return {
            "status": "not_good",
            "errors": [message],
            "tool_error": True,
            "feedback": "Deterministic verifier tool failed before validating the structure.",
        }
