from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

def _clip(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text
    marker = "\n...[truncated]...\n"
    keep = max_chars - len(marker)
    if keep <= 20:
        return text[:max_chars] + "..."
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]

@dataclass
class ExperienceRecord:
    subtask_id: str
    description: str
    code: str
    observation: str
    reasoning: str = ""
    score: float = 0.0  # 1.0 for success, 0.0 for failure
    round: int = 1

    def __repr__(self) -> str:
        return f"ExperienceRecord(subtask='{self.subtask_id}', score={self.score}, round={self.round})"


@dataclass
class ExperiencePool:
    max_records: int = 5
    max_code_chars: int = 1200
    max_observation_chars: int = 1200
    records: List[ExperienceRecord] = field(default_factory=list)

    def add(self, record: ExperienceRecord):
        self.records.append(record)

    def select(
        self,
        subtask_id: Optional[str] = None,
        prioritize_latest_failure: bool = False,
    ) -> List[ExperienceRecord]:
        """
        Select a bounded list of experiences.
        Prioritizes records for the current subtask, then successful and recent attempts.
        Retry callers can pin the latest rejected attempt even when the pool contains
        many successful records or replanning resets the per-subtask round number.
        """
        indexed_records = list(enumerate(self.records))
        selected: List[ExperienceRecord] = []
        selected_indexes: set[int] = set()
        if prioritize_latest_failure and subtask_id:
            failures = [
                (index, record)
                for index, record in indexed_records
                if record.subtask_id == subtask_id and record.score < 1.0
            ]
            if failures:
                latest_index, latest_failure = failures[-1]
                selected.append(latest_failure)
                selected_indexes.add(latest_index)

        remaining = [
            (index, record)
            for index, record in indexed_records
            if index not in selected_indexes
        ]
        sorted_records = sorted(
            remaining,
            key=lambda indexed_record: (
                indexed_record[1].subtask_id == subtask_id if subtask_id else False,
                indexed_record[1].score,
                indexed_record[1].round,
                indexed_record[0],
            ),
            reverse=True,
        )
        selected.extend(
            record
            for _, record in sorted_records[: max(0, self.max_records - len(selected))]
        )
        return selected

    def format(
        self,
        max_code_chars: Optional[int] = None,
        max_observation_chars: Optional[int] = None,
        subtask_id: Optional[str] = None,
        prioritize_latest_failure: bool = False,
    ) -> str:
        """
        Format the selected experiences into a structured text format for the model prompt.
        """
        selected = self.select(
            subtask_id=subtask_id,
            prioritize_latest_failure=prioritize_latest_failure,
        )
        if not selected:
            return "No previous experience."

        max_code_chars = self.max_code_chars if max_code_chars is None else max_code_chars
        max_observation_chars = self.max_observation_chars if max_observation_chars is None else max_observation_chars
        formatted_parts = []
        # Sort back to chronological order for prompt presentation
        selected_chronological = sorted(selected, key=lambda r: r.round)
        for exp in selected_chronological:
            code = _clip(exp.code, max_code_chars)
            observation = _clip(exp.observation, max_observation_chars)
            part = (
                f"<attempt round=\"{exp.round}\" subtask=\"{exp.subtask_id}\">\n"
                f"  <description>{exp.description}</description>\n"
                f"  <reasoning>{_clip(exp.reasoning, max_observation_chars)}</reasoning>\n"
                f"  <code>\n{code}\n  </code>\n"
                f"  <observation>\n{observation}\n  </observation>\n"
                f"</attempt>"
            )
            formatted_parts.append(part)
        return "\n\n".join(formatted_parts)
