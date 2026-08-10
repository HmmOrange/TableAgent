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
    score: float = 0.0
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

    def select(self) -> List[ExperienceRecord]:
        """Select successful and recent attempts up to the configured limit."""
        sorted_records = sorted(
            self.records,
            key=lambda record: (record.score, record.round),
            reverse=True,
        )
        return sorted_records[:self.max_records]

    def format(
        self,
        max_code_chars: Optional[int] = None,
        max_observation_chars: Optional[int] = None,
    ) -> str:
        """Format selected experiences for inclusion in a model prompt."""
        selected = self.select()
        if not selected:
            return "No previous experience."

        max_code_chars = self.max_code_chars if max_code_chars is None else max_code_chars
        max_observation_chars = self.max_observation_chars if max_observation_chars is None else max_observation_chars
        formatted_parts = []
        selected_chronological = sorted(selected, key=lambda record: record.round)
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
