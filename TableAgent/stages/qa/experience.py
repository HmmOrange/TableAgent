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

    def select(self, subtask_id: Optional[str] = None) -> List[ExperienceRecord]:
        """Pick the attempts worth showing, in two tiers.

        A retry prompt needs two different things and they used to be confused for one.
        The first is the lesson: what this subtask just tried and why it failed. The
        second is the demonstration: working code against this workbook's operators and
        column names, which is what the accepted attempts of earlier subtasks are.

        Scoping the pool to the asking subtask alone kept the lesson and threw away the
        demonstrations, and a subtask on its first round has no attempts of its own, so
        three quarters of prompts ended up with no examples at all. Ranking everything by
        score did the opposite: accepted attempts crowded out the failure the prompt
        exists to explain. So take the subtask's own attempts first, then fill the
        remaining budget with the best-scoring attempts from elsewhere.
        """
        if subtask_id is None:
            return sorted(
                self.records,
                key=lambda record: (record.score, record.round),
                reverse=True,
            )[:self.max_records]

        own = sorted(
            (record for record in self.records if record.subtask_id == subtask_id),
            key=lambda record: record.round,
            reverse=True,
        )[:self.max_records]

        remaining = self.max_records - len(own)
        if remaining <= 0:
            return own
        others = sorted(
            (record for record in self.records if record.subtask_id != subtask_id),
            key=lambda record: (record.score, record.round),
            reverse=True,
        )[:remaining]
        return own + others

    def format(
        self,
        max_code_chars: Optional[int] = None,
        max_observation_chars: Optional[int] = None,
        subtask_id: Optional[str] = None,
    ) -> str:
        """Format selected experiences for inclusion in a model prompt."""
        selected = self.select(subtask_id)
        if not selected:
            return "No previous experience."

        max_code_chars = self.max_code_chars if max_code_chars is None else max_code_chars
        max_observation_chars = self.max_observation_chars if max_observation_chars is None else max_observation_chars
        formatted_parts = []
        # Demonstrations first, then this subtask's own attempts, each chronological. The
        # attempt being fixed ends up nearest the instruction that asks to fix it.
        own_first = subtask_id is not None
        selected_chronological = sorted(
            selected,
            key=lambda record: (
                record.subtask_id == subtask_id if own_first else False,
                record.round,
            ),
        )
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
