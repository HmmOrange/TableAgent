from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from TableAgent.schema import EvalSample


@dataclass
class PipelineOutput:
    sample_id: str
    structured_table: Any | None = None
    predicted_answer: str = ""
    latency: float = 0.0
    token_usage: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePipeline(ABC):
    name: str

    @abstractmethod
    def run(self, sample: EvalSample) -> PipelineOutput:
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        raise NotImplementedError

    def start_timer(self) -> float:
        return time.perf_counter()

    def stop_timer(self, start_time: float) -> float:
        return time.perf_counter() - start_time
