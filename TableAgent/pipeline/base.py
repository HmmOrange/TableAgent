from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from TableAgent.schema import EvalSample
from TableAgent.pipeline.contracts import PipelineOutput

__all__ = ["BasePipeline", "PipelineOutput"]


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
