from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eval.datasets.base import EvalSample


class BasePipeline(ABC):
    name: str

    @abstractmethod
    def prepare_table(self, table_id: str) -> dict[str, Any]:
        pass

    @abstractmethod
    def answer_sample(self, sample: EvalSample) -> dict[str, Any]:
        pass
