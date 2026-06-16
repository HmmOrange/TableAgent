from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
import time

from datasets.base import EvalSample

@dataclass
class PipelineOutput:
    sample_id: str
    structured_table: Optional[Any] = None
    predicted_answer: str = ""

    # Metrics and logs
    latency: float = 0.0
    token_usage: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})
    metadata: dict[str, Any] = field(default_factory=dict)

class BasePipeline(ABC):
    name: str

    @abstractmethod
    def run(self, sample: EvalSample) -> PipelineOutput:
        """Execute the pipeline on a sample.
        
        Args:
            sample: Evaluation sample to process
            
        Returns:
            PipelineOutput with predictions and metrics
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration including model and parameters
        
        Returns:
            Dict with pipeline config, model info, and parameters
        """
        pass

    def start_timer(self) -> float:
        return time.perf_counter()

    def stop_timer(self, start_time: float) -> float:
        return time.perf_counter() - start_time


