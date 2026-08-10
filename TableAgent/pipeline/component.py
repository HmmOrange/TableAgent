"""Base support for pipeline components composed around one runtime."""

from __future__ import annotations

from typing import Any

from TableAgent.pipeline.contracts import PipelineRuntimeContract


class RuntimeComponent:
    """Expose the documented runtime contract to migrated pipeline components."""

    def __init__(self, runtime: PipelineRuntimeContract):
        self.runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)
