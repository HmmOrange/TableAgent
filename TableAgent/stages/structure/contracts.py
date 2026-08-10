from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from TableAgent.pipeline.sample import EvalSample

if TYPE_CHECKING:
    from .cache import StructureCacheRecord


@dataclass(frozen=True)
class StructureInput:
    samples: tuple[EvalSample, ...]
    force: bool = True


@dataclass(frozen=True)
class StructureOutput:
    records: tuple[StructureCacheRecord, ...]
