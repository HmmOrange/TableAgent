from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentMessage:
    sent_from: str
    sent_to: str
    content: str
    iteration: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMemory:
    messages: list[AgentMessage] = field(default_factory=list)

    def add(self, message: AgentMessage) -> None:
        self.messages.append(message)


class BaseTableAgent:
    name = "Agent"
    profile = ""
    goal = ""

    def __init__(self):
        self.memory = AgentMemory()

    def remember(self, message: AgentMessage) -> None:
        self.memory.add(message)


__all__ = ["AgentMemory", "AgentMessage", "BaseTableAgent"]
