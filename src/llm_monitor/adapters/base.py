"""Abstract base adapter contract for AI coding agent sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_monitor.models import GenerationSpan, SessionTimeline


class BaseAdapter(ABC):
    """Abstract interface that every agent telemetry adapter implements."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the agent adapter (e.g. 'codex', 'claude')."""
        ...

    @abstractmethod
    def detect(self) -> bool:
        """Return True if this agent's local files or logs exist on disk."""
        ...

    @abstractmethod
    def collect(self, max_sessions: int = 64) -> list[GenerationSpan]:
        """Read local logs in strict read-only mode and return parsed generation spans."""
        ...

    @abstractmethod
    def collect_sessions(self, max_sessions: int = 32) -> list[SessionTimeline]:
        """Read local sessions and return parsed timelines with user/assistant activity."""
        ...
