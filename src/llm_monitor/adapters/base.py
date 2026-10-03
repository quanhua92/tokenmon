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
    def collect(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]:
        """Read local logs in strict read-only mode and return parsed generation spans.

        If min_timestamp is given, skip files/events strictly older than min_timestamp.
        """
        ...

    @abstractmethod
    def collect_sessions(self, max_sessions: int = 32, min_timestamp: float | None = None) -> list[SessionTimeline]:
        """Read local sessions and return parsed timelines with user/assistant activity.

        If min_timestamp is given, skip sessions updated before min_timestamp.
        """
        ...

    def last_generation_timestamp(self, max_probe_sessions: int = 5) -> float | None:
        """Return epoch timestamp of most recent generation span, or None."""
        spans = self.collect(max_sessions=max_probe_sessions)
        if spans:
            return max(s.ended_at for s in spans)
        return None
