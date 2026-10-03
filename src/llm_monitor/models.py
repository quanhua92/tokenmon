"""Core domain models for agent generation spans, timeline events, and session statistics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

MIN_MEASURED_DURATION: float = 1.0  # Spans under 1.0s are ignored to eliminate jitter
MAX_REALISTIC_TPS: float = 400.0    # Ceiling to reject sub-millisecond collapsed timestamps


@dataclass(frozen=True)
class GenerationSpan:
    """Atomic telemetry record for a single model generation stream."""

    agent: str
    session_id: str
    turn_id: str
    model: str
    tokens: int
    started_at: float
    ended_at: float
    timing_source: str = "unknown"
    is_valid: bool = True
    note: str | None = None

    @property
    def duration(self) -> float:
        """Stream output duration in seconds."""
        if self.started_at is None or self.ended_at is None:
            return 0.0
        return max(0.0, self.ended_at - self.started_at)

    @property
    def tps(self) -> float | None:
        """Instantaneous throughput in tokens per second."""
        if not self.is_valid or self.duration <= 0.0 or self.tokens <= 0:
            return None
        return self.tokens / self.duration


def create_span(
    agent: str,
    session_id: str,
    turn_id: str,
    model: str,
    tokens: int | None,
    started_at: float | None,
    ended_at: float | None,
    timing_source: str = "unknown",
    note: str | None = None,
    min_duration: float = MIN_MEASURED_DURATION,
    max_tps: float = MAX_REALISTIC_TPS,
) -> GenerationSpan:
    """Centralized validator and builder for all agent generation spans.

    Ensures consistent duration boundaries, minimum sampling thresholds, and
    sanity limits across all agent adapters (Codex, Claude, etc.).
    """
    is_valid = True
    validated_note = note
    toks = tokens if isinstance(tokens, int) and tokens >= 0 else 0

    if validated_note is not None:
        is_valid = False
    elif (
        started_at is None
        or ended_at is None
        or not math.isfinite(started_at)
        or not math.isfinite(ended_at)
    ):
        is_valid = False
        validated_note = "missing_timestamp"
    else:
        duration = ended_at - started_at
        if duration <= 0.0:
            is_valid = False
            validated_note = "invalid_duration"
        elif duration < min_duration:
            is_valid = False
            validated_note = "duration_under_1s"
        elif toks <= 0:
            is_valid = False
            validated_note = "zero_tokens"
        elif (toks / duration) > max_tps:
            is_valid = False
            validated_note = "unconfirmed_boundary_tps"

    return GenerationSpan(
        agent=agent,
        session_id=session_id,
        turn_id=turn_id,
        model=model,
        tokens=toks,
        started_at=started_at if started_at is not None else 0.0,
        ended_at=ended_at if ended_at is not None else 0.0,
        timing_source=timing_source,
        is_valid=is_valid,
        note=validated_note,
    )


@dataclass(frozen=True)
class WindowSummary:
    """Aggregated throughput and distribution metrics for a time window."""

    window_name: str
    model: str
    total_spans: int
    valid_spans: int
    excluded_spans: int
    total_tokens: int
    total_duration: float
    weighted_tps: float | None
    median_tps: float | None
    min_tps: float | None
    max_tps: float | None


@dataclass(frozen=True)
class TimelineEvent:
    """Discrete event in an agent conversation (user prompt, assistant response, tool run)."""

    timestamp: float
    kind: str  # 'user_message', 'assistant_message', 'reasoning', 'tool_call', 'tool_output', 'turn_start', 'turn_end'
    turn_id: str
    summary: str
    tokens: int | None = None
    duration: float | None = None


@dataclass
class SessionTimeline:
    """Full event lifecycle and activity statistics for a single conversation session."""

    session_id: str
    agent: str
    model: str
    created_at: float
    updated_at: float
    events: list[TimelineEvent] = field(default_factory=list)

    @property
    def user_messages(self) -> int:
        """Total count of user message prompts."""
        return sum(1 for e in self.events if e.kind == "user_message")

    @property
    def assistant_messages(self) -> int:
        """Total count of assistant output responses."""
        return sum(1 for e in self.events if e.kind == "assistant_message")

    @property
    def tool_calls(self) -> int:
        """Total count of tools executed by the assistant."""
        return sum(1 for e in self.events if e.kind == "tool_call")

    @property
    def total_tokens(self) -> int:
        """Sum of all tokens generated across responses in this session."""
        return sum(e.tokens for e in self.events if e.tokens)

    @property
    def session_duration(self) -> float:
        """Total elapsed seconds from first to last event in the session."""
        if not self.events:
            return 0.0
        return max(0.0, self.updated_at - self.created_at)

    def idle_time(self, now: float | None = None) -> float:
        """Seconds elapsed since the last event in this session."""
        ref = now if now is not None else time.time()
        return max(0.0, ref - self.updated_at)

    def status(self, now: float | None = None, unattended_threshold: float = 600.0) -> str:
        """Return human-readable status: Active, Idle, or Unattended."""
        idle_s = self.idle_time(now)
        if idle_s < 120.0:
            return "Active"
        if idle_s >= unattended_threshold:
            minutes = int(idle_s // 60)
            return f"Unattended (idle {minutes}m)"
        minutes = int(idle_s // 60)
        return f"Idle ({minutes}m)"
