"""Windowed aggregation and throughput metrics calculation."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass

from tokenmon.models import GenerationSpan, SessionTimeline, WindowSummary, recorded_speed_mode

WINDOW_DURATIONS: dict[str, float] = {
    "30m": 1800.0,
    "1d": 86400.0,
    "7d": 7 * 86400.0,
    "30d": 30 * 86400.0,
    "all": float("inf"),
}


@dataclass(frozen=True)
class TimelineInterval:
    """One evidenced interval rendered on a visual session timeline."""

    kind: str  # stream or a recorded event category
    started_at: float
    ended_at: float
    turn_id: str
    model: str | None = None
    tokens: int | None = None
    tps: float | None = None
    summary: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    speed: str | None = None

    @property
    def speed_mode(self) -> str | None:
        return recorded_speed_mode(self.speed, self.service_tier)

    @property
    def duration(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


@dataclass(frozen=True)
class TimelineLane:
    """A session and its renderable activity intervals."""

    session: SessionTimeline
    started_at: float
    ended_at: float
    intervals: tuple[TimelineInterval, ...]
    streaming_duration: float
    weighted_tps: float | None

    @property
    def duration(self) -> float:
        return max(0.0, self.ended_at - self.started_at)


def filter_by_window(
    spans: list[GenerationSpan],
    window_seconds: float,
    reference_time: float,
) -> list[GenerationSpan]:
    """Return spans completed within `window_seconds` before `reference_time`."""
    if math_is_inf(window_seconds):
        return list(spans)
    return [s for s in spans if 0 <= reference_time - s.ended_at <= window_seconds]


def math_is_inf(val: float) -> bool:
    return val == float("inf")


def group_by_model(spans: list[GenerationSpan]) -> dict[str, list[GenerationSpan]]:
    """Group spans by model identifier."""
    grouped: dict[str, list[GenerationSpan]] = defaultdict(list)
    for span in spans:
        grouped[span.model].append(span)
    return dict(grouped)


def summarize_spans(
    spans: list[GenerationSpan],
    window_name: str,
    model_name: str,
) -> WindowSummary:
    """Calculate mathematically weighted throughput and speed distribution."""
    valid_spans = [s for s in spans if s.tps is not None]
    rates = [s.tps for s in valid_spans if s.tps is not None]

    total_tokens = sum(s.tokens for s in valid_spans)
    total_duration = sum(s.duration for s in valid_spans)

    weighted_tps = (total_tokens / total_duration) if total_duration > 0 else None
    median_tps = statistics.median(rates) if rates else None
    min_tps = min(rates) if rates else None
    max_tps = max(rates) if rates else None

    return WindowSummary(
        window_name=window_name,
        model=model_name,
        total_spans=len(spans),
        valid_spans=len(valid_spans),
        excluded_spans=len(spans) - len(valid_spans),
        total_tokens=total_tokens,
        total_duration=total_duration,
        weighted_tps=weighted_tps,
        median_tps=median_tps,
        min_tps=min_tps,
        max_tps=max_tps,
    )


def analyze_windows(
    spans: list[GenerationSpan],
    window_names: list[str] | None = None,
    now: float | None = None,
) -> dict[str, list[WindowSummary]]:
    """Generate window summaries grouped by model name."""
    current_time = now if now is not None else time.time()
    windows = window_names or ["30m", "1d", "7d", "all"]

    by_model = group_by_model(spans)
    results: dict[str, list[WindowSummary]] = {}

    for model, m_spans in sorted(by_model.items()):
        model_summaries = []
        for win in windows:
            duration_s = WINDOW_DURATIONS.get(win, float("inf"))
            windowed = filter_by_window(m_spans, duration_s, current_time)
            model_summaries.append(summarize_spans(windowed, win, model))
        results[model] = model_summaries

    return results


def analyze_agent_model_windows(
    spans: list[GenerationSpan],
    window_names: list[str] | None = None,
    now: float | None = None,
) -> dict[tuple[str, str], list[WindowSummary]]:
    """Generate window summaries grouped by source agent and model name."""
    current_time = now if now is not None else time.time()
    windows = window_names or ["30m", "1d", "7d", "all"]

    grouped: dict[tuple[str, str], list[GenerationSpan]] = defaultdict(list)
    for span in spans:
        grouped[(span.agent, span.model)].append(span)

    results: dict[tuple[str, str], list[WindowSummary]] = {}
    for agent_model, grouped_spans in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        _, model = agent_model
        results[agent_model] = [
            summarize_spans(
                filter_by_window(grouped_spans, WINDOW_DURATIONS.get(window, float("inf")), current_time),
                window,
                model,
            )
            for window in windows
        ]
    return results


def build_timeline_lanes(
    timelines: list[SessionTimeline],
    spans: list[GenerationSpan],
) -> list[TimelineLane]:
    """Build longest-first visual lanes from confirmed activity boundaries.

    Generation intervals come only from valid spans. Recorded events become
    duration blocks when supported by telemetry and point markers otherwise.
    """
    candidate_spans: dict[tuple[str, str], list[GenerationSpan]] = defaultdict(list)
    for span in spans:
        if span.tps is not None:
            candidate_spans[(span.agent, span.session_id)].append(span)
    spans_by_session: dict[tuple[str, str], list[GenerationSpan]] = defaultdict(list)
    for timeline in timelines:
        key = (timeline.agent, timeline.session_id)
        for span in candidate_spans.get(key, []):
            if (timeline.created_at <= 0 and timeline.updated_at <= 0) or (
                    span.ended_at >= timeline.created_at and span.started_at <= timeline.updated_at):
                spans_by_session[key].append(span)
    lanes: list[TimelineLane] = []
    for timeline in timelines:
        session_spans = sorted(
            spans_by_session.get((timeline.agent, timeline.session_id), []),
            key=lambda span: (span.started_at, span.ended_at),
        )
        intervals: list[TimelineInterval] = []
        for span in session_spans:
            intervals.append(TimelineInterval(
                kind="stream",
                started_at=span.started_at,
                ended_at=span.ended_at,
                turn_id=span.turn_id,
                model=span.model,
                tokens=span.tokens,
                tps=span.tps,
                reasoning_effort=span.reasoning_effort,
                service_tier=span.service_tier,
                speed=span.speed,
            ))

        event_kinds = {
            "user_message": "user",
            "assistant_message": "assistant",
            "reasoning": "reasoning",
            "tool_call": "tool",
            "tool_output": "tool",
            "turn_start": "turn",
            "turn_end": "turn",
        }
        for event in timeline.events:
            interval_kind = event_kinds.get(event.kind)
            if interval_kind is None:
                continue
            duration = event.duration if (
                event.duration is not None
                and event.duration > 0
                and event.kind in {"assistant_message", "reasoning", "tool_call"}
            ) else 0.0
            intervals.append(TimelineInterval(
                kind=interval_kind,
                started_at=event.timestamp,
                ended_at=event.timestamp + duration,
                turn_id=event.turn_id,
                summary=event.summary,
                reasoning_effort=event.reasoning_effort,
                service_tier=event.service_tier,
                speed=event.speed,
            ))

        intervals.sort(key=lambda interval: (interval.started_at, interval.ended_at, interval.kind))
        boundaries = [timeline.created_at, timeline.updated_at]
        boundaries.extend(interval.started_at for interval in intervals)
        boundaries.extend(interval.ended_at for interval in intervals)
        positive_boundaries = [boundary for boundary in boundaries if boundary > 0]
        started_at = min(positive_boundaries) if positive_boundaries else 0.0
        ended_at = max(positive_boundaries) if positive_boundaries else started_at
        total_tokens = sum(span.tokens for span in session_spans)
        streaming_duration = sum(span.duration for span in session_spans)
        weighted_tps = total_tokens / streaming_duration if streaming_duration > 0 else None
        lanes.append(TimelineLane(
            session=timeline,
            started_at=started_at,
            ended_at=ended_at,
            intervals=tuple(intervals),
            streaming_duration=streaming_duration,
            weighted_tps=weighted_tps,
        ))

    lanes.sort(key=lambda lane: (lane.duration, lane.ended_at), reverse=True)
    return lanes
