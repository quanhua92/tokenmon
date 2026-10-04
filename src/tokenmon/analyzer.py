"""Windowed aggregation and throughput metrics calculation."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict

from tokenmon.models import GenerationSpan, WindowSummary

WINDOW_DURATIONS: dict[str, float] = {
    "30m": 1800.0,
    "1d": 86400.0,
    "7d": 7 * 86400.0,
    "30d": 30 * 86400.0,
    "all": float("inf"),
}


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
