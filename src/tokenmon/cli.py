"""Command line interface for tokenmon."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from datetime import datetime, timezone

from tokenmon import __version__
from tokenmon.adapters import ADAPTER_REGISTRY, BaseAdapter, detect_available_adapters, get_adapter
from tokenmon.analyzer import (
    WINDOW_DURATIONS,
    TimelineLane,
    analyze_agent_model_windows,
    analyze_windows,
    build_timeline_lanes,
    filter_by_window,
    summarize_spans,
)
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, WindowSummary, recorded_speed_mode, format_session_duration

logger = logging.getLogger(__name__)


def metadata_to_dict(value: GenerationSpan | SessionTimeline | TimelineEvent) -> dict:
    return {"reasoning_effort": value.reasoning_effort, "service_tier": value.service_tier,
            "speed": value.speed, "speed_mode": value.speed_mode}


def configuration_label(value: GenerationSpan | SessionTimeline | TimelineEvent) -> str:
    return " ".join(part for part in (value.reasoning_effort, value.speed_mode) if part)


def model_label(value: GenerationSpan | SessionTimeline, compact: bool = False) -> str:
    model = value.model[:12] if compact else value.model
    details = configuration_label(value)
    return model + (f" {details}" if details else "")


def configuration_metadata(spans: list[GenerationSpan]) -> dict:
    settings = {(s.reasoning_effort, s.service_tier, s.speed) for s in spans if s.tps is not None}
    configurations = [
        {"reasoning_effort": effort, "service_tier": tier, "speed": speed,
         "speed_mode": recorded_speed_mode(speed, tier)}
        for effort, tier, speed in sorted(settings, key=lambda values: tuple(v or "" for v in values))
    ]
    common = {}
    for key in ("reasoning_effort", "service_tier", "speed", "speed_mode"):
        values = {config[key] for config in configurations}
        common[key] = next(iter(values)) if len(values) == 1 else None
    return {**common, "configurations": configurations}


def format_table(summaries: list[WindowSummary], compact: bool | None = None) -> str:
    if compact is None:
        term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        compact = term_cols < 88

    if compact:
        headers = ["Window", "Outputs", "Tokens", "TPS", "Median"]
        rows = []
        for s in summaries:
            w_tps = f"{s.weighted_tps:.1f}" if s.weighted_tps is not None else "—"
            med_tps = f"{s.median_tps:.1f}" if s.median_tps is not None else "—"
            rows.append(
                [
                    s.window_name,
                    f"{s.valid_spans}/{s.total_spans}",
                    f"{s.total_tokens:,}",
                    w_tps,
                    med_tps,
                ]
            )
    else:
        headers = [
            "Window",
            "Outputs",
            "Tokens",
            "Time",
            "TPS (wtd)",
            "Median",
            "Range",
        ]
        rows = []
        for s in summaries:
            w_tps = f"{s.weighted_tps:.1f}" if s.weighted_tps is not None else "—"
            med_tps = f"{s.median_tps:.1f}" if s.median_tps is not None else "—"
            range_tps = f"{s.min_tps:.1f} - {s.max_tps:.1f}" if s.min_tps is not None else "—"
            rows.append(
                [
                    s.window_name,
                    f"{s.valid_spans}/{s.total_spans}",
                    f"{s.total_tokens:,}",
                    format_session_duration(s.total_duration),
                    w_tps,
                    med_tps,
                    range_tps,
                ]
            )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def make_line(chars: tuple[str, str, str, str]) -> str:
        return chars[0] + chars[1].join(chars[2] * (w + 2) for w in col_widths) + chars[3]

    lines = [
        make_line(("┌", "┬", "─", "┐")),
        "│ " + " │ ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " │",
        make_line(("├", "┼", "─", "┤")),
    ]
    for row in rows:
        lines.append(
            "│ " + " │ ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " │"
        )
    lines.append(make_line(("└", "┴", "─", "┘")))
    return "\n".join(lines)


def format_sessions_table(
    timelines: list[SessionTimeline],
    now: float,
    compact: bool | None = None,
) -> str:
    if compact is None:
        term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        compact = term_cols < 92

    if compact:
        headers = ["Session", "Model", "Turns", "Tokens", "Duration", "Status"]
        rows = []
        for t in timelines:
            dur_str = format_session_duration(t.session_duration)
            rows.append(
                [
                    t.session_id[:10],
                    model_label(t, compact=True),
                    str(t.assistant_messages),
                    f"{t.total_tokens:,}",
                    dur_str,
                    t.status(now),
                ]
            )
    else:
        headers = [
            "Session ID",
            "Model",
            "Prompts",
            "Turns",
            "Tools",
            "Tokens",
            "Duration",
            "Status",
        ]
        rows = []
        for t in timelines:
            dur_str = format_session_duration(t.session_duration)
            rows.append(
                [
                    t.session_id[:16],
                    model_label(t),
                    str(t.user_messages),
                    str(t.assistant_messages),
                    str(t.tool_calls),
                    f"{t.total_tokens:,}",
                    dur_str,
                    t.status(now),
                ]
            )

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    def make_line(chars: tuple[str, str, str, str]) -> str:
        return chars[0] + chars[1].join(chars[2] * (w + 2) for w in col_widths) + chars[3]

    lines = [
        make_line(("┌", "┬", "─", "┐")),
        "│ " + " │ ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " │",
        make_line(("├", "┼", "─", "┤")),
    ]
    for row in rows:
        lines.append(
            "│ " + " │ ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " │"
        )
    lines.append(make_line(("└", "┴", "─", "┘")))
    return "\n".join(lines)


def format_recent_span(s: GenerationSpan, compact: bool | None = None) -> str:
    if compact is None:
        term_cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        compact = term_cols < 88

    dt = datetime.fromtimestamp(s.ended_at, tz=timezone.utc).astimezone()
    tps_val = s.tps or 0.0
    details = " ".join(value for value in (s.reasoning_effort, s.speed_mode) if value)
    model_label = s.model + (f" {details}" if details else "")

    if compact:
        t_str = dt.strftime("%H:%M:%S")
        m_str = s.model[:12] + (f" {details}" if details else "")
        return (
            f"  {t_str}  {m_str:<12}  "
            f"\033[1;32m{tps_val:5.1f} TPS\033[0m  "
            f"({s.tokens:,} tok / {s.duration:.1f}s)"
        )
    else:
        t_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"  [{t_str}] {model_label:<18} : "
            f"\033[1;32m{tps_val:6.1f} TPS\033[0m  "
            f"({s.tokens:5d} tokens in {s.duration:5.2f}s) "
            f"[{s.timing_source}]"
        )


def format_session_card(t: SessionTimeline, now: float) -> str:
    dur_str = format_session_duration(t.session_duration)

    tok_count = t.total_tokens
    if tok_count >= 1_000_000:
        tok_str = f"{tok_count / 1_000_000:.1f}M"
    elif tok_count >= 1_000:
        tok_str = f"{tok_count / 1_000:.1f}k"
    else:
        tok_str = str(tok_count)

    status_raw = t.status(now)
    if status_raw.startswith("Active"):
        status_badge = "\033[1;32m● Active\033[0m"
    elif status_raw.startswith("Idle"):
        status_badge = f"\033[1;33m○ {status_raw}\033[0m"
    else:
        status_badge = f"\033[2m◌ {status_raw}\033[0m"

    tool_part = f", {t.tool_calls} tools" if t.tool_calls > 0 else ""
    user_part = f"👤 User:      {t.user_messages} prompt{'s' if t.user_messages != 1 else ''}"
    agent_part = f"🤖 Assistant: {t.assistant_messages} turns{tool_part} ({tok_str} tok)"
    time_part = f"⏱️ Elapsed:   {dur_str}"

    lines = [
        f"  📌 \033[1m{t.session_id[:12]}\033[0m  ({model_label(t)})  {status_badge}",
        f"     {user_part}",
        f"     {agent_part}",
        f"     {time_part}",
    ]
    return "\n".join(lines)


def format_timeline_view(timeline: SessionTimeline, now: float) -> str:
    lines = [
        f"🔍 Session Timeline: \033[1m{timeline.session_id}\033[0m ({model_label(timeline)})",
        f"Activity: {timeline.user_messages} User Prompts | {timeline.assistant_messages} Assistant Responses | {timeline.tool_calls} Tool Calls | {timeline.total_tokens:,} Tokens",
        f"Status: \033[1m{timeline.status(now)}\033[0m\n",
    ]

    lines.extend(format_timeline_event(ev) for ev in timeline.events)
    return "\n".join(lines)


def format_timeline_event(ev: TimelineEvent) -> str:
    """Format one event identically in historical and live timelines."""
    event_icons = {
        "user_message": "👤 \033[1;34mUSER\033[0m     ",
        "assistant_message": "🤖 \033[1;32mASSISTANT\033[0m",
        "reasoning": "🧠 \033[1;35mTHINKING\033[0m ",
        "tool_call": "🛠️  \033[1;33mTOOL CALL\033[0m",
        "tool_output": "⚙️  \033[1;33mTOOL DONE\033[0m",
        "turn_start": "▶️  \033[2mSTART\033[0m    ",
        "turn_end": "⏹️  \033[2mCOMPLETE\033[0m ",
    }

    dt = datetime.fromtimestamp(ev.timestamp, tz=timezone.utc).astimezone()
    t_str = dt.strftime("%H:%M:%S")
    tag = event_icons.get(ev.kind, f"•  {ev.kind:<9}")
    dur_str = f" [{ev.duration:.2f}s]" if ev.duration is not None else ""
    details = configuration_label(ev) if ev.kind in {"assistant_message", "reasoning", "tool_call"} else ""
    settings = f" [{details}]" if details else ""
    return f"  {t_str} │ {tag} │ {ev.summary}{dur_str}{settings}"


def format_visual_timeline(
    lanes: list[TimelineLane],
    compact: bool | None = None,
    color: bool | None = None,
) -> str:
    """Render session activity lanes against one shared time axis."""
    if not lanes:
        return "No session activity found."
    term_cols = shutil.get_terminal_size(fallback=(120, 24)).columns
    if compact is None:
        compact = term_cols < 100
    if color is None:
        color = sys.stdout.isatty()

    chart_start = min(lane.started_at for lane in lanes)
    chart_end = max(lane.ended_at for lane in lanes)
    chart_duration = max(1.0, chart_end - chart_start)
    prefix_width = 9 if compact else 20
    suffix_width = 18 if compact else 0
    chart_width = max(20, min(120, term_cols - prefix_width - suffix_width - 2))
    priority = {
        "turn": 1, "user": 2, "assistant": 3, "stream": 4,
        "reasoning": 5, "tool": 6,
    }
    glyphs = {
        "stream": "█", "tool": "▒", "user": "◆",
        "assistant": "●", "reasoning": "◇", "turn": "│",
    }
    colors = {
        "stream": "\033[34m", "tool": "\033[32m",
        "user": "\033[36m", "assistant": "\033[34m", "reasoning": "\033[35m",
        "turn": "\033[2m",
    }

    def local_time(timestamp: float, with_date: bool = False) -> str:
        fmt = "%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S"
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().strftime(fmt)

    def render_cells(cells: list[str | None]) -> str:
        if not color:
            return "".join(glyphs.get(cell, " ") if cell else " " for cell in cells)
        chunks: list[str] = []
        previous: str | None = None
        for cell in cells:
            if cell != previous:
                if previous is not None:
                    chunks.append("\033[0m")
                if cell is not None:
                    chunks.append(colors[cell])
                previous = cell
            chunks.append(glyphs[cell] if cell else " ")
        if previous is not None:
            chunks.append("\033[0m")
        return "".join(chunks)

    lines = [
        f"📈 Session Timeline ({len(lanes)} session{'s' if len(lanes) != 1 else ''}, longest first)",
        f"{local_time(chart_start, True)} — {local_time(chart_end, True)} local time",
        "Legend: █ model output  ▒ tool  ◆ user  ◇ reasoning  ● assistant  │ turn",
        "",
    ]
    for lane in lanes:
        cells: list[str | None] = [None] * chart_width
        for interval in lane.intervals:
            left = int((interval.started_at - chart_start) / chart_duration * chart_width)
            right = int(math.ceil((interval.ended_at - chart_start) / chart_duration * chart_width))
            left = max(0, min(chart_width - 1, left))
            right = max(left + 1, min(chart_width, right))
            for index in range(left, right):
                current = cells[index]
                if current is None or priority[interval.kind] >= priority[current]:
                    cells[index] = interval.kind
        if compact:
            label = lane.session.session_id[:8].ljust(8)
            details = format_session_duration(lane.duration)
            if lane.weighted_tps is not None:
                details += f" · {lane.weighted_tps:.1f} t/s"
        else:
            label = f"{local_time(lane.started_at)} {lane.session.session_id[:10]}".ljust(19)
            details = f"{format_session_duration(lane.duration)} · stream {format_session_duration(lane.streaming_duration)}"
            if lane.weighted_tps is not None:
                details += f" · {lane.weighted_tps:.1f} tok/s"
        lines.append(f"{label} │{render_cells(cells)}│ {details}")

    tick_positions = [0, chart_width // 2, chart_width - 1]
    axis = ["─"] * chart_width
    for position in tick_positions:
        axis[position] = "┼"
    lines.append(" " * prefix_width + "└" + "".join(axis) + "┘")
    labels = [" "] * chart_width
    for position in tick_positions:
        timestamp = chart_start + (position / max(1, chart_width - 1)) * chart_duration
        value = local_time(timestamp)
        start = max(0, min(chart_width - len(value), position - len(value) // 2))
        labels[start:start + len(value)] = value
    lines.append(" " * (prefix_width + 1) + "".join(labels))
    lines.extend(["", "Blank areas are waiting or unavailable boundaries."])
    return "\n".join(lines)


def visual_timeline_to_dict(lanes: list[TimelineLane], adapters: list[BaseAdapter], now: float) -> dict:
    return {
        "meta": {
            "version": __version__,
            "agents": [adapter.name for adapter in adapters],
            "timestamp": now,
            "session_count": len(lanes),
        },
        "sessions": [
            {
                "session_id": lane.session.session_id,
                "agent": lane.session.agent,
                "model": lane.session.model,
                "cwd": lane.session.cwd,
                **metadata_to_dict(lane.session),
                "started_at": lane.started_at,
                "ended_at": lane.ended_at,
                "duration_seconds": round(lane.duration, 3),
                "streaming_duration_seconds": round(lane.streaming_duration, 3),
                "weighted_tps": round(lane.weighted_tps, 2) if lane.weighted_tps is not None else None,
                "intervals": [
                    {
                        "kind": interval.kind,
                        "started_at": interval.started_at,
                        "ended_at": interval.ended_at,
                        "duration_seconds": round(interval.duration, 3),
                        "turn_id": interval.turn_id,
                        "model": interval.model,
                        "tokens": interval.tokens,
                        "tps": round(interval.tps, 2) if interval.tps is not None else None,
                        "summary": interval.summary,
                        "reasoning_effort": interval.reasoning_effort,
                        "service_tier": interval.service_tier,
                        "speed": interval.speed,
                        "speed_mode": interval.speed_mode,
                    }
                    for interval in lane.intervals
                ],
            }
            for lane in lanes
        ],
    }


ASCII_LOGO = r"""
  ______      __              __  ___
 /_  __/___  / /_____  ____  /  |/  /___  ____
  / / / __ \/ //_/ _ \/ __ \/ /|_/ / __ \/ __ \
 / / / /_/ / ,< /  __/ / / / /  / / /_/ / / / /
/_/  \____/_/|_|\___/_/ /_/_/  /_/\____/_/ /_/
             TokenMon · Agent TPS
""".strip("\n")


STATS_GUIDE = """\
📖 How to read this
  • TPS = valid output tokens ÷ summed generation time (overlaps add separately); Median = the middle stream.
  • Outputs = valid/total; streams under 1s or with uncertain timing are excluded from TPS.
  • Timing: Claude turn-span may include latency; OMP starts at the first output item; Pi/OpenCode session tokens do not contribute TPS.
  • Windows: 30m, 1d, 7d, 30d, all · Next: `tokenmon timeline` for activity · `tokenmon logs` for events · `--json` for scripts.
"""


def print_banner(color: bool = True) -> None:
    """Print a compact, modern ASCII banner suitable for standard and narrow split panes."""
    if color and sys.stdout.isatty():
        green = "\033[1;32m"
        reset = "\033[0m"
        print(f"\n{green}{ASCII_LOGO}{reset}\n")
    else:
        print(f"\n{ASCII_LOGO}\n")


def collection_cutoff(window: str | None, include_all: bool, now: float) -> float | None:
    if include_all or window == "all":
        return None
    return now - WINDOW_DURATIONS[window or "30d"]


def stats_windows(window: str | None, include_all: bool) -> list[str]:
    if window:
        return [window]
    return ["30m", "1d", "7d", "30d"] + (["all"] if include_all else [])


def collect_timelines(adapters: list[BaseAdapter], max_sessions: int,
                      min_timestamp: float | None = None) -> list[SessionTimeline]:
    timelines = []
    for adapter in adapters:
        try:
            timelines.extend(adapter.collect_sessions(max_sessions=max_sessions, min_timestamp=min_timestamp))
        except Exception as e:
            logger.warning("Adapter '%s' failed to collect sessions: %s", adapter.name, e)
    timelines.sort(key=lambda t: t.updated_at, reverse=True)
    return timelines


def collect_stats(adapters: list[BaseAdapter], tasks: int, min_timestamp: float | None
                  ) -> tuple[list[GenerationSpan], list[SessionTimeline]]:
    spans = []
    for adapter in adapters:
        try:
            spans.extend(adapter.collect(max_sessions=tasks, min_timestamp=min_timestamp))
        except Exception as e:
            logger.warning("Adapter '%s' failed to collect spans: %s", adapter.name, e)
    spans.sort(key=lambda s: s.ended_at)
    # Recent-session previews are independent of the throughput cutoff.
    return spans, collect_timelines(adapters, max_sessions=5)


def print_stats_dashboard(adapters: list[BaseAdapter], spans: list[GenerationSpan],
                          timelines: list[SessionTimeline], windows: list[str], now: float,
                          tasks: int = 64, recent: int = 10,
                          compact: bool | None = None, guide: bool = True) -> None:
    active_names = ", ".join(a.name for a in adapters)
    if guide:
        print(STATS_GUIDE)
    print(f"\n⚡ TokenMon v{__version__} [Agents: {active_names}]")
    print(f"📊 Inspected: {len(spans)} output streams across up to {tasks} sessions\n")
    if not spans:
        print("No generation output streams found in the inspected sessions.")
    for (agent, model), summaries in analyze_agent_model_windows(
            spans, window_names=windows, now=now).items():
        print(f"🤖 Model: \033[1m{model}\033[0m [Agent: {agent}]")
        print(format_table(summaries, compact=compact))
        print()
    valid_spans = [s for s in spans if s.tps is not None]
    if valid_spans and recent > 0:
        recent_count = min(recent, len(valid_spans))
        print(f"📋 Recent {recent_count} Generation Streams:")
        for s in valid_spans[-recent_count:]:
            print(format_recent_span(s, compact=compact))
        print()
    if timelines:
        preview_count = min(3, len(timelines))
        print(f"🔍 Recent Sessions & User Interactions (latest {preview_count}):\n")
        for t in timelines[:preview_count]:
            print(format_session_card(t, now))
            print()
        latest_id = timelines[0].session_id[:12]
        print(f"💡 Tip: Run `tokenmon logs {latest_id}` for full event chronology or `tokenmon timeline` for activity lanes.\n")


def positive_interval(value: str) -> float:
    try:
        interval = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("interval must be a positive, finite number") from None
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("interval must be a positive, finite number")
    return interval


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tokenmon",
        description="Monitor token throughput, session activity, and timelines for local AI coding agents.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. stats (default, aliases: top)
    p_stats = subparsers.add_parser(
        "stats",
        aliases=["top"],
        help="Display token throughput (TPS) and rolling window metrics (default)",
    )
    p_stats.add_argument(
        "agent",
        nargs="?",
        default=None,
        help=f"Target agents, comma-separated ({', '.join(ADAPTER_REGISTRY.keys())}), or 'all'. Default: auto-detect.",
    )
    p_stats.add_argument("-a", "--agent", dest="agent_opt", default=None, help=argparse.SUPPRESS)
    p_stats.add_argument(
        "--window",
        choices=["30m", "1d", "7d", "30d", "all"],
        default=None,
        help="Filter token throughput to a specific time window",
    )
    p_stats.add_argument(
        "--all",
        action="store_true",
        help="Include full history ('all' window) in throughput summary tables",
    )
    p_stats.add_argument(
        "--tasks",
        type=int,
        default=64,
        help="Maximum recent task sessions to inspect (default: 64)",
    )
    p_stats.add_argument(
        "--recent",
        type=int,
        default=10,
        help="Number of recent generation spans to display (default: 10)",
    )
    p_stats.add_argument(
        "--home",
        type=str,
        default=None,
        help="Override data root path for the selected agent adapter",
    )
    p_stats.add_argument(
        "--compact",
        action="store_true",
        help="Use concise, narrow table layout suited for small screens or split panes",
    )
    p_stats.add_argument(
        "--wide",
        action="store_true",
        help="Force full-width table layout with all diagnostic columns",
    )
    p_stats.add_argument(
        "--json",
        action="store_true",
        help="Shortcut for --output json",
    )
    p_stats.add_argument("--output", choices=["terminal", "json", "html"], default="terminal",
                         help="Output format (default: terminal)")
    p_stats.add_argument("-w", "--watch", action="store_true", help="Refresh the complete stats dashboard until Ctrl+C")
    p_stats.add_argument("--interval", type=positive_interval, default=2.0,
                         help="Watch refresh interval in seconds (default: 2)")

    # 2. sessions (aliases: ps, ls)
    p_sessions = subparsers.add_parser(
        "sessions",
        aliases=["ps", "ls"],
        help="List active and recent agent sessions with user activity and idle status",
    )
    p_sessions.add_argument(
        "agent",
        nargs="?",
        default=None,
        help=f"Target agents, comma-separated ({', '.join(ADAPTER_REGISTRY.keys())}), or 'all'. Default: auto-detect.",
    )
    p_sessions.add_argument("-a", "--agent", dest="agent_opt", default=None, help=argparse.SUPPRESS)
    p_sessions.add_argument(
        "--window",
        choices=["30m", "1d", "7d", "30d", "all"],
        default="7d",
        help="Filter sessions to a specific time window (default: 7d)",
    )
    p_sessions.add_argument(
        "--all",
        action="store_true",
        help="Include full history without the 7-day cutoff",
    )
    p_sessions.add_argument(
        "--tasks",
        type=int,
        default=64,
        help="Maximum recent task sessions to inspect (default: 64)",
    )
    p_sessions.add_argument(
        "--home",
        type=str,
        default=None,
        help="Override data root path for the selected agent adapter",
    )
    p_sessions.add_argument(
        "--compact",
        action="store_true",
        help="Use concise, narrow table layout suited for small screens or split panes",
    )
    p_sessions.add_argument(
        "--wide",
        action="store_true",
        help="Force full-width table layout with all diagnostic columns",
    )
    p_sessions.add_argument(
        "--json",
        action="store_true",
        help="Shortcut for --output json",
    )
    p_sessions.add_argument("--output", choices=["terminal", "json", "html"], default="terminal",
                            help="Output format (default: terminal)")

    # 3. visual session timeline
    p_timeline = subparsers.add_parser(
        "timeline",
        help="Display session activity as shared-time-axis visual lanes",
    )
    p_timeline.add_argument(
        "session_id",
        nargs="?",
        default=None,
        help="Optional session ID/prefix or 'latest'; omit to compare recent sessions",
    )
    p_timeline.add_argument("-a", "--agent", dest="agent_opt", default=None,
                            help="Target agents, comma-separated, or 'all'")
    p_timeline.add_argument(
        "--window",
        choices=["30m", "1d", "7d", "30d", "all"],
        default=None,
        help="Filter sessions and activity to a time window (default: 30d)",
    )
    p_timeline.add_argument(
        "--all",
        action="store_true",
        help="Include full history without 30-day cutoff",
    )
    p_timeline.add_argument(
        "--tasks",
        type=int,
        default=10,
        help="Maximum recent sessions to display (default: 10)",
    )
    p_timeline.add_argument(
        "--home",
        type=str,
        default=None,
        help="Override data root path for the selected agent adapter",
    )
    p_timeline.add_argument(
        "--json",
        action="store_true",
        help="Shortcut for --output json",
    )
    p_timeline.add_argument("--output", choices=["terminal", "json", "html"], default="terminal",
                            help="Output format (default: terminal)")
    p_timeline.add_argument("--compact", action="store_true", help="Use a narrow visual layout")
    p_timeline.add_argument("--wide", action="store_true", help="Force the detailed visual layout")

    # 4. chronological event logs (alias: log)
    p_logs = subparsers.add_parser(
        "logs",
        aliases=["log"],
        help="Display detailed chronological events for a session (or batch window export)",
    )
    p_logs.add_argument(
        "session_id", nargs="?", default="latest",
        help="Session ID/prefix or comma-separated agents to inspect (default: latest)",
    )
    p_logs.add_argument("-a", "--agent", dest="agent_opt", default=None,
                        help="Target agents, comma-separated, or 'all'")
    p_logs.add_argument("--window", choices=["30m", "1d", "7d", "30d", "all"], default=None,
                        help="Export logs for all sessions in time window")
    p_logs.add_argument("--all", action="store_true",
                        help="Include full history without 30-day cutoff")
    p_logs.add_argument("--tasks", type=int, default=64,
                        help="Maximum recent task sessions to inspect (default: 64)")
    p_logs.add_argument("--home", type=str, default=None,
                        help="Override data root path for the selected agent adapter")
    p_logs.add_argument("--json", action="store_true", help="Shortcut for --output json")
    p_logs.add_argument("--output", choices=["terminal", "json", "html"], default="terminal",
                        help="Output format (default: terminal)")
    p_logs.add_argument("-f", "--follow", action="store_true",
                        help="Print only new events from the selected session until Ctrl+C")
    p_logs.add_argument("--interval", type=positive_interval, default=2.0,
                        help="Follow polling interval in seconds (default: 2)")

    # 5. interactive (aliases: repl, shell)
    p_interactive = subparsers.add_parser(
        "interactive",
        aliases=["repl", "shell"],
        help="Launch interactive terminal shell (using stdlib cmd)",
    )
    p_interactive.add_argument(
        "agent",
        nargs="?",
        default=None,
        help=f"Target agents, comma-separated ({', '.join(ADAPTER_REGISTRY.keys())}), or 'all'. Default: auto-detect.",
    )
    p_interactive.add_argument("-a", "--agent", dest="agent_opt", default=None, help=argparse.SUPPRESS)
    p_interactive.add_argument(
        "--home",
        type=str,
        default=None,
        help="Override data root path for the selected agent adapter",
    )

    raw_args = list(sys.argv[1:])
    known_cmds = {
        "stats", "top",
        "sessions", "ps", "ls",
        "timeline", "logs", "log",
        "interactive", "repl", "shell",
        "-h", "--help", "-v", "--version",
    }

    if "-i" in raw_args or "--interactive" in raw_args:
        filtered = [a for a in raw_args if a not in ("-i", "--interactive")]
        raw_args = ["interactive"] + filtered
    elif not raw_args:
        raw_args = ["stats"]
    elif raw_args[0] not in known_cmds:
        raw_args.insert(0, "stats")

    args = parser.parse_args(raw_args)

    cmd = args.command
    if cmd in {"top"}:
        cmd = "stats"
    elif cmd in {"ps", "ls"}:
        cmd = "sessions"
    elif cmd == "log":
        cmd = "logs"
    elif cmd in {"repl", "shell"}:
        cmd = "interactive"

    output_format = getattr(args, "output", "terminal")
    if getattr(args, "json", False):
        if output_format not in {"terminal", "json"}:
            parser.error("--json cannot be combined with --output html")
        output_format = "json"
    live = getattr(args, "watch", False) or getattr(args, "follow", False)
    if live and output_format != "terminal":
        parser.error("non-terminal output cannot be combined with --watch or --follow")
    if getattr(args, "follow", False) and (args.window is not None or args.session_id == "window"):
        parser.error("--follow selects one session; omit --window and use --all for older history")

    # Determine which adapters to run
    adapters = []
    agent_target = getattr(args, "agent_opt", None) or getattr(args, "agent", None)

    if cmd in {"timeline", "logs"}:
        session_target = getattr(args, "session_id", "latest")
        if cmd == "logs" and session_target and not agent_target and (
                session_target.strip().lower() in ADAPTER_REGISTRY or "," in session_target):
            agent_target = session_target
            session_target = "latest"
    else:
        session_target = None

    agent_names = [name.strip().lower() for name in agent_target.split(",")] if agent_target else []
    if agent_names and agent_names != ["all"]:
        if "all" in agent_names:
            print("Error: 'all' must be used alone, not in a comma-separated agent list.", file=sys.stderr)
            return 1
        for name in agent_names:
            if name not in ADAPTER_REGISTRY:
                print(f"Error: Unknown agent '{name}'. Supported: {list(ADAPTER_REGISTRY.keys())}", file=sys.stderr)
                return 1
        seen_classes: set[type] = set()
        for name in agent_names:
            cls = ADAPTER_REGISTRY[name]
            if cls not in seen_classes:
                seen_classes.add(cls)
                adapters.append(get_adapter(name, root=args.home))
    elif agent_names == ["all"]:
        seen_classes: set[type] = set()
        for cls in ADAPTER_REGISTRY.values():
            if cls in seen_classes:  # aliases (e.g. 'agy') share a class
                continue
            seen_classes.add(cls)
            inst = cls(root=args.home)
            if inst.detect():
                adapters.append(inst)
    else:
        detected = detect_available_adapters(max_age_days=30.0, root=args.home)
        if not detected:
            print(
                "No local AI coding agent environments detected (e.g. ~/.codex). "
                "Specify path with --home or check installation.",
                file=sys.stderr,
            )
            return 1
        adapters.extend(detected)

    # Handle interactive shell
    if cmd == "interactive":
        from tokenmon.terminal import run_shell
        return run_shell(adapters, home=args.home)

    now = time.time()

    # Compute cutoff timestamp (default 30 days unless --all or window is 'all')
    window_val = getattr(args, "window", None)
    all_val = getattr(args, "all", False)
    min_ts = collection_cutoff(window_val, all_val, now)

    if getattr(args, "watch", False):
        from tokenmon.live import watch_stats
        compact = True if args.compact else (False if args.wide else None)
        return watch_stats(adapters, interval=args.interval, window=window_val,
                           include_all=all_val, tasks=args.tasks, recent=args.recent, compact=compact)

    # Print ASCII banner in human terminal mode
    if output_format == "terminal" and not getattr(args, "follow", False):
        print_banner()

    # Handle visual timeline command
    if cmd == "timeline":
        timelines = collect_timelines(adapters, max_sessions=args.tasks, min_timestamp=min_ts)
        if not timelines and (not session_target or session_target == "latest"):
            print("No sessions found to visualize.", file=sys.stderr)
            return 1

        if session_target:
            selected = timelines[0] if session_target == "latest" and timelines else next(
                (timeline for timeline in timelines if timeline.session_id == session_target), None)
            if selected is None and session_target != "latest":
                for adapter in adapters:
                    try:
                        candidate = adapter.read_session(session_target)
                    except Exception as e:
                        logger.error("Adapter '%s' error reading session: %s", adapter.name, e)
                        continue
                    if (candidate is not None and candidate.session_id == session_target
                            and (min_ts is None or candidate.updated_at >= min_ts)):
                        selected = candidate
                        break
            if selected is None and session_target != "latest":
                selected = next((timeline for timeline in timelines
                                 if timeline.session_id.startswith(session_target)
                                 or session_target in timeline.session_id), None)
            if selected is None:
                print(f"Error: Session matching '{session_target}' not found.", file=sys.stderr)
                return 1
            timelines = [selected]
        else:
            timelines = timelines[:args.tasks]

        spans: list[GenerationSpan] = []
        for adapter in adapters:
            try:
                spans.extend(adapter.collect(max_sessions=args.tasks, min_timestamp=min_ts))
            except Exception as e:
                logger.warning("Adapter '%s' failed to collect spans for timeline: %s", adapter.name, e)
        lanes = build_timeline_lanes(timelines, spans)
        if output_format == "json":
            print(json.dumps(visual_timeline_to_dict(lanes, adapters, now), indent=2))
            return 0
        if output_format == "html":
            from tokenmon.html import render_timeline_report
            print(render_timeline_report(lanes, now))
            return 0
        compact_flag = True if args.compact else (False if args.wide else None)
        print()
        print(format_visual_timeline(lanes, compact=compact_flag))
        print("\nTip: Run `tokenmon logs <SESSION_ID>` for chronological event details.\n")
        return 0

    # Handle chronological logs command
    if cmd == "logs":
        timelines = []
        for adapter in adapters:
            try:
                timelines.extend(adapter.collect_sessions(max_sessions=args.tasks, min_timestamp=min_ts))
            except Exception as e:
                logger.error("Adapter '%s' error collecting sessions: %s", adapter.name, e)
                print(f"Warning: Adapter '{adapter.name}' failed to parse sessions: {e}", file=sys.stderr)
        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        if session_target not in {"latest", "window"}:
            exact = next((t for t in timelines if t.session_id == session_target), None)
            if exact is None:
                for adapter in adapters:
                    try:
                        candidate = adapter.read_session(session_target)
                    except Exception as e:
                        logger.error("Adapter '%s' error reading session: %s", adapter.name, e)
                        continue
                    if (candidate is not None and candidate.session_id == session_target
                            and (min_ts is None or candidate.updated_at >= min_ts)):
                        timelines.insert(0, candidate)
                        break
        if not timelines:
            print("No sessions found to inspect logs.", file=sys.stderr)
            return 1

        def timeline_to_dict(t: SessionTimeline) -> dict:
            return {
                "session_id": t.session_id,
                "agent": t.agent,
                "cwd": t.cwd,
                "model": t.model,
                **metadata_to_dict(t),
                "created_at": t.created_at,
                "updated_at": t.updated_at,
                "user_messages": t.user_messages,
                "assistant_messages": t.assistant_messages,
                "tool_calls": t.tool_calls,
                "total_tokens": t.total_tokens,
                "duration_seconds": round(t.session_duration, 2),
                "idle_time_seconds": round(t.idle_time(now), 2),
                "status": t.status(now),
                "events": [
                    {
                        "timestamp": ev.timestamp,
                        "kind": ev.kind,
                        "turn_id": ev.turn_id,
                        "summary": ev.summary,
                        "tokens": ev.tokens,
                        "duration": ev.duration,
                        **metadata_to_dict(ev),
                    }
                    for ev in t.events
                ],
            }

        # Multi-session timeline export when --window is explicitly set with default latest or 'window'
        export_window = (window_val is not None and session_target in {"latest", "window"}) or (session_target == "window")

        if export_window:
            if output_format == "json":
                print(json.dumps([timeline_to_dict(t) for t in timelines], indent=2))
                return 0
            if output_format == "html":
                from tokenmon.html import render_logs_report
                print(render_logs_report(timelines, now))
                return 0

            print(f"\n🔍 Session Timelines ({len(timelines)} sessions within window):\n")
            for t in timelines:
                print(format_timeline_view(t, now))
                print("\n" + "─" * 60 + "\n")
            return 0

        # Single session timeline
        selected = next((t for t in timelines if t.session_id == session_target), None)
        if session_target == "latest":
            selected = timelines[0]
        else:
            if selected is None:
                for t in timelines:
                    if t.session_id.startswith(session_target) or session_target in t.session_id:
                        selected = t
                        break
            if not selected:
                print(f"Error: Session matching '{session_target}' not found.", file=sys.stderr)
                return 1

        if output_format == "json":
            print(json.dumps(timeline_to_dict(selected), indent=2))
            return 0
        if output_format == "html":
            from tokenmon.html import render_logs_report
            print(render_logs_report([selected], now))
            return 0

        if args.follow:
            from tokenmon.live import follow_session
            adapter = next(a for a in adapters if a.name == selected.agent)
            return follow_session(adapter, selected, interval=args.interval)

        print()
        print(format_timeline_view(selected, now))
        print()
        return 0

    # Handle sessions command
    if cmd == "sessions":
        timelines = []
        for adapter in adapters:
            try:
                timelines.extend(adapter.collect_sessions(max_sessions=args.tasks, min_timestamp=min_ts))
            except Exception as e:
                logger.error("Adapter '%s' error collecting sessions: %s", adapter.name, e)
                print(f"Warning: Adapter '{adapter.name}' failed to parse sessions: {e}", file=sys.stderr)

        timelines.sort(key=lambda t: t.updated_at, reverse=True)

        if output_format == "json":
            out = [
                {
                    "session_id": t.session_id,
                    "agent": t.agent,
                    "cwd": t.cwd,
                    "model": t.model,
                    **metadata_to_dict(t),
                    "created_at": t.created_at,
                    "updated_at": t.updated_at,
                    "user_messages": t.user_messages,
                    "assistant_messages": t.assistant_messages,
                    "tool_calls": t.tool_calls,
                    "total_tokens": t.total_tokens,
                    "duration_seconds": round(t.session_duration, 2),
                    "idle_time_seconds": round(t.idle_time(now), 2),
                    "status": t.status(now),
                }
                for t in timelines
            ]
            print(json.dumps(out, indent=2))
            return 0
        if output_format == "html":
            from tokenmon.html import render_sessions_report
            print(render_sessions_report(timelines, now))
            return 0

        compact_flag = True if args.compact else (False if args.wide else None)
        print(f"\n📂 Active & Recent Sessions ({len(timelines)} found):")
        if timelines:
            print(format_sessions_table(timelines, now, compact=compact_flag))
            print("Tip: Run `tokenmon timeline` for activity lanes or `tokenmon logs <SESSION_ID>` for event details.\n")
        else:
            print("No sessions found.")
        return 0

    # Default Mode: Collect one snapshot for throughput and recent activity.
    all_spans, timelines = collect_stats(adapters, args.tasks, min_ts)
    windows = stats_windows(args.window, args.all)
    analysis = analyze_windows(all_spans, window_names=windows, now=now)
    valid_spans = [s for s in all_spans if s.tps is not None]

    if output_format == "json":
        # 1. Models throughput breakdown
        models_json = {}
        for model, summaries in analysis.items():
            model_spans = [s for s in all_spans if s.model == model]
            models_json[model] = {
                st.window_name: {
                    "total_spans": st.total_spans,
                    "valid_spans": st.valid_spans,
                    "excluded_spans": st.excluded_spans,
                    "total_tokens": st.total_tokens,
                    "total_duration_seconds": round(st.total_duration, 3),
                    "weighted_tps": round(st.weighted_tps, 2) if st.weighted_tps is not None else None,
                    "median_tps": round(st.median_tps, 2) if st.median_tps is not None else None,
                    "min_tps": round(st.min_tps, 2) if st.min_tps is not None else None,
                    "max_tps": round(st.max_tps, 2) if st.max_tps is not None else None,
                    **configuration_metadata(filter_by_window(model_spans, WINDOW_DURATIONS[st.window_name], now)),
                }
                for st in summaries
            }

        # Overall summary across all models
        overall_summary = {}
        for win in windows:
            dur = WINDOW_DURATIONS[win]
            win_spans = filter_by_window(all_spans, dur, now)
            st = summarize_spans(win_spans, win, "all_models")
            overall_summary[win] = {
                "total_spans": st.total_spans,
                "valid_spans": st.valid_spans,
                "excluded_spans": st.excluded_spans,
                "total_tokens": st.total_tokens,
                "total_duration_seconds": round(st.total_duration, 3),
                "weighted_tps": round(st.weighted_tps, 2) if st.weighted_tps is not None else None,
                "median_tps": round(st.median_tps, 2) if st.median_tps is not None else None,
                "min_tps": round(st.min_tps, 2) if st.min_tps is not None else None,
                "max_tps": round(st.max_tps, 2) if st.max_tps is not None else None,
                **configuration_metadata(win_spans),
            }

        # 2. Recent generation streams
        recent_count = min(args.recent, len(valid_spans)) if args.recent > 0 else 0
        recent_streams_json = [
            {
                "timestamp": s.ended_at,
                "agent": s.agent,
                "session_id": s.session_id,
                "turn_id": s.turn_id,
                "model": s.model,
                "tokens": s.tokens,
                "duration": round(s.duration, 3),
                "tps": round(s.tps, 2) if s.tps is not None else None,
                "timing_source": s.timing_source,
                **metadata_to_dict(s),
            }
            for s in valid_spans[-recent_count:]
        ] if recent_count > 0 else []

        # 3. Recent session cards
        preview_count = min(3, len(timelines))
        recent_sessions_json = [
            {
                "session_id": t.session_id,
                "agent": t.agent,
                "cwd": t.cwd,
                "model": t.model,
                **metadata_to_dict(t),
                "status": t.status(now),
                "user_messages": t.user_messages,
                "assistant_messages": t.assistant_messages,
                "tool_calls": t.tool_calls,
                "total_tokens": t.total_tokens,
                "duration_seconds": round(t.session_duration, 2),
                "created_at": t.created_at,
                "updated_at": t.updated_at,
            }
            for t in timelines[:preview_count]
        ]

        full_output = {
            "meta": {
                "version": __version__,
                "agents": [a.name for a in adapters],
                "inspected_spans": len(all_spans),
                "tasks_limit": args.tasks,
                "timestamp": now,
            },
            "summary": overall_summary,
            "models": models_json,
            "recent_streams": recent_streams_json,
            "recent_sessions": recent_sessions_json,
        }
        print(json.dumps(full_output, indent=2))
        return 0

    if output_format == "html":
        from tokenmon.html import render_stats_report
        print(render_stats_report([adapter.name for adapter in adapters], all_spans, timelines,
                                  windows, now, args.tasks))
        return 0

    compact_flag = True if args.compact else (False if args.wide else None)
    print_stats_dashboard(adapters, all_spans, timelines, windows, now,
                          tasks=args.tasks, recent=args.recent, compact=compact_flag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
