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
from tokenmon.analyzer import WINDOW_DURATIONS, analyze_windows, filter_by_window, summarize_spans
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, WindowSummary, recorded_speed_mode

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


def format_session_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    if mins >= 60:
        hours, mins = divmod(mins, 60)
        return f"{hours}h {mins:02d}m {secs:02d}s"
    return f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"


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
    mins = int(t.session_duration // 60)
    if mins >= 60:
        hrs = mins // 60
        rem_m = mins % 60
        dur_str = f"{hrs}h {rem_m:02d}m"
    elif mins > 0:
        dur_str = f"{mins}m"
    else:
        dur_str = f"{int(t.session_duration)}s"

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
  • TPS uses recorded generation boundaries. Claude single-record turn-span estimates can include latency or waiting.
  • TPS in the table is total tokens ÷ total seconds, not an average of per-stream speeds.
  • Outputs "valid/total": streams under 1s or with unclear timing count in total but not in TPS.
  • Median is the middle stream. It is less affected by one very slow or very fast stream.
  • Windows are 30m, 1d, 7d and 30d. Use --window to pick one, or --all for full history.

➡️  Next: `tokenmon ps` lists sessions · `tokenmon logs` shows a timeline · add --json for scripts.
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
    print(f"\n⚡ TokenMon v{__version__} [Agents: {active_names}]")
    print(f"📊 Inspected: {len(spans)} output streams across up to {tasks} sessions\n")
    if not spans:
        print("No generation output streams found in the inspected sessions.")
    for model, summaries in analyze_windows(spans, window_names=windows, now=now).items():
        print(f"🤖 Model: \033[1m{model}\033[0m")
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
        print(f"💡 Tip: Run `tokenmon timeline {latest_id}` for full step-by-step chronology.\n")
    if guide:
        print(STATS_GUIDE)


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
        help=f"Target agent adapter ({', '.join(ADAPTER_REGISTRY.keys())}, or 'all'). Default: auto-detect.",
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
        help="Output raw machine-readable JSON",
    )
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
        help=f"Target agent adapter ({', '.join(ADAPTER_REGISTRY.keys())}, or 'all'). Default: auto-detect.",
    )
    p_sessions.add_argument("-a", "--agent", dest="agent_opt", default=None, help=argparse.SUPPRESS)
    p_sessions.add_argument(
        "--window",
        choices=["30m", "1d", "7d", "30d", "all"],
        default=None,
        help="Filter sessions to a specific time window",
    )
    p_sessions.add_argument(
        "--all",
        action="store_true",
        help="Include full history without 30-day cutoff",
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
        help="Output raw machine-readable JSON",
    )

    # 3. timeline (aliases: log, logs)
    p_timeline = subparsers.add_parser(
        "timeline",
        aliases=["log", "logs"],
        help="Display detailed chronological event timeline for a session (or batch window export)",
    )
    p_timeline.add_argument(
        "session_id",
        nargs="?",
        default="latest",
        help="Session ID or prefix to inspect (default: latest)",
    )
    p_timeline.add_argument("-a", "--agent", dest="agent_opt", default=None, help="Target agent adapter")
    p_timeline.add_argument(
        "--window",
        choices=["30m", "1d", "7d", "30d", "all"],
        default=None,
        help="Export timelines for all sessions in time window",
    )
    p_timeline.add_argument(
        "--all",
        action="store_true",
        help="Include full history without 30-day cutoff",
    )
    p_timeline.add_argument(
        "--tasks",
        type=int,
        default=64,
        help="Maximum recent task sessions to inspect (default: 64)",
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
        help="Output raw machine-readable JSON",
    )
    p_timeline.add_argument("-f", "--follow", action="store_true",
                            help="Print only new events from the selected session until Ctrl+C")
    p_timeline.add_argument("--interval", type=positive_interval, default=2.0,
                            help="Follow polling interval in seconds (default: 2)")

    # 4. interactive (aliases: repl, shell)
    p_interactive = subparsers.add_parser(
        "interactive",
        aliases=["repl", "shell"],
        help="Launch interactive terminal shell (using stdlib cmd)",
    )
    p_interactive.add_argument(
        "agent",
        nargs="?",
        default=None,
        help=f"Target agent adapter ({', '.join(ADAPTER_REGISTRY.keys())}, or 'all'). Default: auto-detect.",
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
        "timeline", "log", "logs",
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
    elif cmd in {"log", "logs"}:
        cmd = "timeline"
    elif cmd in {"repl", "shell"}:
        cmd = "interactive"

    live = getattr(args, "watch", False) or getattr(args, "follow", False)
    if live and args.json:
        parser.error("--json cannot be combined with --watch or --follow")
    if getattr(args, "follow", False) and (args.window is not None or args.session_id == "window"):
        parser.error("--follow selects one session; omit --window and use --all for older history")

    # Determine which adapters to run
    adapters = []
    agent_target = getattr(args, "agent_opt", None) or getattr(args, "agent", None)

    if cmd == "timeline":
        session_target = getattr(args, "session_id", "latest")
        if session_target and session_target.lower() in ADAPTER_REGISTRY and not agent_target:
            agent_target = session_target.lower()
            session_target = "latest"
    else:
        session_target = None

    if agent_target and agent_target.lower() != "all":
        target = get_adapter(agent_target, root=args.home)
        if not target:
            print(f"Error: Unknown agent '{agent_target}'. Supported: {list(ADAPTER_REGISTRY.keys())}", file=sys.stderr)
            return 1
        adapters.append(target)
    elif agent_target and agent_target.lower() == "all":
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
    if not getattr(args, "json", False) and not getattr(args, "follow", False):
        print_banner()

    # Handle timeline command
    if cmd == "timeline":
        timelines = []
        for adapter in adapters:
            try:
                timelines.extend(adapter.collect_sessions(max_sessions=args.tasks, min_timestamp=min_ts))
            except Exception as e:
                logger.error("Adapter '%s' error collecting sessions: %s", adapter.name, e)
                print(f"Warning: Adapter '{adapter.name}' failed to parse sessions: {e}", file=sys.stderr)
        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        if not timelines:
            print("No sessions found to inspect timeline.", file=sys.stderr)
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
            if args.json:
                print(json.dumps([timeline_to_dict(t) for t in timelines], indent=2))
                return 0

            print(f"\n🔍 Session Timelines ({len(timelines)} sessions within window):\n")
            for t in timelines:
                print(format_timeline_view(t, now))
                print("\n" + "─" * 60 + "\n")
            return 0

        # Single session timeline
        selected = None
        if session_target == "latest":
            selected = timelines[0]
        else:
            for t in timelines:
                if t.session_id.startswith(session_target) or session_target in t.session_id:
                    selected = t
                    break
            if not selected:
                print(f"Error: Session matching '{session_target}' not found.", file=sys.stderr)
                return 1

        if args.json:
            print(json.dumps(timeline_to_dict(selected), indent=2))
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

        if args.json:
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

        compact_flag = True if args.compact else (False if args.wide else None)
        print(f"\n📂 Active & Recent Sessions ({len(timelines)} found):")
        if timelines:
            print(format_sessions_table(timelines, now, compact=compact_flag))
            print("Tip: Run `tokenmon timeline <SESSION_ID>` (or `tokenmon logs <SESSION_ID>`) to see full event chronology.\n")
        else:
            print("No sessions found.")
        return 0

    # Default Mode: Collect one snapshot for throughput and recent activity.
    all_spans, timelines = collect_stats(adapters, args.tasks, min_ts)
    windows = stats_windows(args.window, args.all)
    analysis = analyze_windows(all_spans, window_names=windows, now=now)
    valid_spans = [s for s in all_spans if s.tps is not None]

    if args.json:
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

    compact_flag = True if args.compact else (False if args.wide else None)
    print_stats_dashboard(adapters, all_spans, timelines, windows, now,
                          tasks=args.tasks, recent=args.recent, compact=compact_flag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
