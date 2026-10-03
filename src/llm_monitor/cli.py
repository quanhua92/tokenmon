"""Command line interface for llm-monitor."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone

from llm_monitor import __version__
from llm_monitor.adapters import ADAPTER_REGISTRY, detect_available_adapters, get_adapter
from llm_monitor.analyzer import WINDOW_DURATIONS, analyze_windows, filter_by_window, summarize_spans
from llm_monitor.models import GenerationSpan, SessionTimeline, WindowSummary

logger = logging.getLogger(__name__)


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
            "Time (s)",
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
                    f"{s.total_duration:.1f}",
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
        return f"{hours}h {mins:02d}m"
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
                    t.model[:12],
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
                    t.model,
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

    if compact:
        t_str = dt.strftime("%H:%M:%S")
        m_str = s.model[:12]
        return (
            f"  {t_str}  {m_str:<12}  "
            f"\033[1;32m{tps_val:5.1f} TPS\033[0m  "
            f"({s.tokens:,} tok / {s.duration:.1f}s)"
        )
    else:
        t_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"  [{t_str}] {s.model:<18} : "
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
        f"  📌 \033[1m{t.session_id[:12]}\033[0m  ({t.model})  {status_badge}",
        f"     {user_part}",
        f"     {agent_part}",
        f"     {time_part}",
    ]
    return "\n".join(lines)


def format_timeline_view(timeline: SessionTimeline, now: float) -> str:
    lines = [
        f"🔍 Session Timeline: \033[1m{timeline.session_id}\033[0m ({timeline.model})",
        f"Activity: {timeline.user_messages} User Prompts | {timeline.assistant_messages} Assistant Responses | {timeline.tool_calls} Tool Calls | {timeline.total_tokens:,} Tokens",
        f"Status: \033[1m{timeline.status(now)}\033[0m\n",
    ]

    event_icons = {
        "user_message": "👤 \033[1;34mUSER\033[0m     ",
        "assistant_message": "🤖 \033[1;32mASSISTANT\033[0m",
        "reasoning": "🧠 \033[1;35mTHINKING\033[0m ",
        "tool_call": "🛠️  \033[1;33mTOOL CALL\033[0m",
        "tool_output": "⚙️  \033[1;33mTOOL DONE\033[0m",
        "turn_start": "▶️  \033[2mSTART\033[0m    ",
        "turn_end": "⏹️  \033[2mCOMPLETE\033[0m ",
    }

    for ev in timeline.events:
        dt = datetime.fromtimestamp(ev.timestamp, tz=timezone.utc).astimezone()
        t_str = dt.strftime("%H:%M:%S")
        tag = event_icons.get(ev.kind, f"•  {ev.kind:<9}")
        dur_str = f" [{ev.duration:.2f}s]" if ev.duration is not None else ""
        lines.append(f"  {t_str} │ {tag} │ {ev.summary}{dur_str}")

    return "\n".join(lines)


ASCII_LOGO = r"""
   __    __   __  ___
  / /   / /  /  |/  /   M O N I T O R
 / /___/ /__/ /|_/ /    ── ⚡ Agent TPS ──
/_____/____/_/  /_/
""".strip("\n")


STATS_GUIDE = """\
📖 How to read this
  • TPS is output tokens per second of pure generation. Tool runs and waiting are not counted.
  • TPS in the table is total tokens ÷ total seconds, not an average of per-stream speeds.
  • Outputs "valid/total": streams under 1s or with unclear timing count in total but not in TPS.
  • Median is the middle stream. It is less affected by one very slow or very fast stream.
  • Windows are 30m, 1d, 7d and 30d. Use --window to pick one, or --all for full history.

➡️  Next: `llm-monitor ps` lists sessions · `llm-monitor logs` shows a timeline · add --json for scripts.
"""


def print_banner(color: bool = True) -> None:
    """Print a compact, modern ASCII banner suitable for standard and narrow split panes."""
    lines = ASCII_LOGO.splitlines()
    if color and sys.stdout.isatty():
        green = "\033[1;32m"
        reset = "\033[0m"
        dim = "\033[2m"
        print()
        print(f"{green}{lines[0]}{reset}")
        print(f"{green}{lines[1][:21]}{reset}\033[1m{lines[1][21:]}{reset}")
        print(f"{green}{lines[2][:21]}{reset}{dim}{lines[2][21:]}{reset}")
        print(f"{green}{lines[3]}{reset}")
        print()
    else:
        print(f"\n{ASCII_LOGO}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="llm-monitor",
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
        "-w",
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
        "-w",
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
        "-w",
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
        from llm_monitor.terminal import run_shell
        return run_shell(adapters, home=args.home)

    now = time.time()

    # Compute cutoff timestamp (default 30 days unless --all or window is 'all')
    min_ts: float | None = None
    window_val = getattr(args, "window", None)
    all_val = getattr(args, "all", False)
    if not all_val and window_val != "all":
        if window_val:
            min_ts = now - WINDOW_DURATIONS[window_val]
        else:
            min_ts = now - (30 * 86400.0)

    # Print ASCII banner in human terminal mode
    if not getattr(args, "json", False):
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
            print("Tip: Run `llm-monitor timeline <SESSION_ID>` (or `llm-monitor logs <SESSION_ID>`) to see full event chronology.\n")
        else:
            print("No sessions found.")
        return 0

    # Default Mode: Collect spans for throughput / TPS monitoring
    all_spans: list[GenerationSpan] = []
    for adapter in adapters:
        try:
            all_spans.extend(adapter.collect(max_sessions=args.tasks, min_timestamp=min_ts))
        except Exception as e:
            logger.error("Adapter '%s' error collecting generation spans: %s", adapter.name, e)
            print(f"Warning: Adapter '{adapter.name}' failed to parse spans: {e}", file=sys.stderr)

    all_spans.sort(key=lambda s: s.ended_at)
    if args.window:
        windows = [args.window]
    elif args.all:
        windows = ["30m", "1d", "7d", "30d", "all"]
    else:
        windows = ["30m", "1d", "7d", "30d"]

    analysis = analyze_windows(all_spans, window_names=windows, now=now)
    valid_spans = [s for s in all_spans if s.tps is not None]

    # Collect recent sessions preview
    timelines: list[SessionTimeline] = []
    for adapter in adapters:
        try:
            timelines.extend(adapter.collect_sessions(max_sessions=5))
        except Exception as e:
            logger.error("Adapter '%s' error collecting preview sessions: %s", adapter.name, e)
    timelines.sort(key=lambda t: t.updated_at, reverse=True)

    if args.json:
        # 1. Models throughput breakdown
        models_json = {}
        for model, summaries in analysis.items():
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

    # Human-readable terminal output
    active_names = ", ".join(a.name for a in adapters)
    print(f"\n⚡ llm-monitor v{__version__} [Agents: {active_names}]")
    print(f"📊 Inspected: {len(all_spans)} output streams across up to {args.tasks} sessions\n")

    if not all_spans:
        print("No generation output streams found in the inspected sessions.")
        return 0

    compact_flag = True if args.compact else (False if args.wide else None)
    for model, summaries in analysis.items():
        print(f"🤖 Model: \033[1m{model}\033[0m")
        print(format_table(summaries, compact=compact_flag))
        print()

    # Recent outputs breakdown
    if valid_spans and args.recent > 0:
        recent_count = min(args.recent, len(valid_spans))
        print(f"📋 Recent {recent_count} Generation Streams:")
        for s in valid_spans[-recent_count:]:
            print(format_recent_span(s, compact=compact_flag))
        print()

    # Recent Sessions & User Interactions overview
    if timelines:
        preview_count = min(3, len(timelines))
        print(f"🔍 Recent Sessions & User Interactions (latest {preview_count}):\n")
        for t in timelines[:preview_count]:
            print(format_session_card(t, now))
            print()
        latest_id = timelines[0].session_id[:12]
        print(f"💡 Tip: Run `llm-monitor timeline {latest_id}` for full step-by-step chronology.\n")

    print(STATS_GUIDE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
