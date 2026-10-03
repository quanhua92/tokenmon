"""Command line interface for llm-monitor."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone

from llm_monitor import __version__
from llm_monitor.adapters import ADAPTER_REGISTRY, detect_available_adapters, get_adapter
from llm_monitor.analyzer import WINDOW_DURATIONS, analyze_windows, filter_by_window, summarize_spans
from llm_monitor.models import GenerationSpan, SessionTimeline, WindowSummary


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
            mins = int(t.session_duration // 60)
            secs = int(t.session_duration % 60)
            dur_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
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
            mins = int(t.session_duration // 60)
            secs = int(t.session_duration % 60)
            dur_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="llm-monitor",
        description="Monitor token throughput, session activity, and timelines for local AI coding agents.",
    )
    parser.add_argument(
        "agent",
        nargs="?",
        default=None,
        help=f"Target agent adapter ({', '.join(ADAPTER_REGISTRY.keys())}, or 'all'). Default: auto-detect.",
    )
    parser.add_argument(
        "--window",
        choices=["30m", "1d", "7d", "all"],
        default=None,
        help="Filter token throughput to a specific time window",
    )
    parser.add_argument(
        "--sessions",
        action="store_true",
        help="List recent sessions with user activity and unattended status",
    )
    parser.add_argument(
        "--timeline",
        nargs="?",
        const="latest",
        default=None,
        metavar="SESSION_ID",
        help="Display detailed chronological timeline of a session (default: latest)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=64,
        help="Maximum recent task sessions to inspect (default: 64)",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=10,
        help="Number of recent generation spans to display (default: 10)",
    )
    parser.add_argument(
        "--home",
        type=str,
        default=None,
        help="Override data root path for the selected agent adapter",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw machine-readable JSON",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive terminal shell (using stdlib cmd)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Use concise, narrow table layout suited for small screens or split panes",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Force full-width table layout with all diagnostic columns",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args()

    # Determine which adapters to run
    adapters = []
    is_interactive_cmd = args.agent and args.agent.lower() in {"interactive", "repl", "shell"}
    agent_target = None if is_interactive_cmd else args.agent

    if agent_target and agent_target.lower() != "all":
        target = get_adapter(agent_target, root=args.home)
        if not target:
            print(f"Error: Unknown agent '{agent_target}'. Supported: {list(ADAPTER_REGISTRY.keys())}", file=sys.stderr)
            return 1
        adapters.append(target)
    elif agent_target and agent_target.lower() == "all":
        for cls in ADAPTER_REGISTRY.values():
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
    if args.interactive or is_interactive_cmd:
        from llm_monitor.terminal import run_shell
        return run_shell(adapters, home=args.home)

    now = time.time()

    # Handle --timeline mode
    if args.timeline:
        timelines = []
        for adapter in adapters:
            timelines.extend(adapter.collect_sessions(max_sessions=args.tasks))
        if not timelines:
            print("No sessions found to inspect timeline.", file=sys.stderr)
            return 1

        selected = None
        if args.timeline == "latest":
            selected = timelines[0]
        else:
            for t in timelines:
                if t.session_id.startswith(args.timeline) or args.timeline in t.session_id:
                    selected = t
                    break
            if not selected:
                print(f"Error: Session matching '{args.timeline}' not found.", file=sys.stderr)
                return 1

        if args.json:
            out = {
                "session_id": selected.session_id,
                "agent": selected.agent,
                "model": selected.model,
                "created_at": selected.created_at,
                "updated_at": selected.updated_at,
                "user_messages": selected.user_messages,
                "assistant_messages": selected.assistant_messages,
                "tool_calls": selected.tool_calls,
                "total_tokens": selected.total_tokens,
                "status": selected.status(now),
                "events": [
                    {
                        "timestamp": ev.timestamp,
                        "kind": ev.kind,
                        "turn_id": ev.turn_id,
                        "summary": ev.summary,
                        "tokens": ev.tokens,
                        "duration": ev.duration,
                    }
                    for ev in selected.events
                ],
            }
            print(json.dumps(out, indent=2))
            return 0

        print()
        print(format_timeline_view(selected, now))
        print()
        return 0

    # Handle --sessions mode
    if args.sessions:
        timelines = []
        for adapter in adapters:
            timelines.extend(adapter.collect_sessions(max_sessions=args.tasks))

        if args.json:
            out = [
                {
                    "session_id": t.session_id,
                    "agent": t.agent,
                    "model": t.model,
                    "created_at": t.created_at,
                    "updated_at": t.updated_at,
                    "user_messages": t.user_messages,
                    "assistant_messages": t.assistant_messages,
                    "tool_calls": t.tool_calls,
                    "total_tokens": t.total_tokens,
                    "duration_seconds": round(t.session_duration, 2),
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
            print("Tip: Run `llm-monitor --timeline <SESSION_ID>` to see full event chronology.\n")
        else:
            print("No sessions found.")
        return 0

    # Default Mode: Collect spans for throughput / TPS monitoring
    all_spans: list[GenerationSpan] = []
    for adapter in adapters:
        all_spans.extend(adapter.collect(max_sessions=args.tasks))

    all_spans.sort(key=lambda s: s.ended_at)
    windows = [args.window] if args.window else ["30m", "1d", "7d", "all"]

    if args.json:
        grouped_results = {}
        for win in windows:
            dur = WINDOW_DURATIONS[win]
            win_spans = filter_by_window(all_spans, dur, now)
            st = summarize_spans(win_spans, win, "all_models")
            grouped_results[win] = {
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
        print(json.dumps(grouped_results, indent=2))
        return 0

    # Human-readable terminal output
    active_names = ", ".join(a.name for a in adapters)
    print(f"\n⚡ llm-monitor v{__version__} [Agents: {active_names}]")
    print(f"📊 Inspected: {len(all_spans)} output streams across up to {args.tasks} sessions\n")

    if not all_spans:
        print("No generation output streams found in the inspected sessions.")
        return 0

    compact_flag = True if args.compact else (False if args.wide else None)
    analysis = analyze_windows(all_spans, window_names=windows, now=now)
    for model, summaries in analysis.items():
        print(f"🤖 Model: \033[1m{model}\033[0m")
        print(format_table(summaries, compact=compact_flag))
        print()

    # Recent outputs breakdown
    valid_spans = [s for s in all_spans if s.tps is not None]
    if valid_spans and args.recent > 0:
        recent_count = min(args.recent, len(valid_spans))
        print(f"📋 Recent {recent_count} Generation Streams:")
        for s in valid_spans[-recent_count:]:
            print(format_recent_span(s, compact=compact_flag))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
