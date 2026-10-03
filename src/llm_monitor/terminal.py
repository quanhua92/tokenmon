"""Interactive terminal shell built with standard library cmd.Cmd."""

from __future__ import annotations

import cmd
import os
import readline
import sys
import time
from datetime import datetime, timezone

from llm_monitor import __version__
from llm_monitor.adapters import BaseAdapter, detect_available_adapters, get_adapter
from llm_monitor.analyzer import WINDOW_DURATIONS, analyze_windows, filter_by_window, summarize_spans
from llm_monitor.cli import ASCII_LOGO, format_recent_span, format_sessions_table, format_table, format_timeline_view
from llm_monitor.models import GenerationSpan, SessionTimeline


class MonitorShell(cmd.Cmd):
    """Interactive command-line shell for llm-monitor using Python standard library cmd."""

    intro = (
        f"\n\033[1;32m{ASCII_LOGO}\033[0m\n\n"
        f"\033[1;32m⚡ llm-monitor interactive shell v{__version__}\033[0m\n"
        "Type \033[1mhelp\033[0m or \033[1m?\033[0m to list commands, or \033[1mexit\033[0m to quit.\n"
        "Tab-completion is enabled for commands and session IDs.\n"
    )
    prompt = "\033[1;34m(llm-monitor)\033[0m "

    def __init__(self, adapters: list[BaseAdapter], home: str | None = None):
        super().__init__()
        self.adapters = adapters
        self.home = home
        self._cached_timelines: list[SessionTimeline] = []
        self._last_refresh = 0.0

    def _get_spans(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]:
        all_spans = []
        for a in self.adapters:
            try:
                all_spans.extend(a.collect(max_sessions=max_sessions, min_timestamp=min_timestamp))
            except Exception as e:
                print(f"Warning: Adapter '{a.name}' failed to collect spans: {e}", file=sys.stderr)
        all_spans.sort(key=lambda s: s.ended_at)
        return all_spans

    def _get_timelines(self, max_sessions: int = 32, min_timestamp: float | None = None, force: bool = False) -> list[SessionTimeline]:
        now = time.time()
        if not force and self._cached_timelines and (now - self._last_refresh < 3.0):
            return self._cached_timelines
        timelines = []
        for a in self.adapters:
            try:
                timelines.extend(a.collect_sessions(max_sessions=max_sessions, min_timestamp=min_timestamp))
            except Exception as e:
                print(f"Warning: Adapter '{a.name}' failed to collect sessions: {e}", file=sys.stderr)
        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        self._cached_timelines = timelines
        self._last_refresh = now
        return timelines

    # -----------------------------------------------------------------------
    # Shell Commands
    # -----------------------------------------------------------------------

    def do_summary(self, arg: str):
        """Show token throughput and TPS metrics across windows.
Usage: summary [30m|1d|7d|30d|all]
Alias: s"""
        arg_clean = arg.strip()
        min_ts: float | None = None
        if arg_clean in WINDOW_DURATIONS:
            windows = [arg_clean]
            if arg_clean != "all":
                min_ts = time.time() - WINDOW_DURATIONS[arg_clean]
        elif arg_clean == "all":
            windows = ["30m", "1d", "7d", "30d", "all"]
        else:
            windows = ["30m", "1d", "7d", "30d"]
            min_ts = time.time() - (30 * 86400.0)

        spans = self._get_spans(min_timestamp=min_ts)
        if not spans:
            print("No output streams found.")
            return

        analysis = analyze_windows(spans, window_names=windows, now=time.time())

        for model, summaries in analysis.items():
            print(f"\n🤖 Model: \033[1m{model}\033[0m")
            print(format_table(summaries))
        print()

    def complete_summary(self, text, line, begidx, endidx):
        return [w for w in WINDOW_DURATIONS.keys() if w.startswith(text)]

    do_s = do_summary

    def do_sessions(self, arg: str):
        """List active, idle, and unattended sessions with activity stats.
Usage: sessions [max_count]
Alias: ls"""
        limit = int(arg) if arg.strip().isdigit() else 20
        timelines = self._get_timelines(max_sessions=limit, force=True)
        if not timelines:
            print("No sessions found.")
            return

        now = time.time()
        print(f"\n📂 Active & Recent Sessions ({len(timelines)} found):")
        print(format_sessions_table(timelines[:limit], now))
        print("Tip: Use 'timeline <SESSION_ID>' to view chronological events.\n")

    do_ls = do_sessions

    def do_timeline(self, arg: str):
        """Display step-by-step chronology of a session (user prompts, assistant answers, tools).
Usage: timeline [SESSION_ID|latest]
Alias: t"""
        target = arg.strip() or "latest"
        timelines = self._get_timelines(force=True)
        if not timelines:
            print("No sessions found.")
            return

        selected = None
        if target == "latest":
            selected = timelines[0]
        else:
            for t in timelines:
                if t.session_id.startswith(target) or target in t.session_id:
                    selected = t
                    break

        if not selected:
            print(f"Error: Session '{target}' not found. Use 'sessions' to see available IDs.")
            return

        print()
        print(format_timeline_view(selected, time.time()))
        print()

    def complete_timeline(self, text, line, begidx, endidx):
        timelines = self._get_timelines()
        ids = ["latest"] + [t.session_id for t in timelines]
        return [i for i in ids if i.startswith(text)]

    do_t = do_timeline

    def do_recent(self, arg: str):
        """Display recent generation output streams with instantaneous TPS.
Usage: recent [count]
Alias: r"""
        count = int(arg) if arg.strip().isdigit() else 10
        spans = self._get_spans()
        valid_spans = [s for s in spans if s.tps is not None]
        if not valid_spans:
            print("No valid generation streams found.")
            return

        print(f"\n📋 Last {min(count, len(valid_spans))} Generation Streams:")
        for s in valid_spans[-count:]:
            print(format_recent_span(s))
        print()

    do_r = do_recent

    def do_watch(self, arg: str):
        """Live auto-refresh dashboard mode. Press Ctrl+C to stop.
Usage: watch [interval_seconds] [window]"""
        parts = arg.strip().split()
        interval = 2.0
        window = "all"
        if len(parts) >= 1 and parts[0].replace(".", "", 1).isdigit():
            interval = max(0.5, float(parts[0]))
        if len(parts) >= 2 and parts[1] in WINDOW_DURATIONS:
            window = parts[1]

        print(f"Starting live watch every {interval}s on window '{window}' (Press Ctrl+C to exit)...")
        time.sleep(0.5)

        try:
            while True:
                # Clear terminal screen
                print("\033[2J\033[H", end="")
                now = time.time()
                spans = self._get_spans()
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"⚡ llm-monitor Live Dashboard [{now_str}] (Interval: {interval}s | Ctrl+C to return)\n")

                if spans:
                    analysis = analyze_windows(spans, window_names=[window], now=now)
                    for model, summaries in analysis.items():
                        print(f"🤖 Model: \033[1m{model}\033[0m")
                        print(format_table(summaries))
                        print()

                timelines = self._get_timelines(max_sessions=5, force=True)
                if timelines:
                    print("📂 Latest Sessions:")
                    print(format_sessions_table(timelines[:5], now))

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nExited watch mode.\n")

    def do_clear(self, arg: str):
        """Clear terminal screen.
Alias: cls"""
        os.system("cls" if os.name == "nt" else "clear")

    do_cls = do_clear

    def do_agents(self, arg: str):
        """Show active and available agent adapters."""
        print("\nSupported Adapters:")
        active = {a.name for a in self.adapters}
        for name in sorted(ADAPTER_REGISTRY.keys()):
            status = "\033[1;32m[Active]\033[0m" if name in active else "\033[2m[Inactive]\033[0m"
            print(f"  • {name:<12} {status}")
        print()

    def do_exit(self, arg: str):
        """Exit the interactive shell.
Alias: quit, q"""
        print("Goodbye!")
        return True

    do_quit = do_exit
    do_q = do_exit

    def emptyline(self):
        """Do nothing on empty line (prevents re-executing last command)."""
        pass

    def default(self, line: str):
        print(f"Unknown command: '{line.strip()}'. Type 'help' to see available commands.")


def run_shell(adapters: list[BaseAdapter], home: str | None = None) -> int:
    """Launch the interactive terminal shell."""
    shell = MonitorShell(adapters, home=home)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    return 0
