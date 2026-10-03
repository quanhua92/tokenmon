"""Read-only live dashboards and append-only session following, without readline."""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from datetime import datetime

from tokenmon.adapters.base import BaseAdapter
from tokenmon.cli import (
    collect_stats, collection_cutoff, format_timeline_event,
    print_stats_dashboard, stats_windows,
)
from tokenmon.models import SessionTimeline, TimelineEvent

logger = logging.getLogger(__name__)


def watch_stats(adapters: list[BaseAdapter], interval: float = 2.0,
                window: str | None = None, include_all: bool = False,
                tasks: int = 64, recent: int = 10,
                compact: bool | None = None, guide: bool = True) -> int:
    """Render fresh, complete stats snapshots until interrupted."""
    try:
        while True:
            now = time.time()
            spans, timelines = collect_stats(adapters, tasks, collection_cutoff(window, include_all, now))
            if sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(f"⚡ TokenMon Live Dashboard [{datetime.fromtimestamp(now).astimezone():%Y-%m-%d %H:%M:%S}] "
                  f"(Interval: {interval}s | Ctrl+C to stop)")
            print_stats_dashboard(adapters, spans, timelines, stats_windows(window, include_all),
                                  now, tasks=tasks, recent=recent, compact=compact, guide=guide)
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nExited watch mode.", flush=True)
    except BrokenPipeError:
        return 0
    return 0


def _event_keys(events: list[TimelineEvent]):
    """Distinguish repeated events, with stable source IDs preferred over content."""
    occurrences: Counter = Counter()
    for event in events:
        identity = ("source", event.event_id) if event.event_id is not None else (
            "fallback", event.timestamp, event.kind, event.turn_id, event.summary,
        )
        occurrences[identity] += 1
        yield (identity, occurrences[identity]), event


def follow_session(adapter: BaseAdapter, initial: SessionTimeline, interval: float = 2.0) -> int:
    """Pin one session and emit only events absent from its startup baseline."""
    seen = {key for key, _ in _event_keys(initial.events)}
    try:
        print(f"Following session {initial.session_id} ({initial.agent}); waiting for new events. "
              "Press Ctrl+C to stop.", flush=True)
        while True:
            time.sleep(interval)
            try:
                current = adapter.read_session(initial.session_id)
            except Exception as e:
                logger.warning("Adapter '%s' failed to refresh session '%s': %s",
                               adapter.name, initial.session_id, e)
                continue
            if current is None:
                continue
            for key, event in _event_keys(current.events):
                if key not in seen:
                    print(format_timeline_event(event), flush=True)
                    seen.add(key)
    except KeyboardInterrupt:
        print("\nStopped following session.", flush=True)
    except BrokenPipeError:
        return 0
    return 0
