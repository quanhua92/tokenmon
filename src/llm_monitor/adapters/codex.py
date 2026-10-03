"""Codex adapter: strictly read-only parser for local Codex sessions and diagnostics."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm_monitor.adapters.base import BaseAdapter
from llm_monitor.models import GenerationSpan, SessionTimeline, TimelineEvent, create_span

logger = logging.getLogger(__name__)


def parse_timestamp(val: object) -> float | None:
    if not val:
        return None
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return None
        ts = dt.timestamp()
        return ts if math.isfinite(ts) else None
    except Exception:
        return None


def open_ro_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)
    conn.execute("PRAGMA query_only = ON")
    return conn


def find_newest_db(root: Path, prefix: str) -> Path | None:
    candidates = []
    for p in root.glob(f"{prefix}_*.sqlite"):
        m = re.fullmatch(rf"{prefix}_(\d+)\.sqlite", p.name)
        if m:
            candidates.append((int(m[1]), p))
    return max(candidates)[1] if candidates else None


@dataclass
class _ItemTiming:
    turn_id: str
    start: float
    end: float


_ITEM_START_RE = re.compile(
    r'^Output item item_type="(reasoning|message|function_call|custom_tool_call)" '
    r'item_id="([\w-]{1,160})"$'
)


class CodexAdapter(BaseAdapter):
    """Adapter for extracting generation spans from local Codex CLI data."""

    def __init__(self, root: Path | str | None = None):
        self._custom_root = Path(root).expanduser().resolve() if root else None

    @property
    def name(self) -> str:
        return "codex"

    @property
    def root(self) -> Path:
        if self._custom_root:
            return self._custom_root
        env = os.environ.get("CODEX_HOME")
        if env:
            return Path(env).expanduser().resolve()
        return (Path.home() / ".codex").resolve()

    def detect(self) -> bool:
        r = self.root
        if not r.exists():
            return False
        return (
            (r / "sessions").exists()
            or find_newest_db(r, "state") is not None
            or find_newest_db(r, "logs") is not None
        )

    def collect(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]:
        if not self.root.exists():
            return []

        item_starts = self._load_diagnostic_starts(min_timestamp=min_timestamp)
        sessions = self._discover_sessions(max_sessions, min_timestamp=min_timestamp)

        all_spans: list[GenerationSpan] = []
        for session_id, jsonl_path, default_model in sessions:
            try:
                spans = self._parse_session_file(session_id, jsonl_path, default_model, item_starts, min_timestamp=min_timestamp)
                all_spans.extend(spans)
            except Exception as e:
                logger.error("CodexAdapter: skipping unparseable session file '%s': %s", jsonl_path, e)

        all_spans.sort(key=lambda s: s.ended_at)
        return all_spans

    def _load_diagnostic_starts(self, max_entries: int = 100000, min_timestamp: float | None = None) -> dict[str, float]:
        starts: dict[str, float] = {}
        logs_db = find_newest_db(self.root, "logs")
        if not logs_db:
            return starts

        try:
            with closing(open_ro_db(logs_db)) as conn:
                rows = conn.execute(
                    "SELECT timestamp, message FROM logs "
                    "WHERE target = 'codex_core::stream_events_utils' "
                    "ORDER BY id DESC LIMIT ?",
                    (max_entries,),
                ).fetchall()
        except Exception:
            return starts

        for ts_str, body in reversed(rows):
            if not body or "handle_output_item_done" in body:
                continue
            at = parse_timestamp(ts_str)
            if at is None:
                continue
            if min_timestamp is not None and at < min_timestamp:
                continue

            _, _, tail = body.rpartition(": ")
            tail = tail or body
            m = _ITEM_START_RE.fullmatch(tail)
            if m:
                item_id = m[2]
                starts[item_id] = at

        return starts

    def _discover_sessions(self, max_sessions: int, min_timestamp: float | None = None) -> list[tuple[str, Path, str]]:
        found: list[tuple[str, Path, str]] = []
        state_db = find_newest_db(self.root, "state")

        if state_db:
            try:
                with closing(open_ro_db(state_db)) as conn:
                    cols = {r[1] for r in conn.execute("PRAGMA table_info(threads)")}
                    opt_model = "model" if "model" in cols else "NULL AS model"
                    where_clause = "WHERE archived = 0"
                    params: list[object] = []
                    if min_timestamp is not None:
                        where_clause += " AND updated_at >= ?"
                        params.append(int(min_timestamp))
                    params.append(max_sessions)

                    rows = conn.execute(
                        f"SELECT id, rollout_path, {opt_model} FROM threads "
                        f"{where_clause} ORDER BY updated_at DESC LIMIT ?",
                        tuple(params),
                    ).fetchall()
                    for task_id, path_str, model in rows:
                        if path_str:
                            p = Path(path_str).expanduser()
                            if p.exists() and p.suffix == ".jsonl":
                                found.append((task_id, p, model or "unknown"))
            except Exception:
                pass

        if not found:
            sessions_dir = self.root / "sessions"
            if sessions_dir.exists():
                all_files = list(sessions_dir.glob("**/*.jsonl"))
                if min_timestamp is not None:
                    all_files = [p for p in all_files if p.stat().st_mtime >= min_timestamp]
                candidates = sorted(
                    all_files,
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for p in candidates[:max_sessions]:
                    found.append((p.stem, p, "unknown"))

        return found

    def _parse_session_file(
        self,
        session_id: str,
        jsonl_path: Path,
        default_model: str,
        diag_starts: dict[str, float],
        min_timestamp: float | None = None,
    ) -> list[GenerationSpan]:
        spans: list[GenerationSpan] = []
        model = default_model
        turn_id = ""
        items: dict[str, float] = {}
        first_output_kind: str | None = None
        timings: dict[str, _ItemTiming] = {}
        seen_responses: set[str] = set()

        try:
            with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    val = record.get("payload")
                    at = parse_timestamp(record.get("timestamp"))
                    if not isinstance(val, dict) or at is None:
                        continue
                    if min_timestamp is not None and at < min_timestamp:
                        continue

                    kind = record.get("type")
                    event = val.get("type")

                    if kind == "turn_context":
                        t = val.get("turn_id")
                        if isinstance(t, str):
                            turn_id = t
                        m = val.get("model")
                        if isinstance(m, str) and m:
                            model = m

                    elif kind == "event_msg":
                        if event == "item_completed":
                            item = val.get("item")
                            if isinstance(item, dict) and item.get("type") in {
                                "Reasoning",
                                "AgentMessage",
                            }:
                                item_id = item.get("id")
                                start_ms = val.get("started_at_ms")
                                end_ms = val.get("completed_at_ms")
                                if (
                                    isinstance(item_id, str)
                                    and isinstance(start_ms, int)
                                    and isinstance(end_ms, int)
                                    and 0 < start_ms <= end_ms
                                ):
                                    timings[item_id] = _ItemTiming(
                                        val.get("turn_id", turn_id),
                                        start_ms / 1000.0,
                                        end_ms / 1000.0,
                                    )
                        elif event == "task_started":
                            turn_id = str(val.get("turn_id", ""))
                            items.clear()
                            timings.clear()
                        elif event == "token_count":
                            info = val.get("info")
                            if isinstance(info, dict):
                                usage = info.get("last_token_usage")
                                if isinstance(usage, dict) and items:
                                    self._append_span(
                                        spans,
                                        session_id,
                                        turn_id,
                                        model,
                                        usage,
                                        at,
                                        None,
                                        items,
                                        first_output_kind,
                                        timings,
                                        diag_starts,
                                        seen_responses,
                                    )
                                    items.clear()
                                    first_output_kind = None

                    elif kind == "response_item":
                        is_out = event in {"reasoning", "function_call", "custom_tool_call"} or (
                            event == "message" and val.get("role") == "assistant"
                        )
                        if is_out:
                            if not items:
                                first_output_kind = event
                            item_id = val.get("id")
                            if isinstance(item_id, str) and item_id:
                                items.setdefault(item_id, at)

                    elif kind == "token_usage_record":
                        self._append_span(
                            spans,
                            session_id,
                            turn_id,
                            model,
                            val.get("usage"),
                            at,
                            val.get("response_id"),
                            items,
                            first_output_kind,
                            timings,
                            diag_starts,
                            seen_responses,
                        )
                        items.clear()
                        first_output_kind = None
        except Exception:
            pass

        return spans

    def _append_span(
        self,
        spans: list[GenerationSpan],
        session_id: str,
        turn_id: str,
        model: str,
        usage: object,
        at: float,
        response_id: object,
        items: dict[str, float],
        first_output_kind: str | None,
        timings: dict[str, _ItemTiming],
        diag_starts: dict[str, float],
        seen_responses: set[str],
    ) -> None:
        if not items:
            return

        resp_id = str(response_id) if response_id else f"resp_{len(spans)}_{at}"
        if resp_id in seen_responses:
            return
        seen_responses.add(resp_id)

        tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        item_list = list(items.items())
        first_id, _ = item_list[0]

        note = None
        timing_source = "unknown"
        start_at: float | None = None

        if first_output_kind in {"function_call", "custom_tool_call"}:
            note = "unconfirmed_tool_start"

        first_diag = diag_starts.get(first_id)
        first_timing = timings.get(first_id)

        # Detect sub-millisecond placeholder collision
        if (
            first_timing is not None
            and first_timing.start == first_timing.end
            and first_diag is not None
            and abs(first_diag - first_timing.end) <= 0.001
        ):
            note = note or "unconfirmed_initial_span"

        if first_diag is not None:
            start_at = first_diag
            timing_source = "stream-log"
        elif first_timing is not None:
            if first_timing.start == first_timing.end:
                note = note or "missing_stream_start"
            start_at = first_timing.start
            timing_source = "item-event"
        else:
            note = note or "missing_start"

        end_at = max(
            timings[i_id].end if i_id in timings else i_at for i_id, i_at in item_list
        )

        spans.append(
            create_span(
                agent=self.name,
                session_id=session_id,
                turn_id=turn_id,
                model=model,
                tokens=tokens,
                started_at=start_at,
                ended_at=end_at,
                timing_source=timing_source,
                note=note,
            )
        )

    def collect_sessions(self, max_sessions: int = 32, min_timestamp: float | None = None) -> list[SessionTimeline]:
        if not self.root.exists():
            return []

        sessions = self._discover_sessions(max_sessions, min_timestamp=min_timestamp)
        timelines: list[SessionTimeline] = []
        for session_id, jsonl_path, default_model in sessions:
            try:
                timeline = self._parse_session_timeline(session_id, jsonl_path, default_model)
                if timeline.events:
                    timelines.append(timeline)
            except Exception as e:
                logger.error("CodexAdapter: skipping unparseable session timeline '%s': %s", jsonl_path, e)

        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        return timelines

    def _parse_session_timeline(
        self,
        session_id: str,
        jsonl_path: Path,
        default_model: str,
    ) -> SessionTimeline:
        events: list[TimelineEvent] = []
        model = default_model
        turn_id = ""
        cwd: str | None = None

        try:
            with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    val = record.get("payload")
                    at = parse_timestamp(record.get("timestamp"))
                    if not isinstance(val, dict) or at is None:
                        continue

                    kind = record.get("type")
                    event = val.get("type")

                    if cwd is None and kind in {"session_meta", "turn_context"}:
                        c = val.get("cwd")
                        if isinstance(c, str) and c:
                            cwd = c

                    if kind == "turn_context":
                        t = val.get("turn_id")
                        if isinstance(t, str):
                            turn_id = t
                        m = val.get("model")
                        if isinstance(m, str) and m:
                            model = m

                    elif kind == "event_msg":
                        if event == "task_started":
                            turn_id = str(val.get("turn_id", ""))
                            events.append(
                                TimelineEvent(
                                    timestamp=at,
                                    kind="user_message",
                                    turn_id=turn_id,
                                    summary="User started turn / prompt",
                                )
                            )
                        elif event == "item_completed":
                            item = val.get("item")
                            if isinstance(item, dict):
                                itype = item.get("type")
                                if itype == "ToolCall":
                                    tname = item.get("name") or "tool"
                                    events.append(
                                        TimelineEvent(
                                            timestamp=at,
                                            kind="tool_output",
                                            turn_id=turn_id,
                                            summary=f"Tool execution complete: {tname}",
                                        )
                                    )
                        elif event in {"task_complete", "turn_aborted", "task_failed"}:
                            dur_ms = val.get("duration_ms")
                            dur_s = dur_ms / 1000.0 if isinstance(dur_ms, (int, float)) else None
                            events.append(
                                TimelineEvent(
                                    timestamp=at,
                                    kind="turn_end",
                                    turn_id=turn_id,
                                    summary=f"Turn ended ({event})",
                                    duration=dur_s,
                                )
                            )

                    elif kind == "response_item":
                        if event == "message":
                            role = val.get("role")
                            if role == "user":
                                if not (events and events[-1].kind == "user_message" and abs(events[-1].timestamp - at) < 1.0):
                                    events.append(
                                        TimelineEvent(
                                            timestamp=at,
                                            kind="user_message",
                                            turn_id=turn_id,
                                            summary="User message",
                                        )
                                    )
                            elif role == "assistant":
                                events.append(
                                    TimelineEvent(
                                        timestamp=at,
                                        kind="assistant_message",
                                        turn_id=turn_id,
                                        summary="Assistant message",
                                    )
                                )
                        elif event == "reasoning":
                            events.append(
                                TimelineEvent(
                                    timestamp=at,
                                    kind="reasoning",
                                    turn_id=turn_id,
                                    summary="Thinking / reasoning",
                                )
                            )
                        elif event in {"function_call", "custom_tool_call"}:
                            tname = val.get("name") or "tool"
                            events.append(
                                TimelineEvent(
                                    timestamp=at,
                                    kind="tool_call",
                                    turn_id=turn_id,
                                    summary=f"Tool call requested: {tname}",
                                )
                            )

                    elif kind == "token_usage_record" or (kind == "event_msg" and event == "token_count"):
                        info = val.get("usage") if kind == "token_usage_record" else val.get("info", {}).get("last_token_usage")
                        tokens = info.get("output_tokens") if isinstance(info, dict) else None
                        if tokens and events:
                            for idx in range(len(events) - 1, -1, -1):
                                ev = events[idx]
                                if ev.kind in {"assistant_message", "reasoning"} and ev.tokens is None:
                                    events[idx] = TimelineEvent(
                                        timestamp=ev.timestamp,
                                        kind=ev.kind,
                                        turn_id=ev.turn_id,
                                        summary=f"{ev.summary} ({tokens} tokens)",
                                        tokens=tokens,
                                        duration=ev.duration,
                                    )
                                    break
        except Exception:
            pass

        events.sort(key=lambda e: e.timestamp)
        created_at = events[0].timestamp if events else 0.0
        updated_at = events[-1].timestamp if events else 0.0

        return SessionTimeline(
            session_id=session_id,
            agent=self.name,
            model=model,
            created_at=created_at,
            updated_at=updated_at,
            events=events,
            cwd=cwd,
        )
