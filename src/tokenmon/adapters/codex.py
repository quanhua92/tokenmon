"""Codex adapter: strictly read-only parser for local Codex sessions and diagnostics."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from tokenmon.adapters.base import BaseAdapter
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, create_span, generation_metadata

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


class _UsageSnapshots:
    """Repeated cumulative token_count snapshots do not describe new output."""

    def __init__(self):
        self.last_total: dict | None = None

    def is_new(self, info: object) -> bool:
        if not isinstance(info, dict) or not isinstance(info.get("last_token_usage"), dict):
            return False
        total = info.get("total_token_usage")
        if not isinstance(total, dict) or not total:
            # Older formats have no cumulative counter to establish duplication.
            return True
        if total == self.last_total:
            return False
        self.last_total = total.copy()
        return True


_ITEM_START_RE = re.compile(
    r'^Output item item_type="(reasoning|message|function_call|custom_tool_call)" '
    r'item_id="([\w-]{1,160})"$'
)


class CodexAdapter(BaseAdapter):
    """Adapter for extracting generation spans from local Codex CLI data."""

    def __init__(self, root: Path | str | None = None):
        self._custom_root = Path(root).expanduser().resolve() if root else None
        self._timeline_sources: dict[str, tuple[Path, str]] = {}

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
                columns = {row[1] for row in conn.execute("PRAGMA table_info(logs)")}
                if {"timestamp", "message"} <= columns:
                    timestamp_column, message_column = "timestamp", "message"
                elif {"ts", "feedback_log_body"} <= columns:
                    timestamp_column = "ts + ts_nanos / 1000000000.0" if "ts_nanos" in columns else "ts"
                    message_column = "feedback_log_body"
                else:
                    return starts
                rows = conn.execute(
                    f"SELECT {timestamp_column}, {message_column} FROM logs "
                    "WHERE target = ? "
                    "ORDER BY id DESC LIMIT ?",
                    ("codex_core::stream_events_utils", max_entries),
                ).fetchall()
        except Exception:
            return starts

        for ts_str, body in reversed(rows):
            if not isinstance(body, str) or not body or "handle_output_item_done" in body:
                continue
            at = float(ts_str) if isinstance(ts_str, (int, float)) else parse_timestamp(ts_str)
            if at is None or not math.isfinite(at):
                continue
            # Starts before the completion cutoff remain needed by overlapping spans.

            _, _, tail = body.rpartition(": ")
            tail = tail or body
            m = _ITEM_START_RE.fullmatch(tail)
            if m:
                item_id = m[2]
                starts[item_id] = at

        return starts

    @staticmethod
    def _session_meta(path: Path) -> dict | None:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(record, dict) and record.get("type") == "session_meta":
                        payload = record.get("payload")
                        if isinstance(payload, dict):
                            return payload
                    if isinstance(record, dict):
                        return None
        except OSError:
            pass
        return None

    @staticmethod
    def _is_subagent(source: object) -> bool:
        if isinstance(source, str):
            try:
                source = json.loads(source)
            except ValueError:
                return False
        return isinstance(source, dict) and "subagent" in source

    def _session_records(self, path: Path):
        meta = self._session_meta(path) or {}
        boundary = meta.get("subagent_history_start_ordinal")
        boundary = boundary if isinstance(boundary, int) and not isinstance(boundary, bool) and boundary >= 0 else None
        ambiguous_fork = (
            self._is_subagent(meta.get("source"))
            and (bool(meta.get("forked_from_id")) or "subagent_history_start_ordinal" in meta)
            and boundary is None
        )
        if ambiguous_fork:
            logger.warning("CodexAdapter: skipping subagent activity without a recorded ownership boundary: %s", path)
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for number, line in enumerate(stream):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                ordinal = record.get("ordinal")
                owned = not ambiguous_fork and (
                    boundary is None
                    or (isinstance(ordinal, int) and not isinstance(ordinal, bool) and ordinal >= boundary)
                )
                yield number, record, owned

    def _discover_sessions(
        self, max_sessions: int, min_timestamp: float | None = None, *, session_id: str | None = None,
    ) -> list[tuple[str, Path, str]]:
        if max_sessions <= 0:
            return []
        candidates: dict[str, tuple[float, Path, str]] = {}
        indexed_ids: set[str] = set()
        indexed_paths: set[Path] = set()
        state_db = find_newest_db(self.root, "state")

        if state_db:
            try:
                with closing(open_ro_db(state_db)) as conn:
                    cols = {r[1] for r in conn.execute("PRAGMA table_info(threads)")}
                    opt_model = "model" if "model" in cols else "NULL AS model"
                    rows = conn.execute(
                        f"SELECT id, rollout_path, {opt_model}, archived, updated_at FROM threads "
                        "ORDER BY updated_at DESC",
                    ).fetchall()
                for task_id, path_str, model, archived, updated in rows:
                    if isinstance(task_id, str):
                        indexed_ids.add(task_id)
                    if not isinstance(path_str, str) or not path_str:
                        continue
                    path = Path(path_str).expanduser().resolve()
                    indexed_paths.add(path)
                    if archived or path.suffix != ".jsonl":
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    meta = self._session_meta(path) or {}
                    native_id = meta.get("id")
                    task_id = native_id if isinstance(native_id, str) and native_id else task_id
                    if not isinstance(task_id, str) or not task_id:
                        continue
                    indexed_ids.add(task_id)
                    rank = updated if isinstance(updated, (int, float)) and math.isfinite(updated) else stat.st_mtime
                    candidates.setdefault(task_id, (rank, path, model or "unknown"))
            except (OSError, sqlite3.Error):
                pass

        # Indexed main threads do not imply that every persisted worker is indexed.
        # Keep archived indexed threads excluded rather than reviving their old files.
        for directory, _, filenames in os.walk(self.root / "sessions", followlinks=False):
            for filename in filenames:
                if filename.startswith(".") or not filename.endswith(".jsonl"):
                    continue
                path = (Path(directory) / filename).resolve()
                if path in indexed_paths:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if not stat.st_size:
                    continue
                meta = self._session_meta(path) or {}
                native_id = meta.get("id")
                task_id = native_id if isinstance(native_id, str) and native_id else path.stem
                if task_id in indexed_ids:
                    continue
                candidate = (stat.st_mtime, path, "unknown")
                previous = candidates.get(task_id)
                if previous is None or (candidate[0], str(path)) > (previous[0], str(previous[1])):
                    candidates[task_id] = candidate

        ranked = sorted(candidates.items(), key=lambda item: (-item[1][0], str(item[1][1])))
        found = [
            (task_id, path, model) for task_id, (_, path, model) in ranked
            if session_id is None or task_id == session_id
        ][:max_sessions]
        for task_id, path, model in found:
            self._timeline_sources.setdefault(task_id, (path, model))
        # Completion/event timestamps, not file/index freshness, enforce cutoffs.
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
        usage_snapshots = _UsageSnapshots()
        reasoning_effort = service_tier = speed = None
        output_settings = (None, None, None)

        try:
            with closing(self._session_records(jsonl_path)) as records:
                for _, record, owned in records:

                    val = record.get("payload")
                    at = parse_timestamp(record.get("timestamp"))
                    if not isinstance(val, dict) or at is None:
                        continue

                    kind = record.get("type")
                    event = val.get("type")
                    if not isinstance(kind, str) or (event is not None and not isinstance(event, str)):
                        continue
                    # Inherited settings remain context, but neither inherited items
                    # nor cumulative counters belong to this worker's output.
                    if not owned and not (
                        kind == "turn_context"
                        or (kind == "event_msg" and event == "thread_settings_applied")
                    ):
                        continue

                    if kind == "turn_context":
                        t = val.get("turn_id")
                        if isinstance(t, str):
                            turn_id = t
                        m = val.get("model")
                        if isinstance(m, str) and m:
                            model = m
                        effort, tier, mode = generation_metadata(val)
                        if "effort" in val or "reasoning_effort" in val or effort is not None:
                            reasoning_effort = effort
                        if "service_tier" in val:
                            service_tier = tier
                        if "speed" in val:
                            speed = mode

                    elif kind == "event_msg":
                        if event == "thread_settings_applied":
                            settings = val.get("thread_settings")
                            if isinstance(settings, dict):
                                effort, tier, mode = generation_metadata(settings)
                                if "reasoning_effort" in settings or "effort" in settings or effort is not None:
                                    reasoning_effort = effort
                                if "service_tier" in settings:
                                    service_tier = tier
                                if "speed" in settings:
                                    speed = mode
                        elif event == "item_completed":
                            item = val.get("item")
                            if isinstance(item, dict) and isinstance(item.get("type"), str) and item.get("type") in {
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
                            if isinstance(info, dict) and usage_snapshots.is_new(info):
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
                                        *output_settings,
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
                                output_settings = (reasoning_effort, service_tier, speed)
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
                            *output_settings,
                        )
                        items.clear()
                        first_output_kind = None
        except Exception:
            pass

        # Parse earlier context and usage snapshots before filtering completions.
        return [s for s in spans if min_timestamp is None or s.ended_at >= min_timestamp]

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
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
        speed: str | None = None,
    ) -> None:
        if not items:
            return

        resp_id = str(response_id) if response_id else f"resp_{len(spans)}_{at}"
        if resp_id in seen_responses:
            return
        seen_responses.add(resp_id)

        tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        _, usage_tier, usage_speed = generation_metadata(usage)
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
                reasoning_effort=reasoning_effort,
                service_tier=usage_tier or service_tier,
                speed=usage_speed or speed,
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
                self._timeline_sources.setdefault(session_id, (jsonl_path, default_model))
                if timeline is not None and timeline.events and (min_timestamp is None or timeline.updated_at >= min_timestamp):
                    timelines.append(timeline)
            except Exception as e:
                logger.error("CodexAdapter: skipping unparseable session timeline '%s': %s", jsonl_path, e)

        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        return timelines

    def read_session(self, session_id: str) -> SessionTimeline | None:
        source = self._timeline_sources.get(session_id)
        if source is None:
            sources = self._discover_sessions(1, session_id=session_id)
            if not sources:
                return None
            _, path, model = sources[0]
            source = (path, model)
        path, model = source
        if not path.is_file():
            return None
        meta = self._session_meta(path)
        if meta is not None and meta.get("id") is not None and meta.get("id") != session_id:
            return None
        return self._parse_session_timeline(session_id, path, model)

    def _parse_session_timeline(
        self,
        session_id: str,
        jsonl_path: Path,
        default_model: str,
    ) -> SessionTimeline | None:
        events: list[TimelineEvent] = []
        model = default_model
        turn_id = ""
        cwd: str | None = None
        usage_snapshots = _UsageSnapshots()
        reasoning_effort = service_tier = speed = None

        try:
            with closing(self._session_records(jsonl_path)) as records:
                for record_number, record, owned in records:
                    first_new_event = len(events)

                    val = record.get("payload")
                    at = parse_timestamp(record.get("timestamp"))
                    if not isinstance(val, dict) or at is None:
                        continue

                    kind = record.get("type")
                    event = val.get("type")
                    if not isinstance(kind, str) or (event is not None and not isinstance(event, str)):
                        continue

                    if cwd is None and kind in {"session_meta", "turn_context"}:
                        c = val.get("cwd")
                        if isinstance(c, str) and c:
                            cwd = c
                    if not owned and not (
                        kind in {"session_meta", "turn_context"}
                        or (kind == "event_msg" and event == "thread_settings_applied")
                    ):
                        continue

                    if kind == "turn_context":
                        t = val.get("turn_id")
                        if isinstance(t, str):
                            turn_id = t
                        m = val.get("model")
                        if isinstance(m, str) and m:
                            model = m
                        effort, tier, mode = generation_metadata(val)
                        if "effort" in val or "reasoning_effort" in val or effort is not None:
                            reasoning_effort = effort
                        if "service_tier" in val:
                            service_tier = tier
                        if "speed" in val:
                            speed = mode

                    elif kind == "event_msg":
                        if event == "thread_settings_applied":
                            settings = val.get("thread_settings")
                            if isinstance(settings, dict):
                                effort, tier, mode = generation_metadata(settings)
                                if "reasoning_effort" in settings or "effort" in settings or effort is not None:
                                    reasoning_effort = effort
                                if "service_tier" in settings:
                                    service_tier = tier
                                if "speed" in settings:
                                    speed = mode
                        elif event == "task_started":
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

                    # Usage events must also be processed after the event_msg branch.
                    if kind == "token_usage_record" or (kind == "event_msg" and event == "token_count"):
                        info = val.get("usage")
                        if kind == "event_msg":
                            count_info = val.get("info")
                            if not usage_snapshots.is_new(count_info):
                                continue
                            info = count_info.get("last_token_usage") if isinstance(count_info, dict) else None
                        tokens = info.get("output_tokens") if isinstance(info, dict) else None
                        if isinstance(tokens, int) and tokens >= 0 and events:
                            for idx in range(len(events) - 1, -1, -1):
                                ev = events[idx]
                                if ev.turn_id != turn_id:
                                    break
                                if ev.kind in {"assistant_message", "reasoning"}:
                                    # The two usage formats may describe the same output.
                                    # Update its latest event instead of backfilling earlier reasoning.
                                    summary = ev.summary.removesuffix(f" ({ev.tokens} tokens)")
                                    _, usage_tier, usage_speed = generation_metadata(info)
                                    events[idx] = replace(
                                        ev,
                                        summary=f"{summary} ({tokens} tokens)",
                                        tokens=tokens,
                                        service_tier=usage_tier or ev.service_tier,
                                        speed=usage_speed or ev.speed,
                                    )
                                    break
                    for idx in range(first_new_event, len(events)):
                        events[idx] = replace(events[idx], event_id=f"record:{record_number}:{events[idx].kind}",
                                              reasoning_effort=reasoning_effort, service_tier=service_tier, speed=speed)
        except OSError:
            return None
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
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
            speed=speed,
        )
