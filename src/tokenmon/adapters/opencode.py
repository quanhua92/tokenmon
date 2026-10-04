"""Read-only OpenCode v1 JSON/SQLite and v2 projected session activity."""

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

from tokenmon.adapters.base import BaseAdapter
from tokenmon.adapters.codex import open_ro_db
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, create_span, generation_metadata

logger = logging.getLogger(__name__)
_COPIED_ID = re.compile(r"msg_[0-9a-f]{12}[0-9A-Za-z]{14}_(0|[1-9][0-9]*)\Z")


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _milliseconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        seconds = value / 1000
        if not math.isfinite(seconds):
            return None
        datetime.fromtimestamp(seconds).astimezone()
        return seconds
    except (OverflowError, OSError, ValueError):
        return None


def _object(value: object) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            return {}
    return value if isinstance(value, dict) else {}


def _output(value: object) -> int | None:
    value = _object(value).get("output")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _file_key(value: object) -> str | None:
    value = _string(value)
    if value is None or value in (".", "..") or any(c in value for c in ("/", "\\", "\0")):
        return None
    return value


def _json_file(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return _object(json.load(stream))
    except (OSError, ValueError, RecursionError) as exc:
        logger.debug("OpenCodeAdapter: cannot read %s: %s", path, exc)
        return {}


def _json_children(directory: Path, root: Path) -> list[Path]:
    try:
        directory.resolve().relative_to(root.resolve())
        return sorted(path for path in directory.glob("*.json")
                      if not path.name.startswith(".") and path.is_file())
    except (OSError, ValueError):
        return []


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = ?", ("table",))}


def _layout(tables: set[str]) -> str | None:
    if {"session_v2", "session_message"} <= tables:
        return "v2"
    if {"session", "message"} <= tables:
        return "v1"
    return None


def _sql_header(row: sqlite3.Row) -> dict:
    header = dict(row)
    header["time"] = {"created": header.get("time_created"), "updated": header.get("time_updated")}
    header["model"] = _object(header.get("model"))
    header["metadata"] = _object(header.get("metadata"))
    header["fork_boundary"] = _object(header.get("fork_boundary"))
    return header


@dataclass(frozen=True)
class _Source:
    session_id: str
    kind: str
    path: Path
    header: dict

    @property
    def updated_at(self) -> float:
        times = _object(self.header.get("time"))
        return _milliseconds(times.get("updated")) or _milliseconds(times.get("created")) or 0.0


@dataclass
class _Record:
    message_id: str
    kind: str
    data: dict
    parts: list[dict]
    seq: int | None = None


@dataclass(frozen=True)
class _Settings:
    model: str = "unknown"
    effort: str | None = None
    tier: str | None = None
    speed: str | None = None

    def recorded(self, data: dict) -> _Settings:
        model = _object(data.get("model"))
        provider = _string(data.get("providerID")) or _string(model.get("providerID"))
        name = _string(data.get("modelID")) or _string(model.get("id")) or _string(model.get("modelID"))
        display = f"{provider}/{name}" if provider and name and not name.startswith(provider + "/") else name
        effort, tier, speed = generation_metadata(data.get("tokens"), data, data.get("metadata"), model)
        changed = display is not None and display != self.model
        # Variant is a configuration selector, not evidence of effective effort.
        return _Settings(display or self.model,
                         effort if effort is not None else None if changed else self.effort,
                         tier if tier is not None else None if changed else self.tier,
                         speed if speed is not None else None if changed else self.speed)


class OpenCodeAdapter(BaseAdapter):
    """Read native stores without migrations, utility usage, or invented TPS."""

    def __init__(self, root: Path | str | None = None):
        self._custom_root = Path(root).expanduser().resolve() if root else None
        self._session_sources: dict[str, _Source] = {}

    @property
    def name(self) -> str:
        return "opencode"

    @property
    def root(self) -> Path:
        if self._custom_root is not None:
            return self._custom_root
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg).expanduser() if xdg and xdg.strip() else Path.home() / ".local" / "share"
        return (base / "opencode").resolve()

    def _database_paths(self) -> list[Path]:
        override = os.environ.get("OPENCODE_DB") if self._custom_root is None else None
        if override is not None:
            if override == ":memory:":
                return []
            path = Path(override).expanduser()
            path = path if path.is_absolute() else self.root / path
            return [path.resolve()] if path.is_file() else []
        paths = []
        for path in self.root.glob("opencode*.db"):
            try:
                if path.is_file() and (path.name == "opencode.db" or path.name.startswith("opencode-")):
                    paths.append((path.name != "opencode.db", -path.stat().st_mtime, path.name, path))
            except OSError:
                continue
        paths.sort(key=lambda item: item[:3])
        return [item[3] for item in paths]

    def detect(self) -> bool:
        try:
            for path in self._database_paths():
                try:
                    with closing(open_ro_db(path)) as conn:
                        if _layout(_tables(conn)) is not None:
                            return True
                except (OSError, sqlite3.Error):
                    continue
            return (self.root / "storage" / "session").is_dir()
        except Exception as exc:
            logger.warning("OpenCodeAdapter: cannot detect source: %s", exc)
            return False

    def _json_session_paths(self) -> list[Path]:
        files = []
        directory = self.root / "storage" / "session"
        for parent, _, names in os.walk(directory, followlinks=False):
            for name in names:
                if not name.startswith(".") and name.endswith(".json"):
                    path = Path(parent) / name
                    try:
                        if path.is_file():
                            files.append((-path.stat().st_mtime, str(path), path))
                    except OSError:
                        continue
        files.sort(key=lambda item: item[:2])
        return [item[2] for item in files]

    def _find_source(self, session_id: str) -> _Source | None:
        for path in self._database_paths():
            try:
                with closing(open_ro_db(path)) as conn:
                    conn.row_factory = sqlite3.Row
                    kind = _layout(_tables(conn))
                    if kind is None:
                        continue
                    table = "session_v2" if kind == "v2" else "session"
                    row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (session_id,)).fetchone()
                    if row is not None:
                        return _Source(session_id, kind, path, _sql_header(row))
            except (OSError, sqlite3.Error, ValueError) as exc:
                logger.debug("OpenCodeAdapter: cannot find session %s in %s: %s", session_id, path, exc)
        if _file_key(session_id) != session_id:
            return None
        for path in self._json_session_paths():
            header = _json_file(path)
            if _file_key(header.get("id")) == session_id:
                return _Source(session_id, "json", path, header)
        return None

    def _discover(self, max_sessions: int) -> list[_Source]:
        if max_sessions <= 0:
            return []
        candidates: dict[str, _Source] = {}
        for path in self._database_paths():
            try:
                with closing(open_ro_db(path)) as conn:
                    conn.row_factory = sqlite3.Row
                    kind = _layout(_tables(conn))
                    if kind is None:
                        continue
                    table = "session_v2" if kind == "v2" else "session"
                    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                    order = "time_updated" if "time_updated" in columns else "time_created" if "time_created" in columns else "id"
                    rows = conn.execute(f"SELECT * FROM {table} ORDER BY {order} DESC, id LIMIT ?", (max_sessions,))
                    for row in rows:
                        header = _sql_header(row)
                        session_id = _string(header.get("id"))
                        if session_id is not None and session_id not in candidates:
                            candidates[session_id] = _Source(session_id, kind, path, header)
            except (OSError, sqlite3.Error, ValueError) as exc:
                logger.debug("OpenCodeAdapter: cannot discover %s: %s", path, exc)
        for path in self._json_session_paths()[:max_sessions]:
            header = _json_file(path)
            session_id = _file_key(header.get("id"))
            if session_id is not None and session_id not in candidates:
                candidates[session_id] = _Source(session_id, "json", path, header)
        return sorted(candidates.values(), key=lambda source: source.updated_at, reverse=True)[:max_sessions]

    def _json_records(self, source: _Source) -> list[_Record]:
        records = {}
        storage = self.root / "storage"
        for path in _json_children(storage / "message" / source.session_id, storage):
            data = _json_file(path)
            message_id = _file_key(data.get("id"))
            if message_id is None or data.get("sessionID", source.session_id) != source.session_id:
                continue
            kind = _string(data.get("role"))
            if kind is None:
                continue
            parts = {}
            for part_path in _json_children(storage / "part" / message_id, storage):
                part = _json_file(part_path)
                part_id = _file_key(part.get("id"))
                if (part_id is not None and part.get("sessionID", source.session_id) == source.session_id
                        and part.get("messageID", message_id) == message_id):
                    parts[part_id] = part
            records[message_id] = _Record(message_id, kind, data, list(parts.values()))
        return sorted(records.values(), key=lambda record: (
            _milliseconds(_object(record.data.get("time")).get("created")) or 0.0, record.message_id))

    def _fork_watermark(self, conn: sqlite3.Connection, header: dict) -> tuple[bool, int | None]:
        parent = _string(header.get("fork_session_id"))
        boundary = _object(header.get("fork_boundary"))
        message_id = _string(boundary.get("messageID"))
        operator = {"before": "<", "through": "<="}.get(_string(boundary.get("type")) or "")
        if parent is None or message_id is None or operator is None:
            return False, None
        row = conn.execute("SELECT seq FROM session_message WHERE session_id = ? AND id = ?", (parent, message_id)).fetchone()
        if row is None or isinstance(row[0], bool) or not isinstance(row[0], int):
            return False, None
        # Sequence gaps make B-1 unsafe: the child reserves the selected maximum C.
        selected = conn.execute(f"SELECT MAX(seq) FROM session_message WHERE session_id = ? AND seq {operator} ?",
                                (parent, row[0])).fetchone()
        return True, selected[0] if selected is not None else None

    def _sql_records(self, conn: sqlite3.Connection, source: _Source) -> list[_Record]:
        records = []
        if source.kind == "v2":
            rows = conn.execute("SELECT * FROM session_message WHERE session_id = ? ORDER BY seq, id", (source.session_id,))
            for row in rows:
                data = _object(row["data"])
                if not data:
                    continue
                content = data.get("content", [])
                parts = [part for part in content if isinstance(part, dict)] if isinstance(content, list) else []
                records.append(_Record(row["id"], row["type"], data, parts, row["seq"]))
            return records
        part_map: dict[str, list[dict]] = {}
        if "part" in _tables(conn):
            columns = {row[1] for row in conn.execute("PRAGMA table_info(part)")}
            if "session_id" in columns:
                rows = conn.execute("SELECT * FROM part WHERE session_id = ? ORDER BY id", (source.session_id,))
            else:
                rows = conn.execute("SELECT part.* FROM part JOIN message ON message.id = part.message_id WHERE message.session_id = ? ORDER BY part.id", (source.session_id,))
            for row in rows:
                part = _object(row["data"])
                if part:
                    part["id"] = row["id"]
                    part_map.setdefault(row["message_id"], []).append(part)
        for row in conn.execute("SELECT * FROM message WHERE session_id = ? ORDER BY id", (source.session_id,)):
            data = _object(row["data"])
            kind = _string(data.get("role"))
            if data and kind is not None:
                records.append(_Record(row["id"], kind, data, part_map.get(row["id"], [])))
        records.sort(key=lambda record: (_milliseconds(_object(record.data.get("time")).get("created")) or 0.0,
                                         record.message_id))
        return records

    def _read_source(self, source: _Source) -> tuple[list[GenerationSpan], SessionTimeline] | None:
        if source.kind == "json":
            header = _json_file(source.path)
            if header.get("id") != source.session_id:
                return None
            return self._reduce(source, header, self._json_records(source), (False, None))
        with closing(open_ro_db(source.path)) as conn:
            conn.row_factory = sqlite3.Row
            table = "session_v2" if source.kind == "v2" else "session"
            row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (source.session_id,)).fetchone()
            if row is None:
                return None
            header = _sql_header(row)
            watermark = self._fork_watermark(conn, header) if source.kind == "v2" else (False, None)
            return self._reduce(source, header, self._sql_records(conn, source), watermark)

    def _reduce(self, source: _Source, header: dict, records: list[_Record],
                watermark: tuple[bool, int | None]) -> tuple[list[GenerationSpan], SessionTimeline] | None:
        events: list[TimelineEvent] = []
        spans: list[GenerationSpan] = []
        created = _milliseconds(_object(header.get("time")).get("created"))
        cwd = _string(header.get("directory"))
        settings = _Settings()
        user_settings: dict[str, _Settings] = {}
        latest_user = None
        forked = _string(header.get("fork_session_id")) is not None

        for record in records:
            try:
                data = record.data
                times = _object(data.get("time"))
                started = _milliseconds(times.get("created"))
                ended = _milliseconds(times.get("completed"))
                current = user_settings.get(_string(data.get("parentID")) or "", settings).recorded(data)
                if cwd is None:
                    cwd = _string(_object(data.get("path")).get("cwd"))
                if source.kind == "v2" and forked:
                    copied = _COPIED_ID.fullmatch(record.message_id)
                    is_copy = copied is not None and record.seq is not None and int(copied[1]) == record.seq
                    known, maximum = watermark
                    owned = not is_copy and (record.seq is not None and (maximum is None or record.seq > maximum)) if known else (
                        not is_copy and (created is None or started is not None and started >= created))
                else:
                    # v1 forks assign fresh IDs but retain the old message timestamps.
                    owned = created is None or started is not None and started >= created
                if record.kind == "user":
                    user_settings[record.message_id] = current
                    latest_user = record.message_id
                settings = current
                if not owned:
                    continue
                turn_id = _string(data.get("parentID")) or latest_user or record.message_id

                def event(kind: str, at: float | None, summary: str, suffix: str, tokens: int | None = None):
                    if at is not None:
                        events.append(TimelineEvent(
                            timestamp=at, kind=kind, turn_id=turn_id, summary=summary, tokens=tokens,
                            event_id=f"message:{record.message_id}:{suffix}", reasoning_effort=current.effort,
                            service_tier=current.tier, speed=current.speed))

                if record.kind == "user":
                    text = _string(data.get("text")) or next((_string(part.get("text")) for part in record.parts
                            if part.get("type") == "text" and _string(part.get("text")) is not None), None)
                    event("user_message", started, f"User: {text[:60]}" if text else "User message", "user")
                elif record.kind == "assistant":
                    usage = []
                    if source.kind != "v2":
                        for part in record.parts:
                            if part.get("type") == "step-finish" and (count := _output(part.get("tokens"))) is not None:
                                usage.append((_string(part.get("id")) or record.message_id, count))
                    if not usage:
                        usage.append((record.message_id, _output(data.get("tokens")) or 0))
                    response_at = ended or _milliseconds(times.get("streamed")) or started
                    note = "incomplete_response" if data.get("error") or ended is None or data.get("finish") in ("error", "aborted", "cancelled") else "unconfirmed_generation_timing"
                    for identity, tokens in usage:
                        spans.append(create_span(agent=self.name, session_id=source.session_id, turn_id=identity,
                            model=current.model, tokens=tokens, started_at=None, ended_at=response_at,
                            timing_source="opencode-unconfirmed", note=note, reasoning_effort=current.effort,
                            service_tier=current.tier, speed=current.speed))
                    event("assistant_message", response_at, f"Assistant response ({sum(count for _, count in usage)} tokens)",
                          "assistant", sum(count for _, count in usage))
                    seen_tools: set[str] = set()
                    for index, part in enumerate(record.parts):
                        kind = part.get("type")
                        identity = _string(part.get("id")) or str(index)
                        part_time = _object(part.get("time"))
                        if kind == "reasoning":
                            at = _milliseconds(part_time.get("created")) if source.kind == "v2" else _milliseconds(part_time.get("start"))
                            event("reasoning", at, "Thinking / reasoning", f"reasoning:{identity}")
                        elif kind == "tool":
                            identity = _string(part.get("callID")) or _string(part.get("id")) or str(index)
                            if identity in seen_tools:
                                continue
                            seen_tools.add(identity)
                            state = _object(part.get("state"))
                            name = _string(part.get("name")) or _string(part.get("tool")) or "tool"
                            tool_time = part_time if source.kind == "v2" else _object(state.get("time"))
                            call = _milliseconds(tool_time.get("ran")) or _milliseconds(tool_time.get("created")) if source.kind == "v2" else _milliseconds(tool_time.get("start"))
                            done = _milliseconds(tool_time.get("completed")) if source.kind == "v2" else _milliseconds(tool_time.get("end"))
                            event("tool_call", call, f"Tool call requested: {name}", f"tool:{identity}:call")
                            if state.get("status") in ("completed", "error"):
                                event("tool_output", done, f"Tool execution complete: {name}", f"tool:{identity}:output")
                elif record.kind == "model-switched":
                    event("model_change", started, f"Model selected: {current.model}", "model")
                elif record.kind == "agent-switched":
                    event("agent_change", started, f"Agent selected: {_string(data.get('agent')) or 'unknown'}", "agent")
                elif record.kind == "location-switched":
                    cwd = _string(_object(data.get("location")).get("directory")) or cwd
                    event("location_change", started, "Workspace location changed", "location")
                elif record.kind == "shell":
                    event("tool_call", started, "Tool call requested: shell", "shell:call")
                    event("tool_output", ended, "Tool execution complete: shell", "shell:output")
                elif record.kind == "idle":
                    event("turn_end", started, "Session became idle", "idle")
            except Exception as exc:
                logger.debug("OpenCodeAdapter: skipping bad message %s in %s: %s", record.message_id, source.path, exc)
                continue
        events.sort(key=lambda event: event.timestamp)
        spans.sort(key=lambda span: span.ended_at)
        first = events[0].timestamp if events else created
        last = events[-1].timestamp if events else _milliseconds(_object(header.get("time")).get("updated")) or created
        if first is None or last is None:
            return None
        if settings.model == "unknown":
            settings = settings.recorded(header)
        timeline = SessionTimeline(session_id=source.session_id, agent=self.name, model=settings.model,
            created_at=first, updated_at=last, events=events, cwd=cwd, reasoning_effort=settings.effort,
            service_tier=settings.tier, speed=settings.speed)
        return spans, timeline

    def _collect_sources(self, max_sessions: int):
        try:
            sources = self._discover(max_sessions)
        except Exception as exc:
            logger.warning("OpenCodeAdapter: cannot discover sources: %s", exc)
            return
        for source in sources:
            try:
                parsed = self._read_source(source)
            except Exception as exc:
                logger.debug("OpenCodeAdapter: cannot read session %s: %s", source.session_id, exc)
                continue
            if parsed is not None:
                self._session_sources[source.session_id] = source
                yield parsed

    def collect(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]:
        spans = [span for session_spans, _ in self._collect_sources(max_sessions) for span in session_spans
                 if min_timestamp is None or span.ended_at >= min_timestamp]
        spans.sort(key=lambda span: span.ended_at)
        return spans

    def collect_sessions(self, max_sessions: int = 32, min_timestamp: float | None = None) -> list[SessionTimeline]:
        sessions = [session for _, session in self._collect_sources(max_sessions)
                    if min_timestamp is None or session.updated_at >= min_timestamp]
        sessions.sort(key=lambda session: session.updated_at, reverse=True)
        return sessions

    def last_generation_timestamp(self, max_probe_sessions: int = 5) -> float | None:
        sessions = self.collect_sessions(max_sessions=max_probe_sessions)
        return max((session.updated_at for session in sessions), default=None)

    def read_session(self, session_id: str) -> SessionTimeline | None:
        source = self._session_sources.get(session_id)
        try:
            if source is None:
                source = self._find_source(session_id)
                if source is None:
                    return None
            parsed = self._read_source(source)
        except Exception as exc:
            logger.debug("OpenCodeAdapter: cannot refresh session %s: %s", session_id, exc)
            return None
        if parsed is None or parsed[1].session_id != session_id:
            return None
        self._session_sources[session_id] = source
        return parsed[1]
