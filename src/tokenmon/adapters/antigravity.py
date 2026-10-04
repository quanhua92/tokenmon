"""Antigravity (agy) adapter: strictly read-only parser for local Antigravity sessions and steps."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import stat
from bisect import bisect_right
from contextlib import closing
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from tokenmon.adapters.base import BaseAdapter
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, create_span, generation_metadata

logger = logging.getLogger(__name__)


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    res = 0
    shift = 0
    limit = len(data)
    while offset < limit:
        b = data[offset]
        offset += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, offset
        shift += 7
        if shift > 64:
            break
    return res, offset


def parse_proto_fields(data: bytes) -> list[tuple[int, str, int | bytes | None]]:
    if not isinstance(data, bytes):
        return []
    offset = 0
    fields: list[tuple[int, str, int | bytes | None]] = []
    limit = len(data)
    try:
        while offset < limit:
            tag_byte, offset = decode_varint(data, offset)
            field_num = tag_byte >> 3
            wire_type = tag_byte & 7
            if wire_type == 0:  # varint
                val, offset = decode_varint(data, offset)
                fields.append((field_num, "varint", val))
            elif wire_type == 2:  # length-delimited
                length, offset = decode_varint(data, offset)
                if offset + length > limit:
                    break
                val = data[offset : offset + length]
                offset += length
                fields.append((field_num, "bytes", val))
            elif wire_type == 1:  # 64-bit
                if offset + 8 > limit:
                    break
                val = data[offset : offset + 8]
                offset += 8
                fields.append((field_num, "fixed64", val))
            elif wire_type == 5:  # 32-bit
                if offset + 4 > limit:
                    break
                val = data[offset : offset + 4]
                offset += 4
                fields.append((field_num, "fixed32", val))
            else:
                break
    except Exception as e:
        logger.debug("Antigravity: error parsing proto fields: %s", e)
    return fields


def parse_proto_timestamp(val_bytes: bytes) -> float | None:
    try:
        sub = parse_proto_fields(val_bytes)
        sec = 0
        nanos = 0
        for fnum, wtype, val in sub:
            if fnum == 1 and isinstance(val, int):
                sec = val
            elif fnum == 2 and isinstance(val, int):
                nanos = val
        if sec > 0:
            ts = sec + nanos * 1e-9
            return ts if math.isfinite(ts) else None
    except Exception:
        pass
    return None


def open_ro_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)
    conn.execute("PRAGMA query_only = ON")
    return conn


class AntigravityAdapter(BaseAdapter):
    """Adapter for extracting generation spans and session timelines from Antigravity (agy)."""

    def __init__(self, root: Path | str | None = None):
        self._custom_root = Path(root).expanduser().resolve() if root else None

    @property
    def name(self) -> str:
        return "antigravity"

    @property
    def root(self) -> Path:
        if self._custom_root:
            return self._custom_root
        env = os.environ.get("ANTIGRAVITY_HOME")
        if env:
            return Path(env).expanduser().resolve()
        return (Path.home() / ".gemini" / "antigravity-cli").resolve()

    def detect(self) -> bool:
        r = self.root
        if not r.exists():
            return False
        conv_dir = r / "conversations"
        summary_db = r / "conversation_summaries.db"
        return conv_dir.is_dir() or summary_db.is_file()

    def _discover_session_dbs(self, max_sessions: int) -> list[Path]:
        if max_sessions <= 0:
            return []
        conv_dir = self.root / "conversations"
        sources: dict[str, tuple[Path, float]] = {}
        try:
            for path in conv_dir.glob("*.db"):
                try:
                    info = path.stat()
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    sources[path.stem] = (path, info.st_mtime)
        except OSError:
            return []

        # The summary index orders known conversations, but is not an exhaustive
        # source inventory. Independently persisted workers may have no row.
        # Summary times and filesystem mtimes order sources only; recorded step
        # times determine the cutoff after the bounded sources have been parsed.
        indexed: list[tuple[Path, float]] = []
        seen: set[str] = set()
        summary_db = self.root / "conversation_summaries.db"
        try:
            with closing(open_ro_db(summary_db)) as conn:
                rows = conn.execute(
                    "SELECT conversation_id, last_modified_time FROM conversation_summaries "
                    "ORDER BY last_modified_time DESC"
                )
                for session_id, modified in rows:
                    if session_id in sources and session_id not in seen:
                        path, timestamp = sources[session_id]
                        try:
                            recorded = datetime.fromisoformat(str(modified).replace("Z", "+00:00"))
                            if recorded.tzinfo is not None and math.isfinite(recorded.timestamp()):
                                timestamp = recorded.timestamp()
                        except (ValueError, TypeError, OverflowError, OSError):
                            pass
                        indexed.append((path, timestamp))
                        seen.add(session_id)
        except (sqlite3.Error, OSError):
            pass

        unindexed: list[tuple[Path, float]] = []
        for session_id, (path, timestamp) in sources.items():
            if session_id in seen:
                continue
            # Native conversations also persist SQLite WALs: a live worker can
            # append output without checkpointing or touching the main DB file.
            try:
                wal = path.with_name(path.name + "-wal").stat()
                if stat.S_ISREG(wal.st_mode) and wal.st_size:
                    timestamp = max(timestamp, wal.st_mtime)
            except OSError:
                pass
            unindexed.append((path, timestamp))
        unindexed.sort(key=lambda source: (-source[1], source[0].name))

        # Merge recency streams instead of filling the limit from the summary
        # alone. Keep the native index's relative order even for unusual dates.
        candidates: list[Path] = []
        indexed_pos = unindexed_pos = 0
        while len(candidates) < max_sessions and (indexed_pos < len(indexed) or unindexed_pos < len(unindexed)):
            if (unindexed_pos < len(unindexed) and
                    (indexed_pos >= len(indexed) or unindexed[unindexed_pos][1] > indexed[indexed_pos][1])):
                candidates.append(unindexed[unindexed_pos][0])
                unindexed_pos += 1
            else:
                candidates.append(indexed[indexed_pos][0])
                indexed_pos += 1
        return candidates

    def last_generation_timestamp(self) -> float | None:
        dbs = self._discover_session_dbs(max_sessions=5)
        latest_ts: float | None = None

        for db_path in dbs:
            try:
                with closing(open_ro_db(db_path)) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT metadata FROM steps WHERE step_type = 15 AND metadata IS NOT NULL "
                        "ORDER BY idx DESC LIMIT 1;"
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        fields = {f[0]: f[2] for f in parse_proto_fields(row[0])}
                        f7 = fields.get(7) or fields.get(8) or fields.get(1)
                        if isinstance(f7, bytes):
                            ts = parse_proto_timestamp(f7)
                            if ts is not None:
                                if latest_ts is None or ts > latest_ts:
                                    latest_ts = ts
            except Exception:
                continue

        return latest_ts

    def _generation_settings(self, conn):
        """Read the same recorded model/configuration boundaries for spans and timelines."""
        cur = conn.cursor()
        default_model = "gemini"
        step_model_map: dict[int, str] = {}
        step_settings_map: dict[int, tuple[str | None, str | None, str | None]] = {}
        default_settings = (None, None, None)
        try:
            cur.execute("SELECT data FROM gen_metadata WHERE data IS NOT NULL;")
            for (data,) in cur.fetchall():
                for fn, wt, val in parse_proto_fields(data):
                    if fn == 1 and isinstance(val, bytes):
                        m_name: str | None = None
                        last_idx: int | None = None
                        metadata_settings = {}
                        for sfn, swt, sval in parse_proto_fields(val):
                            if sfn == 19 and isinstance(sval, bytes):
                                m_name = sval.decode("utf-8", errors="ignore").strip()
                            elif sfn == 20 and isinstance(sval, bytes):
                                try:
                                    k_v = parse_proto_fields(sval)
                                    d = {k[0]: k[2] for k in k_v}
                                    k_b = d.get(1)
                                    v_b = d.get(2)
                                    if isinstance(k_b, bytes) and isinstance(v_b, bytes):
                                        key = k_b.decode("utf-8", errors="ignore")
                                        if key in {"reasoning_effort", "effort", "thinking_level", "service_tier", "speed"}:
                                            metadata_settings[key] = v_b.decode("utf-8", errors="ignore")
                                    if k_b == b"last_step_index" and isinstance(v_b, bytes):
                                        last_idx = int(v_b.decode("utf-8", errors="ignore"))
                                except Exception:
                                    pass
                        if m_name:
                            default_model = m_name
                            default_settings = generation_metadata(metadata_settings)
                            if last_idx is not None:
                                step_model_map[last_idx + 1] = m_name
                                step_settings_map[last_idx + 1] = default_settings
        except Exception:
            pass
        return default_model, default_settings, step_model_map, step_settings_map

    def collect(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]:
        dbs = self._discover_session_dbs(max_sessions)
        spans: list[GenerationSpan] = []

        for db_path in dbs:
            session_id = db_path.stem
            try:
                with closing(open_ro_db(db_path)) as conn:
                    cur = conn.cursor()

                    # 1. Parse model from gen_metadata
                    default_model, default_settings, step_model_map, step_settings_map = self._generation_settings(conn)

                    model_boundaries = sorted(step_model_map)

                    # 2. Extract step_type 15 (model outputs)
                    cur.execute(
                        "SELECT idx, metadata FROM steps WHERE step_type = 15 "
                        "AND metadata IS NOT NULL ORDER BY idx;"
                    )
                    for idx, meta in cur.fetchall():
                        fields = {f[0]: f[2] for f in parse_proto_fields(meta)}
                        f1 = fields.get(1)
                        f7 = fields.get(7)
                        f8 = fields.get(8)
                        f9 = fields.get(9)

                        if not isinstance(f1, bytes) or not isinstance(f9, bytes):
                            continue

                        t_start = parse_proto_timestamp(f1)
                        t_end = parse_proto_timestamp(f7) if isinstance(f7, bytes) else None
                        if t_end is None and isinstance(f8, bytes):
                            t_end = parse_proto_timestamp(f8)
                        if min_timestamp is not None and t_end is not None and t_end < min_timestamp:
                            continue

                        u_dict = {
                            f[0]: f[2]
                            for f in parse_proto_fields(f9)
                            if f[1] == "varint" and isinstance(f[2], int)
                        }
                        out_tokens = u_dict.get(3, 0)
                        boundary = bisect_right(model_boundaries, idx) - 1
                        model = step_model_map[model_boundaries[boundary]] if boundary >= 0 else default_model
                        effort, tier, speed = step_settings_map[model_boundaries[boundary]] if boundary >= 0 else default_settings
                        span = create_span(
                            agent=self.name,
                            session_id=session_id,
                            turn_id=str(idx),
                            model=model,
                            started_at=t_start,
                            ended_at=t_end,
                            tokens=out_tokens,
                            timing_source="agy-step-proto",
                            reasoning_effort=effort,
                            service_tier=tier,
                            speed=speed,
                        )
                        if span is not None:
                            spans.append(span)
            except Exception as e:
                logger.error("AntigravityAdapter: error parsing session db '%s': %s", db_path, e)
                continue

        spans.sort(key=lambda s: s.ended_at)
        return spans

    def collect_sessions(self, max_sessions: int = 32, min_timestamp: float | None = None) -> list[SessionTimeline]:
        dbs = self._discover_session_dbs(max_sessions)
        timelines: list[SessionTimeline] = []

        for p in dbs:
            try:
                timeline = self.parse_session_timeline(p.stem)
                if timeline.events and (min_timestamp is None or timeline.updated_at >= min_timestamp):
                    timelines.append(timeline)
            except Exception as e:
                logger.error("AntigravityAdapter: error parsing session timeline '%s': %s", p, e)
                continue

        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        return timelines

    def read_session(self, session_id: str) -> SessionTimeline | None:
        # Follow uses the exact ID, even when it leaves the recent-session limit.
        if not session_id or Path(session_id).name != session_id:
            return None
        try:
            if not (self.root / "conversations" / f"{session_id}.db").is_file():
                return None
            timeline = self.parse_session_timeline(session_id)
        except OSError:
            return None
        return timeline if timeline.events else None

    def parse_session_timeline(self, session_id: str) -> SessionTimeline:
        # Never substitute a prefix-matched conversation after a source vanishes.
        db_path = self.root / "conversations" / f"{session_id}.db"

        if not db_path.exists():
            return SessionTimeline(
                session_id=session_id,
                agent=self.name,
                model="gemini",
                created_at=0.0,
                updated_at=0.0,
                events=[],
            )

        events: list[TimelineEvent] = []
        model = "gemini"
        latest_settings = (None, None, None)
        last_activity: float | None = None

        try:
            with closing(open_ro_db(db_path)) as conn:
                cur = conn.cursor()

                model, default_settings, step_model_map, step_settings_map = self._generation_settings(conn)
                latest_settings = default_settings
                model_boundaries = sorted(step_model_map)

                # Read all steps
                columns = {row[1] for row in conn.execute("PRAGMA table_info(steps)")}
                payload_column = "step_payload" if "step_payload" in columns else "NULL"
                cur.execute(
                    f"SELECT idx, step_type, metadata, {payload_column} FROM steps "
                    "WHERE metadata IS NOT NULL ORDER BY idx;"
                )
                for idx, step_type, meta, payload in cur.fetchall():
                    first_new_event = len(events)
                    boundary = bisect_right(model_boundaries, idx) - 1
                    settings = step_settings_map[model_boundaries[boundary]] if boundary >= 0 else default_settings
                    fields = {f[0]: f[2] for f in parse_proto_fields(meta)}
                    f1 = fields.get(1)
                    f7 = fields.get(7)
                    f8 = fields.get(8)

                    t_start = parse_proto_timestamp(f1) if isinstance(f1, bytes) else None
                    t_end = parse_proto_timestamp(f7) if isinstance(f7, bytes) else None
                    if t_end is None and isinstance(f8, bytes):
                        t_end = parse_proto_timestamp(f8)
                    if t_end is None:
                        t_end = t_start
                    if t_start is None:
                        continue

                    # Step 14: User Input
                    if step_type == 14:
                        user_text = "User message"
                        if payload and isinstance(payload, bytes):
                            try:
                                for fn, wt, val in parse_proto_fields(payload):
                                    if wt == "bytes" and isinstance(val, bytes):
                                        try:
                                            sub = parse_proto_fields(val)
                                            for sfn, swt, sval in sub:
                                                if sfn == 2 and isinstance(sval, bytes):
                                                    txt = sval.decode("utf-8", errors="ignore").strip()
                                                    if txt:
                                                        # Strip <USER_REQUEST> tags if present
                                                        txt = re.sub(r"</?[A-Z_]+>", "", txt).strip()
                                                        user_text = f"User: {txt[:60]}"
                                                        break
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        events.append(
                            TimelineEvent(
                                timestamp=t_start,
                                kind="user_message",
                                turn_id=str(idx),
                                summary=user_text,
                                event_id=f"step:{idx}:user",
                            )
                        )

                    # Step 15: Assistant Response
                    elif step_type == 15:
                        model = step_model_map[model_boundaries[boundary]] if boundary >= 0 else model
                        latest_settings = settings
                        out_tokens = 0
                        f9 = fields.get(9)
                        if isinstance(f9, bytes):
                            u_dict = {
                                f[0]: f[2]
                                for f in parse_proto_fields(f9)
                                if f[1] == "varint" and isinstance(f[2], int)
                            }
                            out_tokens = u_dict.get(3, 0)
                        dur = (t_end - t_start) if (t_end and t_end > t_start) else None
                        events.append(
                            TimelineEvent(
                                timestamp=t_start,
                                kind="assistant_message",
                                turn_id=str(idx),
                                summary=f"Assistant response ({out_tokens} tokens)",
                                duration=dur,
                                tokens=out_tokens,
                                event_id=f"step:{idx}:assistant",
                            )
                        )

                    # Step 132: Tool Call / Tool Result
                    elif step_type == 132:
                        tool_name = "tool"
                        if payload and isinstance(payload, bytes):
                            try:
                                for fn, wt, val in parse_proto_fields(payload):
                                    if wt == "bytes" and isinstance(val, bytes):
                                        try:
                                            sub = parse_proto_fields(val)
                                            for sfn, swt, sval in sub:
                                                if sfn == 4 and isinstance(sval, bytes):
                                                    for tsfn, tswt, tsval in parse_proto_fields(sval):
                                                        if tsfn == 2 and isinstance(tsval, bytes):
                                                            t_name = tsval.decode("utf-8", errors="ignore").strip()
                                                            if t_name:
                                                                tool_name = t_name
                                                                break
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        dur = (t_end - t_start) if (t_end and t_end > t_start) else None
                        events.append(
                            TimelineEvent(
                                timestamp=t_start,
                                kind="tool_call",
                                turn_id=str(idx),
                                summary=f"Tool call: {tool_name}",
                                event_id=f"step:{idx}:tool",
                                duration=dur,
                            )
                        )
                    for index in range(first_new_event, len(events)):
                        events[index] = replace(events[index], reasoning_effort=settings[0],
                                                service_tier=settings[1], speed=settings[2])
                    if len(events) > first_new_event:
                        end = max(t_start, t_end if t_end is not None else t_start)
                        last_activity = end if last_activity is None else max(last_activity, end)
        except Exception:
            pass

        events.sort(key=lambda e: e.timestamp)
        created_at = events[0].timestamp if events else 0.0
        updated_at = last_activity if last_activity is not None else 0.0

        return SessionTimeline(
            session_id=session_id,
            agent=self.name,
            model=model,
            created_at=created_at,
            updated_at=updated_at,
            events=events,
            cwd=self._workspace_path(session_id),
            reasoning_effort=latest_settings[0],
            service_tier=latest_settings[1],
            speed=latest_settings[2],
        )

    def _workspace_path(self, session_id: str) -> str | None:
        """Look up the workspace folder for a session from conversation_summaries.db.

        The `workspace_uris` column holds a JSON array of file:// URIs and may be empty.
        Returns the first workspace as a plain filesystem path, or None if unknown.
        """
        summary_db = self.root / "conversation_summaries.db"
        if not summary_db.is_file():
            return None
        try:
            with closing(open_ro_db(summary_db)) as conn:
                row = conn.execute(
                    "SELECT workspace_uris FROM conversation_summaries WHERE conversation_id = ?",
                    (session_id,),
                ).fetchone()
        except Exception as e:
            logger.debug("Antigravity: could not read workspace for '%s': %s", session_id, e)
            return None

        if not row or not row[0]:
            return None
        try:
            uris = json.loads(row[0])
        except Exception:
            return None
        if not isinstance(uris, list):
            return None
        for uri in uris:
            if isinstance(uri, str) and uri.startswith("file://"):
                path = unquote(urlparse(uri).path)
                if path:
                    return path
        return None
