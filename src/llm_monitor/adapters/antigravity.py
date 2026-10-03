"""Antigravity (agy) adapter: strictly read-only parser for local Antigravity sessions and steps."""

from __future__ import annotations

import math
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from llm_monitor.adapters.base import BaseAdapter
from llm_monitor.models import GenerationSpan, SessionTimeline, TimelineEvent, create_span


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    res = 0
    shift = 0
    while True:
        b = data[offset]
        offset += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return res, offset


def parse_proto_fields(data: bytes) -> list[tuple[int, str, int | bytes | None]]:
    offset = 0
    fields: list[tuple[int, str, int | bytes | None]] = []
    limit = len(data)
    while offset < limit:
        tag_byte, offset = decode_varint(data, offset)
        field_num = tag_byte >> 3
        wire_type = tag_byte & 7
        if wire_type == 0:  # varint
            val, offset = decode_varint(data, offset)
            fields.append((field_num, "varint", val))
        elif wire_type == 2:  # length-delimited
            length, offset = decode_varint(data, offset)
            val = data[offset : offset + length]
            offset += length
            fields.append((field_num, "bytes", val))
        elif wire_type == 1:  # 64-bit
            val = data[offset : offset + 8]
            offset += 8
            fields.append((field_num, "fixed64", val))
        elif wire_type == 5:  # 32-bit
            val = data[offset : offset + 4]
            offset += 4
            fields.append((field_num, "fixed32", val))
        else:
            break
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

    def _discover_session_dbs(self, max_sessions: int, min_timestamp: float | None = None) -> list[Path]:
        conv_dir = self.root / "conversations"
        if not conv_dir.exists():
            return []

        # Check conversation_summaries.db for order if available
        summary_db = self.root / "conversation_summaries.db"
        candidates: list[Path] = []
        if summary_db.exists():
            try:
                with closing(open_ro_db(summary_db)) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT conversation_id, last_modified_time FROM conversation_summaries "
                        "ORDER BY last_modified_time DESC LIMIT ?",
                        (max_sessions * 2,),
                    )
                    for cid, lmt in cur.fetchall():
                        if min_timestamp is not None and lmt:
                            try:
                                dt = datetime.fromisoformat(str(lmt).replace("Z", "+00:00"))
                                if dt.timestamp() < min_timestamp:
                                    continue
                            except Exception:
                                pass
                        db_p = conv_dir / f"{cid}.db"
                        if db_p.is_file():
                            candidates.append(db_p)
                            if len(candidates) >= max_sessions:
                                return candidates
            except Exception:
                pass

        if not candidates:
            all_files = list(conv_dir.glob("*.db"))
            if min_timestamp is not None:
                all_files = [p for p in all_files if p.stat().st_mtime >= min_timestamp]
            all_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidates = all_files[:max_sessions]

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

    def collect(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]:
        dbs = self._discover_session_dbs(max_sessions, min_timestamp=min_timestamp)
        spans: list[GenerationSpan] = []

        for db_path in dbs:
            session_id = db_path.stem
            try:
                with closing(open_ro_db(db_path)) as conn:
                    cur = conn.cursor()

                    # 1. Parse model from gen_metadata
                    default_model = "gemini"
                    step_model_map: dict[int, str] = {}
                    try:
                        cur.execute("SELECT data FROM gen_metadata WHERE data IS NOT NULL;")
                        for (data,) in cur.fetchall():
                            for fn, wt, val in parse_proto_fields(data):
                                if fn == 1 and isinstance(val, bytes):
                                    m_name: str | None = None
                                    last_idx: int | None = None
                                    for sfn, swt, sval in parse_proto_fields(val):
                                        if sfn == 19 and isinstance(sval, bytes):
                                            m_name = sval.decode("utf-8", errors="ignore").strip()
                                        elif sfn == 20 and isinstance(sval, bytes):
                                            try:
                                                k_v = parse_proto_fields(sval)
                                                d = {k[0]: k[2] for k in k_v}
                                                k_b = d.get(1)
                                                v_b = d.get(2)
                                                if k_b == b"last_step_index" and isinstance(v_b, bytes):
                                                    last_idx = int(v_b.decode("utf-8", errors="ignore"))
                                            except Exception:
                                                pass
                                    if m_name:
                                        default_model = m_name
                                        if last_idx is not None:
                                            step_model_map[last_idx + 1] = m_name
                    except Exception:
                        pass

                    # 2. Extract step_type 15 (model outputs)
                    cur.execute(
                        "SELECT idx, metadata FROM steps WHERE step_type = 15 "
                        "AND metadata IS NOT NULL ORDER BY idx;"
                    )
                    for idx, meta in cur.fetchall():
                        fields = {f[0]: f[2] for f in parse_proto_fields(meta)}
                        f1 = fields.get(1)
                        f7 = fields.get(7) or fields.get(8)
                        f9 = fields.get(9)

                        if not isinstance(f1, bytes) or not isinstance(f7, bytes) or not isinstance(f9, bytes):
                            continue

                        t_start = parse_proto_timestamp(f1)
                        t_end = parse_proto_timestamp(f7)
                        if t_start is None or t_end is None or t_end <= t_start:
                            continue
                        if min_timestamp is not None and t_end < min_timestamp:
                            continue

                        u_dict = {
                            f[0]: f[2]
                            for f in parse_proto_fields(f9)
                            if f[1] == "varint" and isinstance(f[2], int)
                        }
                        out_tokens = u_dict.get(3, 0)
                        if out_tokens <= 0:
                            continue

                        model = step_model_map.get(idx, default_model)
                        span = create_span(
                            agent=self.name,
                            session_id=session_id,
                            turn_id=str(idx),
                            model=model,
                            started_at=t_start,
                            ended_at=t_end,
                            tokens=out_tokens,
                            timing_source="agy-step-proto",
                        )
                        if span is not None:
                            spans.append(span)
            except Exception:
                continue

        spans.sort(key=lambda s: s.ended_at)
        return spans

    def collect_sessions(self, max_sessions: int = 32, min_timestamp: float | None = None) -> list[SessionTimeline]:
        dbs = self._discover_session_dbs(max_sessions, min_timestamp=min_timestamp)
        timelines: list[SessionTimeline] = []

        for p in dbs:
            timeline = self.parse_session_timeline(p.stem)
            if timeline.events:
                timelines.append(timeline)

        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        return timelines

    def parse_session_timeline(self, session_id: str) -> SessionTimeline:
        db_path = self.root / "conversations" / f"{session_id}.db"
        if not db_path.exists():
            # Try fuzzy match if session_id is a prefix
            conv_dir = self.root / "conversations"
            matches = list(conv_dir.glob(f"{session_id}*.db"))
            if matches:
                db_path = matches[0]
                session_id = db_path.stem

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

        try:
            with closing(open_ro_db(db_path)) as conn:
                cur = conn.cursor()

                # Get model name from gen_metadata
                try:
                    cur.execute("SELECT data FROM gen_metadata WHERE data IS NOT NULL;")
                    for (data,) in cur.fetchall():
                        for fn, wt, val in parse_proto_fields(data):
                            if fn == 1 and isinstance(val, bytes):
                                for sfn, swt, sval in parse_proto_fields(val):
                                    if sfn == 19 and isinstance(sval, bytes):
                                        m_name = sval.decode("utf-8", errors="ignore").strip()
                                        if m_name:
                                            model = m_name
                except Exception:
                    pass

                # Read all steps
                cur.execute(
                    "SELECT idx, step_type, metadata, step_payload FROM steps "
                    "WHERE metadata IS NOT NULL ORDER BY idx;"
                )
                for idx, step_type, meta, payload in cur.fetchall():
                    fields = {f[0]: f[2] for f in parse_proto_fields(meta)}
                    f1 = fields.get(1)
                    f7 = fields.get(7) or fields.get(8) or f1

                    t_start = parse_proto_timestamp(f1) if isinstance(f1, bytes) else None
                    t_end = parse_proto_timestamp(f7) if isinstance(f7, bytes) else t_start
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
                            )
                        )

                    # Step 15: Assistant Response
                    elif step_type == 15:
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
                                duration=dur,
                            )
                        )
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
        )
