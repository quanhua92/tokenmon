"""Claude Code adapter: strictly read-only parser for local Claude Code session transcripts."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

from llm_monitor.adapters.base import BaseAdapter
from llm_monitor.models import GenerationSpan, SessionTimeline, TimelineEvent


def parse_iso_timestamp(val: object) -> float | None:
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


class ClaudeAdapter(BaseAdapter):
    """Adapter for extracting generation spans and session timelines from Claude Code."""

    def __init__(self, root: Path | str | None = None):
        self._custom_root = Path(root).expanduser().resolve() if root else None

    @property
    def name(self) -> str:
        return "claude"

    @property
    def root(self) -> Path:
        if self._custom_root:
            return self._custom_root
        env = os.environ.get("CLAUDE_HOME")
        if env:
            return Path(env).expanduser().resolve()
        return (Path.home() / ".claude").resolve()

    def detect(self) -> bool:
        r = self.root
        if not r.exists():
            return False
        projects_dir = r / "projects"
        return projects_dir.exists() or (r / "history.jsonl").exists()

    def _discover_session_files(self, max_sessions: int) -> list[Path]:
        projects_dir = self.root / "projects"
        candidates: list[Path] = []

        if projects_dir.exists():
            # Find all *.jsonl files across project directories
            for jsonl_file in projects_dir.glob("*/*.jsonl"):
                if jsonl_file.is_file() and jsonl_file.stat().st_size > 0:
                    candidates.append(jsonl_file)

        # Sort by modification time descending
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[:max_sessions]

    def collect(self, max_sessions: int = 64) -> list[GenerationSpan]:
        session_files = self._discover_session_files(max_sessions)
        all_spans: list[GenerationSpan] = []

        for p in session_files:
            session_id = p.stem
            spans = self._parse_session_spans(session_id, p)
            all_spans.extend(spans)

        all_spans.sort(key=lambda s: s.ended_at)
        return all_spans

    def _parse_session_spans(self, session_id: str, jsonl_path: Path) -> list[GenerationSpan]:
        spans: list[GenerationSpan] = []
        prev_event_time: float | None = None

        # Group assistant chunks by message id: msg_id -> list of (timestamp, record)
        message_chunks: dict[str, list[tuple[float, dict]]] = {}
        message_starts: dict[str, float | None] = {}

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

                    rec_type = record.get("type")
                    at = parse_iso_timestamp(record.get("timestamp"))
                    if at is None:
                        continue

                    if rec_type in {"user", "last-prompt", "tool_result"}:
                        prev_event_time = at

                    elif rec_type == "assistant":
                        msg = record.get("message", {})
                        msg_id = msg.get("id") or record.get("uuid") or f"msg_{at}"
                        if msg_id not in message_chunks:
                            message_chunks[msg_id] = []
                            message_starts[msg_id] = prev_event_time
                        message_chunks[msg_id].append((at, record))
        except Exception:
            return spans

        # Convert grouped messages into GenerationSpans
        for msg_id, chunks in message_chunks.items():
            if not chunks:
                continue

            first_at, first_rec = chunks[0]
            last_at, last_rec = chunks[-1]
            req_start = message_starts.get(msg_id)

            msg = last_rec.get("message", {})
            model = msg.get("model") or "claude"
            usage = msg.get("usage", {})
            tokens = usage.get("output_tokens") or 0

            # Any stream duration under 1.0s is excluded from speed metrics to avoid
            # sub-second timestamp jitter and placeholder block flush distortion.
            chunk_dur = last_at - first_at
            if len(chunks) > 1 and chunk_dur >= 1.0:
                started_at = first_at
                ended_at = last_at
                timing_source = "chunk-stream"
            elif req_start is not None and (last_at - req_start >= 1.0):
                started_at = req_start
                ended_at = last_at
                timing_source = "turn-span"
            else:
                started_at = first_at
                ended_at = last_at
                timing_source = "instant"

            duration = ended_at - started_at
            is_valid = True
            note = None

            if duration < 1.0:
                is_valid = False
                note = "duration_under_1s"
            elif tokens <= 0:
                is_valid = False
                note = "zero_tokens"
            elif (tokens / duration) > 400.0:
                # Sanity ceiling: Speeds > 400 TPS indicate collapsed event boundaries.
                is_valid = False
                note = "unconfirmed_boundary_tps"

            spans.append(
                GenerationSpan(
                    agent=self.name,
                    session_id=session_id,
                    turn_id=msg_id,
                    model=model,
                    tokens=tokens,
                    started_at=started_at,
                    ended_at=ended_at,
                    timing_source=timing_source,
                    is_valid=is_valid,
                    note=note,
                )
            )

        return spans

    def collect_sessions(self, max_sessions: int = 32) -> list[SessionTimeline]:
        session_files = self._discover_session_files(max_sessions)
        timelines: list[SessionTimeline] = []

        for p in session_files:
            timeline = self._parse_session_timeline(p.stem, p)
            if timeline.events:
                timelines.append(timeline)

        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        return timelines

    def _parse_session_timeline(self, session_id: str, jsonl_path: Path) -> SessionTimeline:
        events: list[TimelineEvent] = []
        model = "claude"
        seen_assistant_ids: set[str] = set()

        try:
            with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    rec_type = rec.get("type")
                    at = parse_iso_timestamp(rec.get("timestamp"))
                    if at is None:
                        continue

                    turn_id = rec.get("uuid") or f"turn_{at}"

                    if rec_type == "user":
                        msg = rec.get("message", {})
                        content = msg.get("content")
                        summary = "User message"
                        if isinstance(content, str) and content.strip():
                            summary = f"User: {content.strip()[:60]}"
                        elif isinstance(content, list) and content:
                            first_block = content[0]
                            if isinstance(first_block, dict) and first_block.get("text"):
                                summary = f"User: {first_block['text'][:60]}"

                        events.append(
                            TimelineEvent(
                                timestamp=at,
                                kind="user_message",
                                turn_id=turn_id,
                                summary=summary,
                            )
                        )

                    elif rec_type == "last-prompt":
                        prompt_text = rec.get("lastPrompt")
                        if prompt_text and not (events and events[-1].kind == "user_message" and abs(events[-1].timestamp - at) < 1.0):
                            events.append(
                                TimelineEvent(
                                    timestamp=at,
                                    kind="user_message",
                                    turn_id=turn_id,
                                    summary=f"User prompt: {prompt_text[:60]}",
                                )
                            )

                    elif rec_type == "assistant":
                        msg = rec.get("message", {})
                        msg_id = msg.get("id") or turn_id
                        cur_model = msg.get("model")
                        if cur_model:
                            model = cur_model

                        usage = msg.get("usage", {})
                        out_tokens = usage.get("output_tokens") or 0

                        # Check content blocks for tools or thinking
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    b_type = block.get("type")
                                    if b_type == "thinking":
                                        events.append(
                                            TimelineEvent(
                                                timestamp=at,
                                                kind="reasoning",
                                                turn_id=turn_id,
                                                summary="Thinking / reasoning",
                                            )
                                        )
                                    elif b_type == "tool_use":
                                        tname = block.get("name") or "tool"
                                        events.append(
                                            TimelineEvent(
                                                timestamp=at,
                                                kind="tool_call",
                                                turn_id=turn_id,
                                                summary=f"Tool call requested: {tname}",
                                            )
                                        )

                        # Only add one assistant summary event per message ID to prevent duplicate counts
                        if msg_id not in seen_assistant_ids:
                            seen_assistant_ids.add(msg_id)
                            events.append(
                                TimelineEvent(
                                    timestamp=at,
                                    kind="assistant_message",
                                    turn_id=turn_id,
                                    summary=f"Assistant response ({out_tokens} tokens)",
                                    tokens=out_tokens,
                                )
                            )

                    elif rec_type == "tool_result":
                        tname = rec.get("name") or "tool"
                        events.append(
                            TimelineEvent(
                                timestamp=at,
                                kind="tool_output",
                                turn_id=turn_id,
                                summary=f"Tool execution complete: {tname}",
                            )
                        )

                    elif rec_type == "system" and rec.get("subtype") == "turn_duration":
                        dur_ms = rec.get("durationMs")
                        dur_s = dur_ms / 1000.0 if isinstance(dur_ms, (int, float)) else None
                        events.append(
                            TimelineEvent(
                                timestamp=at,
                                kind="turn_end",
                                turn_id=turn_id,
                                summary="Turn complete",
                                duration=dur_s,
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
