"""Claude Code adapter: strictly read-only parser for local Claude Code session transcripts."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from tokenmon.adapters.base import BaseAdapter
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, create_span, generation_metadata

logger = logging.getLogger(__name__)


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
        self._timeline_sources: dict[str, Path] = {}

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

    def _discover_session_files(self, max_sessions: int, min_timestamp: float | None = None) -> list[Path]:
        projects_dir = self.root / "projects"
        candidates: list[Path] = []

        if projects_dir.exists():
            # Find all *.jsonl files across project directories
            for jsonl_file in projects_dir.glob("*/*.jsonl"):
                if jsonl_file.is_file() and jsonl_file.stat().st_size > 0:
                    if min_timestamp is not None and jsonl_file.stat().st_mtime < min_timestamp:
                        continue
                    candidates.append(jsonl_file)

        # Sort by modification time descending
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[:max_sessions]

    def collect(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]:
        session_files = self._discover_session_files(max_sessions, min_timestamp=min_timestamp)
        all_spans: list[GenerationSpan] = []

        for p in session_files:
            try:
                session_id = p.stem
                spans = self._parse_session_spans(session_id, p, min_timestamp=min_timestamp)
                all_spans.extend(spans)
            except Exception as e:
                logger.error("ClaudeAdapter: skipping unparseable session file '%s': %s", p, e)

        all_spans.sort(key=lambda s: s.ended_at)
        return all_spans

    def _parse_session_spans(
        self,
        session_id: str,
        jsonl_path: Path,
        min_timestamp: float | None = None,
    ) -> list[GenerationSpan]:
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
                    if not isinstance(record, dict):
                        continue

                    rec_type = record.get("type")
                    at = parse_iso_timestamp(record.get("timestamp"))
                    if at is None or not isinstance(rec_type, str):
                        continue

                    if rec_type in {"user", "last-prompt", "tool_result"}:
                        prev_event_time = at

                    elif rec_type == "assistant":
                        msg = record.get("message", {})
                        if not isinstance(msg, dict) or not isinstance(msg.get("usage", {}), dict):
                            continue
                        msg_id = msg.get("id") or record.get("uuid")
                        if not isinstance(msg_id, str) or not msg_id:
                            msg_id = f"msg_{at}"
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
            if min_timestamp is not None and last_at < min_timestamp:
                continue
            req_start = message_starts.get(msg_id)

            msg = last_rec.get("message", {})
            model = msg.get("model")
            if not isinstance(model, str) or not model:
                model = "claude"
            usage = msg.get("usage", {})
            tokens = usage.get("output_tokens") or 0
            metadata_sources = []
            for _, chunk in reversed(chunks):
                chunk_msg = chunk.get("message", {})
                metadata_sources.extend((chunk_msg.get("usage"), chunk_msg, chunk))
            effort, tier, speed = generation_metadata(*metadata_sources)

            if len(chunks) > 1:
                started_at = first_at
                ended_at = last_at
                timing_source = "chunk-stream"
            elif req_start is not None:
                started_at = req_start
                ended_at = last_at
                timing_source = "turn-span"
            else:
                started_at = first_at
                ended_at = last_at
                timing_source = "instant"

            spans.append(
                create_span(
                    agent=self.name,
                    session_id=session_id,
                    turn_id=msg_id,
                    model=model,
                    tokens=tokens,
                    started_at=started_at,
                    ended_at=ended_at,
                    timing_source=timing_source,
                    reasoning_effort=effort,
                    service_tier=tier,
                    speed=speed,
                )
            )

        return spans

    def collect_sessions(self, max_sessions: int = 32, min_timestamp: float | None = None) -> list[SessionTimeline]:
        session_files = self._discover_session_files(max_sessions, min_timestamp=min_timestamp)
        timelines: list[SessionTimeline] = []

        for p in session_files:
            try:
                timeline = self._parse_session_timeline(p.stem, p)
                self._timeline_sources[p.stem] = p
                if timeline.events and (min_timestamp is None or timeline.updated_at >= min_timestamp):
                    timelines.append(timeline)
            except Exception as e:
                logger.error("ClaudeAdapter: skipping unparseable session timeline '%s': %s", p, e)

        timelines.sort(key=lambda t: t.updated_at, reverse=True)
        return timelines

    def read_session(self, session_id: str) -> SessionTimeline | None:
        path = self._timeline_sources.get(session_id)
        if path is None:
            return super().read_session(session_id)
        return self._parse_session_timeline(session_id, path)

    def _parse_session_timeline(self, session_id: str, jsonl_path: Path) -> SessionTimeline:
        events: list[TimelineEvent] = []
        model = "claude"
        assistant_event_indexes: dict[str, int] = {}
        message_event_indexes: dict[str, list[int]] = {}
        message_settings: dict[str, tuple[str | None, str | None, str | None]] = {}
        seen_content_blocks: set[tuple[str, str, str | int]] = set()
        cwd: str | None = None
        reasoning_effort = service_tier = speed = None

        try:
            with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
                for record_number, line in enumerate(f):
                    first_new_event = len(events)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue

                    if cwd is None:
                        c = rec.get("cwd")
                        if isinstance(c, str) and c:
                            cwd = c

                    rec_type = rec.get("type")
                    at = parse_iso_timestamp(rec.get("timestamp"))
                    if at is None or not isinstance(rec_type, str):
                        continue

                    turn_id = rec.get("uuid")
                    if not isinstance(turn_id, str) or not turn_id:
                        turn_id = f"turn_{at}"
                    record_id = rec.get("uuid")
                    if not isinstance(record_id, str) or not record_id:
                        record_id = f"record:{record_number}"
                    effort, tier, mode = generation_metadata(rec)
                    if effort is not None:
                        reasoning_effort = effort
                    if tier is not None:
                        service_tier = tier
                    if mode is not None:
                        speed = mode

                    if rec_type == "user":
                        msg = rec.get("message", {})
                        if not isinstance(msg, dict):
                            continue
                        content = msg.get("content")
                        summary = "User message"
                        if isinstance(content, str) and content.strip():
                            summary = f"User: {content.strip()[:60]}"
                        elif isinstance(content, list) and content:
                            first_block = content[0]
                            if isinstance(first_block, dict) and isinstance(first_block.get("text"), str):
                                summary = f"User: {first_block['text'][:60]}"

                        events.append(
                            TimelineEvent(
                                timestamp=at,
                                kind="user_message",
                                turn_id=turn_id,
                                summary=summary,
                                event_id=f"{record_id}:user",
                            )
                        )

                    elif rec_type == "last-prompt":
                        prompt_text = rec.get("lastPrompt")
                        if isinstance(prompt_text, str) and prompt_text and not (events and events[-1].kind == "user_message" and abs(events[-1].timestamp - at) < 1.0):
                            events.append(
                                TimelineEvent(
                                    timestamp=at,
                                    kind="user_message",
                                    turn_id=turn_id,
                                    summary=f"User prompt: {prompt_text[:60]}",
                                    event_id=f"{record_id}:prompt",
                                )
                            )

                    elif rec_type == "assistant":
                        msg = rec.get("message", {})
                        if not isinstance(msg, dict) or not isinstance(msg.get("usage", {}), dict):
                            continue
                        msg_id = msg.get("id")
                        if not isinstance(msg_id, str) or not msg_id:
                            msg_id = record_id
                        cur_model = msg.get("model")
                        if isinstance(cur_model, str) and cur_model:
                            model = cur_model

                        usage = msg.get("usage", {})
                        previous = message_settings.get(msg_id, (None, None, None))
                        reasoning_effort, service_tier, speed = generation_metadata(usage, msg, rec, {
                            "reasoning_effort": previous[0], "service_tier": previous[1], "speed": previous[2]})
                        message_settings[msg_id] = (reasoning_effort, service_tier, speed)
                        out_tokens = usage.get("output_tokens") or 0
                        if not isinstance(out_tokens, int) or out_tokens < 0:
                            out_tokens = 0

                        # Check content blocks for tools or thinking
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for block_index, block in enumerate(content):
                                if isinstance(block, dict):
                                    b_type = block.get("type")
                                    if not isinstance(b_type, str) or b_type not in {"thinking", "tool_use"}:
                                        continue
                                    block_id = block.get("id")
                                    block_key = (msg_id, b_type, block_id if isinstance(block_id, str) else block_index)
                                    if block_key in seen_content_blocks:
                                        continue
                                    seen_content_blocks.add(block_key)
                                    if b_type == "thinking":
                                        events.append(
                                            TimelineEvent(
                                                timestamp=at,
                                                kind="reasoning",
                                                turn_id=turn_id,
                                                summary="Thinking / reasoning",
                                                event_id=f"message:{msg_id}:thinking:{block_key[2]}",
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
                                                event_id=f"message:{msg_id}:tool:{block_key[2]}",
                                            )
                                        )

                        # Keep one response per message ID, updated with the final chunk's usage.
                        assistant_event = TimelineEvent(
                            timestamp=at,
                            kind="assistant_message",
                            turn_id=turn_id,
                            summary=f"Assistant response ({out_tokens} tokens)",
                            tokens=out_tokens,
                            event_id=f"message:{msg_id}:assistant",
                            reasoning_effort=reasoning_effort,
                            service_tier=service_tier,
                            speed=speed,
                        )
                        if msg_id in assistant_event_indexes:
                            index = assistant_event_indexes[msg_id]
                            events[index] = replace(
                                assistant_event, turn_id=events[index].turn_id,
                            )
                        else:
                            assistant_event_indexes[msg_id] = len(events)
                            events.append(assistant_event)
                        indexes = message_event_indexes.setdefault(msg_id, [])
                        indexes.extend(range(first_new_event, len(events)))
                        for index in indexes:
                            events[index] = replace(events[index], reasoning_effort=reasoning_effort,
                                                    service_tier=service_tier, speed=speed)

                    elif rec_type == "tool_result":
                        tname = rec.get("name") or "tool"
                        events.append(
                            TimelineEvent(
                                timestamp=at,
                                kind="tool_output",
                                turn_id=turn_id,
                                summary=f"Tool execution complete: {tname}",
                                event_id=f"{record_id}:tool_output",
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
                                event_id=f"{record_id}:turn_end",
                            )
                        )
                    for index in range(first_new_event, len(events)):
                        events[index] = replace(events[index], reasoning_effort=reasoning_effort,
                                                service_tier=service_tier, speed=speed)
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
