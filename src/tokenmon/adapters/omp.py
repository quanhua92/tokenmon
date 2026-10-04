"""Oh My Pi adapter: read-only parsing of main and nested worker journals."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from tokenmon.adapters.base import BaseAdapter
from tokenmon.adapters.claude import parse_iso_timestamp
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, create_span, generation_metadata

logger = logging.getLogger(__name__)

_PROFILE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_RESERVED_PROFILE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\..*)?\Z", re.IGNORECASE)


def _resolve_root(custom_root: Path | None) -> Path:
    if custom_root is not None:
        return custom_root
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if agent_dir and agent_dir.strip():
        return Path(agent_dir).expanduser().resolve()
    profile = os.environ.get("OMP_PROFILE", os.environ.get("PI_PROFILE", "")).strip()
    if profile == "default":
        profile = ""
    if profile and (not _PROFILE.fullmatch(profile) or profile.endswith(".") or _RESERVED_PROFILE.fullmatch(profile)):
        logger.warning("OMPAdapter: ignoring invalid profile name %r", profile)
        profile = ""
    config_name = os.environ.get("PI_CONFIG_DIR") or ".omp"
    config_root = Path.home() / config_name.lstrip("/\\")
    if profile:
        config_root = config_root / "profiles" / profile
    xdg_home = os.environ.get("XDG_DATA_HOME")
    if xdg_home and sys.platform in {"linux", "darwin"}:
        data_root = Path(xdg_home).expanduser() / "omp"
        if profile:
            data_root = data_root / "profiles" / profile
        if data_root.is_dir():
            return data_root.resolve()
    return (config_root / "agent").resolve()


def _discover_session_files(root: Path, max_sessions: int) -> list[Path]:
    if max_sessions <= 0:
        return []
    directory = root / "sessions"
    candidates: list[tuple[float, str, Path]] = []
    # os.walk does not follow directory symlinks; no source paths are created.
    for parent, _, filenames in os.walk(directory, followlinks=False):
        for filename in filenames:
            if filename.startswith(".") or not filename.endswith(".jsonl"):
                continue
            path = Path(parent) / filename
            try:
                info = path.stat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode) and info.st_size:
                candidates.append((-info.st_mtime, str(path.relative_to(directory)), path))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in candidates[:max_sessions]]


def _string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _epoch_seconds(milliseconds: float | None) -> float | None:
    if milliseconds is None:
        return None
    seconds = milliseconds / 1000
    try:
        # Reject finite but unrenderable timestamps before they reach CLI views.
        datetime.fromtimestamp(seconds).astimezone()
    except (OverflowError, OSError, ValueError):
        return None
    return seconds


@dataclass(frozen=True)
class _State:
    model: str = "unknown"
    provider: str | None = None
    api: str | None = None
    effort: str | None = None
    tiers: dict[str, str] | str | None = None
    tier_override: str | None = None
    speed: str | None = None
    prompt: str | None = None

    def metadata(self) -> tuple[str | None, str | None, str | None]:
        tier = self.tier_override
        if tier is None and isinstance(self.tiers, str):
            tier = self.tiers
        elif tier is None and isinstance(self.tiers, dict):
            family = None
            if self.provider in {"openai", "openai-codex"}:
                family = "openai"
            elif self.api == "anthropic-messages":
                family = "anthropic"
            elif self.provider in {"google", "google-vertex"}:
                family = "google"
            tier = self.tiers.get(family) if family is not None else None
        return self.effort, tier, self.speed


def _model_state(state: _State, model: str, provider: str | None = None,
                 api: str | None = None) -> _State:
    if provider is None and "/" in model:
        provider = model.split("/", 1)[0]
    if provider and not model.startswith(provider + "/"):
        model = f"{provider}/{model}"
    changed = model != state.model or provider != state.provider
    return replace(state, model=model, provider=provider, api=api,
                   tier_override=None if changed else state.tier_override)


@dataclass
class _Message:
    source_id: str
    role: str
    timestamp: float | None
    completed_at: float | None
    confirmed_completion: bool
    duration_ms: float | None
    ttft_ms: float | None
    tokens: int
    content: str | list
    tool_name: str | None
    stop_reason: str | None
    state: _State
    owned: bool


@dataclass
class _Journal:
    session_id: str
    timestamp: float | None
    cwd: str | None
    state: _State
    messages: list[_Message]


def _read_journal(path: Path) -> _Journal | None:
    session_id = None
    header_at = None
    cwd = None
    parented = False
    state = _State()
    states: dict[str, _State] = {}
    messages: dict[str, _Message] = {}
    aliases: dict[str, str] = {}
    responses: dict[tuple[str | None, str], str] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream):
                try:
                    entry = json.loads(line)
                except (ValueError, RecursionError):
                    continue
                try:
                    if not isinstance(entry, dict):
                        continue
                    kind = entry.get("type")
                    if kind == "session" and session_id is None:
                        session_id = _string(entry.get("id"))
                        if session_id is None:
                            continue
                        header_at = parse_iso_timestamp(entry.get("timestamp"))
                        cwd = _string(entry.get("cwd"))
                        parented = _string(entry.get("parentSession")) is not None
                        if parented and header_at is None:
                            logger.warning("OMPAdapter: missing creation time for parented session %s", path)
                            return None
                        continue
                    if session_id is None or not isinstance(kind, str):
                        continue
                    entry_id = _string(entry.get("id"))
                    parent_id = _string(entry.get("parentId"))
                    current = states.get(parent_id, state) if parent_id is not None else state
                    if kind == "model_change":
                        model = _string(entry.get("model"))
                        if model is not None:
                            current = _model_state(current, model)
                    elif kind == "session_init":
                        model = _string(entry.get("resolvedModel"))
                        if current.model == "unknown" and model is not None:
                            current = _model_state(current, model)
                    elif kind == "thinking_level_change" and "thinkingLevel" in entry:
                        if entry.get("thinkingLevel") is None:
                            current = replace(current, effort=None)
                        elif _string(entry.get("thinkingLevel")) is not None:
                            current = replace(current, effort=_string(entry["thinkingLevel"]))
                    elif kind == "service_tier_change" and "serviceTier" in entry:
                        tiers = entry.get("serviceTier")
                        if tiers is None or isinstance(tiers, (str, dict)):
                            if isinstance(tiers, dict):
                                tiers = {key: value for key, raw in tiers.items()
                                         if isinstance(key, str) and (value := _string(raw)) is not None}
                            elif isinstance(tiers, str):
                                tiers = _string(tiers)
                            current = replace(current, tiers=tiers, tier_override=None)
                    elif kind == "message":
                        message = entry.get("message")
                        if not isinstance(message, dict):
                            continue
                        role = message.get("role")
                        if role not in ("user", "assistant", "toolResult"):
                            continue
                        usage = message.get("usage", {})
                        if not isinstance(usage, dict):
                            continue
                        content = message.get("content", [])
                        if not isinstance(content, (str, list)):
                            continue
                        source_id = aliases.get(entry_id, entry_id) if entry_id else f"line:{line_number}"
                        provider = _string(message.get("provider")) or current.provider
                        response_id = _string(message.get("responseId")) if role == "assistant" else None
                        response_key = (provider, response_id) if response_id else None
                        if response_key is not None:
                            source_id = responses.get(response_key, source_id)
                        previous = messages.get(source_id)
                        if previous is not None:
                            current = previous.state
                        model = _string(message.get("model"))
                        if role == "assistant" and model is not None:
                            current = _model_state(current, model, provider, _string(message.get("api")))
                        effort, tier, speed = generation_metadata(usage, message, entry)
                        current = replace(current,
                                          effort=effort if effort is not None else current.effort,
                                          tier_override=tier if tier is not None else current.tier_override,
                                          speed=speed if speed is not None else current.speed)
                        raw_start = _number(message.get("timestamp"))
                        at = _epoch_seconds(raw_start)
                        if at is None:
                            raw_start = None
                            at = parse_iso_timestamp(entry.get("timestamp"))
                        owned = not parented or (at is not None and header_at is not None and at >= header_at)
                        if role == "user" and owned:
                            current = replace(current, prompt=source_id)
                        duration = _number(message.get("duration"))
                        ended_at = _epoch_seconds(_number(message.get("completedAt")))
                        if ended_at is None and raw_start is not None and duration is not None and duration >= 0:
                            ended_at = _epoch_seconds(raw_start + duration)
                        confirmed_completion = ended_at is not None
                        if ended_at is None:
                            ended_at = parse_iso_timestamp(entry.get("timestamp"))
                            if ended_at is None:
                                ended_at = at
                        tokens = usage.get("output")
                        if (isinstance(tokens, bool) or not isinstance(tokens, int)
                                or tokens < 0 or _number(tokens) is None):
                            tokens = 0
                        # Tool output bodies/aggregate usage are not retained or counted.
                        messages[source_id] = _Message(
                            source_id, role, at, ended_at, confirmed_completion,
                            duration, _number(message.get("ttft")),
                            tokens, content if role != "toolResult" else [],
                            _string(message.get("toolName")), _string(message.get("stopReason")),
                            current, owned,
                        )
                        if entry_id is not None:
                            aliases[entry_id] = source_id
                        if response_key is not None:
                            responses[response_key] = source_id
                    if entry_id is not None:
                        states[entry_id] = current
                    state = current
                except Exception as exc:
                    logger.debug("OMPAdapter: skipping bad record %s:%s: %s", path, line_number + 1, exc)
                    continue
    except OSError as exc:
        logger.debug("OMPAdapter: cannot read session %s: %s", path, exc)
        return None
    if session_id is None:
        logger.debug("OMPAdapter: no usable session header in %s", path)
        return None
    return _Journal(session_id, header_at, cwd, state, list(messages.values()))


def _timeline(journal: _Journal) -> SessionTimeline | None:
    events: list[TimelineEvent] = []
    for message in journal.messages:
        if not message.owned:
            continue
        at = message.completed_at if message.role == "assistant" else message.timestamp
        if at is None:
            continue
        timestamp: float = at
        effort, tier, speed = message.state.metadata()
        turn_id = message.state.prompt or message.source_id

        def event(kind: str, summary: str, suffix: str, tokens: int | None = None) -> TimelineEvent:
            return TimelineEvent(
                timestamp=timestamp, kind=kind, turn_id=turn_id, summary=summary, tokens=tokens,
                event_id=f"entry:{message.source_id}:{suffix}",
                reasoning_effort=effort, service_tier=tier, speed=speed,
            )

        if message.role == "user":
            text = message.content if isinstance(message.content, str) else next(
                (block["text"] for block in message.content
                 if isinstance(block, dict) and isinstance(block.get("text"), str)), ""
            )
            events.append(event("user_message", f"User: {text.strip()[:60]}" if text.strip() else "User message", "user"))
        elif message.role == "toolResult":
            events.append(event("tool_output", f"Tool execution complete: {message.tool_name or 'tool'}", "tool_output"))
        elif message.role == "assistant":
            seen_tools: set[str | int] = set()
            if isinstance(message.content, list):
                for index, block in enumerate(message.content):
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if kind in ("thinking", "redactedThinking"):
                        events.append(event("reasoning", "Thinking / reasoning", f"reasoning:{index}"))
                    elif kind == "toolCall":
                        block_id = _string(block.get("id")) or index
                        if block_id not in seen_tools:
                            seen_tools.add(block_id)
                            name = _string(block.get("name")) or "tool"
                            events.append(event("tool_call", f"Tool call requested: {name}", f"tool:{block_id}"))
            events.append(event("assistant_message", f"Assistant response ({message.tokens} tokens)",
                                "assistant", message.tokens))
    events.sort(key=lambda event: event.timestamp)
    created_at = events[0].timestamp if events else journal.timestamp
    updated_at = events[-1].timestamp if events else journal.timestamp
    if created_at is None or updated_at is None:
        return None
    effort, tier, speed = journal.state.metadata()
    return SessionTimeline(
        session_id=journal.session_id, agent="omp", model=journal.state.model,
        created_at=created_at, updated_at=updated_at, events=events, cwd=journal.cwd,
        reasoning_effort=effort, service_tier=tier, speed=speed,
    )


def _span(session_id: str, message: _Message) -> GenerationSpan:
    duration, ttft = message.duration_ms, message.ttft_ms
    start = None
    timing_source = "omp-unconfirmed"
    note: str | None = "unconfirmed_generation_timing"
    if (message.confirmed_completion and message.completed_at is not None
            and duration is not None and ttft is not None and duration > 0
            and 0 <= ttft < duration):
        start = message.completed_at - (duration - ttft) / 1000
        if _epoch_seconds(start * 1000) is not None:
            timing_source = "omp-ttft"
            note = None
            first = next((block for block in message.content if isinstance(block, dict)), None) \
                if isinstance(message.content, list) else None
            if (first is None or first.get("type") not in ("text", "thinking")
                    or _string(first.get("text") if first.get("type") == "text" else first.get("thinking")) is None):
                note = "unconfirmed_output_boundary"
        else:
            start = None
    if message.stop_reason in ("aborted", "error"):
        note = "incomplete_response"
    effort, tier, speed = message.state.metadata()
    return create_span(
        agent="omp", session_id=session_id, turn_id=message.source_id,
        model=message.state.model, tokens=message.tokens, started_at=start,
        ended_at=message.completed_at, timing_source=timing_source, note=note,
        reasoning_effort=effort, service_tier=tier, speed=speed,
    )


class OMPAdapter(BaseAdapter):
    """Read recorded OMP output windows and session activity without modifying logs."""

    def __init__(self, root: Path | str | None = None):
        self._custom_root = Path(root).expanduser().resolve() if root else None
        self._timeline_sources: dict[str, Path] = {}

    @property
    def name(self) -> str:
        return "omp"

    @property
    def root(self) -> Path:
        return _resolve_root(self._custom_root)

    def detect(self) -> bool:
        try:
            return (self.root / "sessions").is_dir()
        except Exception as exc:
            logger.warning("OMPAdapter: cannot detect source: %s", exc)
            return False

    def _parse_session(self, jsonl_path: Path) -> tuple[list[GenerationSpan], SessionTimeline] | None:
        journal = _read_journal(jsonl_path)
        if journal is None:
            return None
        timeline = _timeline(journal)
        if timeline is None:
            return None
        spans: list[GenerationSpan] = []
        for message in journal.messages:
            if message.owned and message.role == "assistant":
                try:
                    spans.append(_span(journal.session_id, message))
                except Exception as exc:
                    logger.debug("OMPAdapter: skipping bad response %s in %s: %s",
                                 message.source_id, jsonl_path, exc)
        return spans, timeline

    def _collect_sources(self, max_sessions: int):
        seen: set[str] = set()
        try:
            paths = _discover_session_files(self.root, max_sessions)
        except Exception as exc:
            logger.warning("OMPAdapter: cannot discover sessions: %s", exc)
            return
        for path in paths:
            try:
                parsed = self._parse_session(path)
            except Exception as exc:
                logger.warning("OMPAdapter: skipping unparseable session %s: %s", path, exc)
                continue
            if parsed is None:
                continue
            spans, timeline = parsed
            if timeline.session_id in seen:
                continue
            seen.add(timeline.session_id)
            self._timeline_sources[timeline.session_id] = path
            yield spans, timeline

    def collect(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]:
        spans = [span for session_spans, _ in self._collect_sources(max_sessions) for span in session_spans
                 if min_timestamp is None or span.ended_at >= min_timestamp]
        spans.sort(key=lambda span: span.ended_at)
        return spans

    def collect_sessions(self, max_sessions: int = 32, min_timestamp: float | None = None) -> list[SessionTimeline]:
        timelines = [timeline for _, timeline in self._collect_sources(max_sessions)
                     if min_timestamp is None or timeline.updated_at >= min_timestamp]
        timelines.sort(key=lambda timeline: timeline.updated_at, reverse=True)
        return timelines

    def read_session(self, session_id: str) -> SessionTimeline | None:
        path = self._timeline_sources.get(session_id)
        if path is None:
            return super().read_session(session_id)
        try:
            parsed = self._parse_session(path)
        except Exception as exc:
            logger.warning("OMPAdapter: cannot refresh session %s: %s", session_id, exc)
            return None
        if parsed is None or parsed[1].session_id != session_id:
            return None
        return parsed[1]
