"""Shared read-only journals for Pi and Oh My Pi, with explicit source dialects."""

from __future__ import annotations

import json
import logging
import math
import os
import stat
from abc import abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from tokenmon.adapters.base import BaseAdapter
from tokenmon.adapters.claude import parse_iso_timestamp
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, create_span, generation_metadata


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


def _entry_agent(entry: dict) -> str | None:
    """Identify only explicit native schema markers, never model names."""
    if entry.get("type") == "model_change":
        if _string(entry.get("model")) is not None:
            return "omp"
        if _string(entry.get("provider")) and _string(entry.get("modelId")):
            return "pi"
    return None


def _probe_source_agents(root: Path) -> set[str]:
    agents: set[str] = set()
    for path in _discover_session_files(root, 5):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for index, line in enumerate(stream):
                    if index >= 64:
                        break
                    try:
                        entry = json.loads(line)
                    except (ValueError, RecursionError):
                        continue
                    if isinstance(entry, dict) and (agent := _entry_agent(entry)) is not None:
                        agents.add(agent)
                        break
        except OSError:
            continue
    return agents


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
    source_agent: str | None


def _read_journal(path: Path, agent: str) -> _Journal | None:
    logger = logging.getLogger(f"tokenmon.adapters.{agent}")
    source_agent = None
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
                    source_agent = source_agent or _entry_agent(entry)
                    if kind == "session" and session_id is None:
                        session_id = _string(entry.get("id"))
                        if session_id is None:
                            continue
                        header_at = parse_iso_timestamp(entry.get("timestamp"))
                        cwd = _string(entry.get("cwd"))
                        parented = _string(entry.get("parentSession")) is not None
                        if parented and header_at is None:
                            logger.warning("%sAdapter: missing creation time for parented session %s", agent, path)
                            return None
                        continue
                    if session_id is None or not isinstance(kind, str):
                        continue
                    entry_id = _string(entry.get("id"))
                    parent_id = _string(entry.get("parentId"))
                    current = states.get(parent_id, state) if parent_id is not None else state
                    if kind == "model_change":
                        model = _string(entry.get("model")) or _string(entry.get("modelId"))
                        if model is not None:
                            current = _model_state(current, model, _string(entry.get("provider")))
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
                            # System/custom messages still bridge recorded branch settings.
                            if entry_id is not None:
                                states[entry_id] = current
                            state = current
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
                        if agent == "pi":
                            effort = (_string(message.get("providerThinkingLevel"))
                                      or _string(message.get("thinkingLevel")) or effort)
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
                        duration = _number(message.get("duration")) if agent == "omp" else None
                        ended_at = _epoch_seconds(_number(message.get("completedAt"))) if agent == "omp" else None
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
                    logger.debug("%sAdapter: skipping bad record %s:%s: %s", agent, path, line_number + 1, exc)
                    continue
    except OSError as exc:
        logger.debug("%sAdapter: cannot read session %s: %s", agent, path, exc)
        return None
    if session_id is None:
        logger.debug("%sAdapter: no usable session header in %s", agent, path)
        return None
    return _Journal(session_id, header_at, cwd, state, list(messages.values()), source_agent)


def _timeline(journal: _Journal, agent: str) -> SessionTimeline | None:
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
        session_id=journal.session_id, agent=agent, model=journal.state.model,
        created_at=created_at, updated_at=updated_at, events=events, cwd=journal.cwd,
        reasoning_effort=effort, service_tier=tier, speed=speed,
    )


def _span(session_id: str, message: _Message, agent: str) -> GenerationSpan:
    duration, ttft = message.duration_ms, message.ttft_ms
    start = None
    timing_source = f"{agent}-unconfirmed"
    note: str | None = "unconfirmed_generation_timing"
    if (agent == "omp" and message.confirmed_completion and message.completed_at is not None
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
        agent=agent, session_id=session_id, turn_id=message.source_id,
        model=message.state.model, tokens=message.tokens, started_at=start,
        ended_at=message.completed_at, timing_source=timing_source, note=note,
        reasoning_effort=effort, service_tier=tier, speed=speed,
    )


class _PiJSONLAdapter(BaseAdapter):
    """Share discovery, reduction, timelines, and pinned reads across the Pi family."""

    def __init__(self, root: Path | str | None = None):
        self._custom_root = Path(root).expanduser().resolve() if root else None
        self._timeline_sources: dict[str, Path] = {}

    @property
    @abstractmethod
    def root(self) -> Path:
        ...

    @property
    def _default_source_agent(self) -> str:
        return "pi" if self.root.name == "agent" and self.root.parent.name == ".pi" else "omp"

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(f"tokenmon.adapters.{self.name}")

    def detect(self) -> bool:
        try:
            if not (self.root / "sessions").is_dir():
                return False
            agents = _probe_source_agents(self.root)
            return self.name in agents if agents else self.name == self._default_source_agent
        except Exception as exc:
            self._logger.warning("%sAdapter: cannot detect source: %s", self.name, exc)
            return False

    def _parse_session(self, jsonl_path: Path) -> tuple[list[GenerationSpan], SessionTimeline, str | None] | None:
        journal = _read_journal(jsonl_path, self.name)
        if journal is None:
            return None
        timeline = _timeline(journal, self.name)
        if timeline is None:
            return None
        spans: list[GenerationSpan] = []
        for message in journal.messages:
            if message.owned and message.role == "assistant":
                try:
                    spans.append(_span(journal.session_id, message, self.name))
                except Exception as exc:
                    self._logger.debug("%sAdapter: skipping bad response %s in %s: %s",
                                       self.name, message.source_id, jsonl_path, exc)
        return spans, timeline, journal.source_agent

    def _collect_sources(self, max_sessions: int):
        seen: set[str] = set()
        try:
            paths = _discover_session_files(self.root, max_sessions)
        except Exception as exc:
            self._logger.warning("%sAdapter: cannot discover sessions: %s", self.name, exc)
            return
        parsed_sources = []
        for path in paths:
            try:
                parsed = self._parse_session(path)
            except Exception as exc:
                self._logger.warning("%sAdapter: skipping unparseable session %s: %s", self.name, path, exc)
                continue
            if parsed is None:
                continue
            parsed_sources.append((path, parsed))
        source_agents = {parsed[2] for _, parsed in parsed_sources if parsed[2] is not None}
        unknown_owner = next(iter(source_agents)) if len(source_agents) == 1 else self._default_source_agent
        for path, (spans, timeline, source_agent) in parsed_sources:
            if source_agent is not None and source_agent != self.name:
                continue
            if source_agent is None and source_agents and unknown_owner != self.name:
                continue
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
            self._logger.warning("%sAdapter: cannot refresh session %s: %s", self.name, session_id, exc)
            return None
        if (parsed is None or parsed[1].session_id != session_id
                or parsed[2] not in (None, self.name)):
            return None
        return parsed[1]
