"""Native Pi behavior regressions using temporary, synthetic journals only."""

import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tokenmon.adapters import detect_available_adapters, get_adapter
from tokenmon.adapters.omp import OMPAdapter
from tokenmon.adapters.pi import PiAdapter
from tokenmon.analyzer import summarize_spans


EPOCH = datetime(2026, 10, 3, 10, tzinfo=timezone.utc).timestamp()


def iso(seconds=0):
    return datetime.fromtimestamp(EPOCH + seconds, timezone.utc).isoformat()


def header(session_id="pi-main", seconds=0, **fields):
    return {"type": "session", "version": 3, "id": session_id,
            "timestamp": iso(seconds), "cwd": "/synthetic/pi-project", **fields}


def entry(kind, entry_id, seconds=0, parent=None, **fields):
    return {"type": kind, "id": entry_id, "parentId": parent,
            "timestamp": iso(seconds), **fields}


def user(entry_id="u-1", seconds=1, parent=None, **fields):
    return entry("message", entry_id, seconds, parent, message={
        "role": "user", "timestamp": (EPOCH + seconds) * 1000,
        "content": [{"type": "text", "text": "synthetic prompt"}], **fields})


def assistant(entry_id="a-1", seconds=7, tokens=120, parent="u-1",
              request_seconds=2, **fields):
    return entry("message", entry_id, seconds, parent, message={
        "role": "assistant", "provider": "openai", "model": "pi-test-model",
        "api": "openai-responses", "responseId": "response-" + entry_id,
        "timestamp": (EPOCH + request_seconds) * 1000, "stopReason": "stop",
        "content": [{"type": "text", "text": "synthetic answer"}],
        "usage": {"output": tokens, "input": 900, "cacheRead": 800,
                  "cacheWrite": 700, "totalTokens": 2400}, **fields})


def canonical_records(session_id="pi-main", tokens=120):
    return [
        header(session_id),
        entry("model_change", "model", provider="openai", modelId="pi-test-model"),
        entry("thinking_level_change", "effort", parent="model", thinkingLevel="high"),
        user(parent="effort"),
        assistant(tokens=tokens, content=[
            {"type": "text", "text": "synthetic answer"},
            {"type": "thinking", "thinking": "synthetic reasoning"},
            {"type": "redactedThinking", "data": "opaque"},
            {"type": "toolCall", "id": "tool-1", "name": "synthetic_tool",
             "arguments": {"command": "never executed"}},
        ]),
        entry("message", "result", 8, "a-1", message={
            "role": "toolResult", "toolName": "synthetic_tool", "toolCallId": "tool-1",
            "timestamp": (EPOCH + 8) * 1000,
            "content": [{"type": "text", "text": "synthetic tool output"}],
            "details": {"usage": {"output": 999999}},
        }),
    ]


class TestPiAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_records(self, path, records, mode="w"):
        with path.open(mode, encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def write_session(self, records, relative="project/main.jsonl", mtime=None):
        path = self.root / "sessions" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write_records(path, records)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def single_span(self, response):
        self.write_session([header(), user(), response])
        spans = PiAdapter(self.root).collect()
        self.assertEqual(len(spans), 1)
        return spans[0]

    def test_explicit_root_overrides_env_and_expands_home(self):
        env = {"HOME": str(self.root), "PI_CODING_AGENT_DIR": "~/env-agent",
               "OMP_PROFILE": "work", "PI_PROFILE": "another",
               "PI_CONFIG_DIR": ".omp", "XDG_DATA_HOME": str(self.root / "xdg")}
        with patch.dict(os.environ, env, clear=True), patch.object(Path, "home", return_value=self.root):
            self.assertEqual(PiAdapter(self.root / "explicit" / "..").root, self.root)
            self.assertEqual(PiAdapter("~/explicit").root, self.root / "explicit")
            self.assertEqual(PiAdapter().root, self.root / "env-agent")

    def test_native_default_ignores_omp_config_profiles_and_xdg(self):
        migrated = self.root / "xdg" / "omp" / "profiles" / "work"
        migrated.mkdir(parents=True)
        env = {"HOME": str(self.root), "PI_CONFIG_DIR": ".omp", "OMP_PROFILE": "work", "PI_PROFILE": "another",
               "XDG_DATA_HOME": str(self.root / "xdg")}
        with patch.dict(os.environ, env, clear=True), patch.object(Path, "home", return_value=self.root):
            self.assertEqual(PiAdapter().root, self.root / ".pi" / "agent")
            os.environ["PI_CODING_AGENT_DIR"] = ""
            self.assertEqual(PiAdapter().root, self.root / ".pi" / "agent")

    def test_missing_and_non_directory_sources_are_not_detected(self):
        adapter = PiAdapter(self.root)
        self.assertFalse(adapter.detect())
        self.assertEqual(adapter.collect(), [])
        self.assertEqual(adapter.collect_sessions(), [])
        sessions = self.root / "sessions"
        sessions.write_text("not a directory", encoding="utf-8")
        self.assertFalse(adapter.detect())

    def test_native_messages_preserve_tokens_without_throughput(self):
        self.write_session(canonical_records())
        adapter = PiAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual((span.agent, span.session_id, span.turn_id), ("pi", "pi-main", "a-1"))
        self.assertEqual(span.model, "openai/pi-test-model")
        self.assertEqual(span.reasoning_effort, "high")
        self.assertEqual(span.tokens, 120)
        self.assertEqual(span.ended_at, EPOCH + 7)
        self.assertEqual(span.timing_source, "pi-unconfirmed")
        self.assertEqual(span.note, "unconfirmed_generation_timing")
        self.assertFalse(span.is_valid)
        self.assertIsNone(span.tps)
        summary = summarize_spans(spans, "all", span.model)
        self.assertEqual(summary.excluded_spans, 1)
        self.assertEqual(summary.valid_spans, 0)
        self.assertIsNone(summary.weighted_tps)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual((timeline.agent, timeline.cwd), ("pi", "/synthetic/pi-project"))
        self.assertEqual(timeline.model, span.model)
        self.assertEqual(timeline.reasoning_effort, "high")
        self.assertEqual((timeline.user_messages, timeline.assistant_messages, timeline.tool_calls), (1, 1, 1))
        self.assertEqual(timeline.total_tokens, 120)
        self.assertEqual(sum(event.kind == "reasoning" for event in timeline.events), 2)
        self.assertEqual(sum(event.kind == "tool_output" for event in timeline.events), 1)
        for event in timeline.events:
            self.assertEqual(event.turn_id, "u-1")
            if event.kind != "assistant_message":
                self.assertIsNone(event.tokens)
            if event.kind in {"reasoning", "tool_call", "assistant_message"}:
                self.assertEqual(event.timestamp, EPOCH + 7)
                self.assertEqual(event.reasoning_effort, "high")
            if event.kind in {"tool_call", "tool_output"}:
                self.assertIn("synthetic_tool", event.summary)
        self.assertEqual((timeline.created_at, timeline.updated_at), (EPOCH + 1, EPOCH + 8))

    def test_pi_never_uses_optional_request_duration_as_generation_timing(self):
        extras = [{}, {"duration": 5000}, {"duration": 5000, "ttft": 1000},
                  {"duration": 5000, "ttft": 1000, "completedAt": (EPOCH + 99) * 1000},
                  {"duration": float("nan"), "ttft": True, "completedAt": float("inf")}]
        for fields in extras:
            with self.subTest(fields=fields):
                span = self.single_span(assistant(**fields))
                self.assertFalse(span.is_valid)
                self.assertIsNone(span.tps)
                self.assertEqual(span.ended_at, EPOCH + 7)
                self.assertEqual(span.timing_source, "pi-unconfirmed")
                self.assertEqual(span.note, "unconfirmed_generation_timing")
                self.assertEqual(PiAdapter(self.root).collect_sessions()[0].updated_at, EPOCH + 7)

    def test_aborted_and_error_responses_preserve_output_as_incomplete(self):
        for reason in ("aborted", "error"):
            with self.subTest(reason=reason):
                span = self.single_span(assistant(stopReason=reason, duration=5000, ttft=1000))
                self.assertEqual(span.tokens, 120)
                self.assertEqual(span.note, "incomplete_response")
                self.assertEqual(span.timing_source, "pi-unconfirmed")
                self.assertFalse(span.is_valid)
                self.assertIsNone(span.tps)
                self.assertEqual(PiAdapter(self.root).collect_sessions()[0].total_tokens, 120)

    def test_completion_uses_entry_iso_not_request_start_and_events_sort(self):
        self.write_session([header(), user("late-user", 6),
                            assistant(seconds=20, request_seconds=2, parent="late-user"),
                            user("earlier-user", 4)])
        adapter = PiAdapter(self.root)
        self.assertEqual(adapter.collect()[0].ended_at, EPOCH + 20)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual([event.timestamp for event in timeline.events],
                         [EPOCH + 4, EPOCH + 6, EPOCH + 20])
        response = next(event for event in timeline.events if event.kind == "assistant_message")
        self.assertEqual(response.turn_id, "late-user")
        self.assertEqual(timeline.updated_at, EPOCH + 20)

    def test_missing_entry_time_falls_back_to_message_milliseconds(self):
        for timestamp in (None, "invalid", "2026-10-03T10:00:07", True, []):
            with self.subTest(timestamp=timestamp):
                response = assistant(request_seconds=3)
                response["timestamp"] = timestamp
                span = self.single_span(response)
                self.assertEqual(span.ended_at, EPOCH + 3)
                timeline = PiAdapter(self.root).collect_sessions()[0]
                self.assertEqual(timeline.updated_at, EPOCH + 3)
                self.assertEqual(timeline.events[-1].timestamp, EPOCH + 3)
                self.assertIsNone(span.tps)

    def test_parent_linked_models_effort_and_prompt_context(self):
        left = assistant("left-a", 10, parent="left-user")
        right = assistant("right-a", 20, parent="right-user")
        for response in (left, right):
            response["message"].pop("provider")
            response["message"].pop("model")
            response["message"]["content"].append({"type": "thinking", "thinking": "reasoning"})
        self.write_session([
            header(), entry("model_change", "base", provider="openai", modelId="left-model"),
            entry("thinking_level_change", "left-effort", parent="base", thinkingLevel="high"),
            user("left-user", 1, "left-effort"),
            entry("model_change", "right-model", parent="base", provider="google", modelId="right-model"),
            entry("thinking_level_change", "right-effort", parent="right-model", thinkingLevel="low"),
            user("right-user", 8, "right-effort"), left, right,
        ])
        adapter = PiAdapter(self.root)
        spans = {span.turn_id: span for span in adapter.collect()}
        self.assertEqual((spans["left-a"].model, spans["left-a"].reasoning_effort),
                         ("openai/left-model", "high"))
        self.assertEqual((spans["right-a"].model, spans["right-a"].reasoning_effort),
                         ("google/right-model", "low"))
        timeline = adapter.collect_sessions()[0]
        self.assertEqual((timeline.model, timeline.reasoning_effort), ("google/right-model", "low"))
        observed = sorted((event.kind, event.turn_id, event.reasoning_effort)
                          for event in timeline.events
                          if event.kind in ("reasoning", "assistant_message"))
        self.assertEqual(observed, sorted([
            ("reasoning", "left-user", "high"),
            ("assistant_message", "left-user", "high"),
            ("reasoning", "right-user", "low"),
            ("assistant_message", "right-user", "low"),
        ]))

    def test_system_message_keeps_parent_branch_settings_without_a_timeline_event(self):
        self.write_session([
            header(),
            entry("model_change", "left", provider="openai", modelId="left-model"),
            entry("thinking_level_change", "left-effort", parent="left", thinkingLevel="high"),
            entry("model_change", "right", provider="google", modelId="right-model"),
            entry("thinking_level_change", "right-effort", parent="right", thinkingLevel="low"),
            entry("message", "system", 1, "left-effort", message={
                "role": "system", "timestamp": (EPOCH + 1) * 1000,
                "content": "synthetic system message", "sections": [], "toolsAdded": [],
            }),
            user("branch-user", 2, "system"),
            assistant("branch-response", 10, parent="branch-user", provider=None, model=None),
        ])
        adapter = PiAdapter(self.root)
        span = adapter.collect()[0]
        self.assertEqual((span.model, span.reasoning_effort), ("openai/left-model", "high"))
        timeline = adapter.collect_sessions()[0]
        self.assertEqual([event.kind for event in timeline.events],
                         ["user_message", "assistant_message"])
        self.assertEqual(timeline.events[-1].reasoning_effort, "high")

    def test_assistant_explicit_model_and_native_effort_precedence(self):
        cases = [({}, "high"), ({"thinkingLevel": "medium"}, "medium"),
                 ({"thinkingLevel": "medium", "providerThinkingLevel": "minimal"}, "minimal")]
        for fields, expected in cases:
            with self.subTest(fields=fields):
                self.write_session([header(),
                    entry("model_change", "model", provider="google", modelId="inherited"),
                    entry("thinking_level_change", "effort", parent="model", thinkingLevel="high"),
                    user(parent="effort"), assistant(model="own-model", **fields)])
                adapter = PiAdapter(self.root)
                span = adapter.collect()[0]
                self.assertEqual(span.model, "openai/own-model")
                self.assertEqual(span.reasoning_effort, expected)
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.reasoning_effort, expected)
                self.assertEqual(timeline.events[-1].reasoning_effort, expected)
        span = self.single_span(assistant(model="openai/own-model"))
        self.assertEqual(span.model, "openai/own-model")

    def test_missing_metadata_is_null_without_model_name_inference(self):
        self.write_session([header(), user(), assistant(model="fast-high-priority-looking-model")])
        adapter = PiAdapter(self.root)
        span = adapter.collect()[0]
        timeline = adapter.collect_sessions()[0]
        for item in (span, timeline, *timeline.events):
            self.assertIsNone(item.reasoning_effort)
            self.assertIsNone(item.service_tier)
            self.assertIsNone(item.speed)
            self.assertIsNone(item.speed_mode)
        self.assertEqual(self.single_span(assistant(provider=None, model=None)).model, "unknown")

    def test_effort_reset_does_not_rewrite_historical_response(self):
        records = canonical_records()[:5]
        records.extend([
            entry("thinking_level_change", "reset", 9, "a-1", thinkingLevel=None),
            user("new-user", 10, "reset"), assistant("new-a", 20, parent="new-user"),
        ])
        self.write_session(records)
        adapter = PiAdapter(self.root)
        spans = {span.turn_id: span for span in adapter.collect()}
        self.assertEqual(spans["a-1"].reasoning_effort, "high")
        self.assertIsNone(spans["new-a"].reasoning_effort)
        self.assertIsNone(adapter.collect_sessions()[0].reasoning_effort)

    def test_only_nonnegative_integer_usage_output_is_counted(self):
        for usage in ({}, {"input": 900, "totalTokens": 2400}, {"output": True},
                      {"output": -1}, {"output": 1.5}, {"output": "120"},
                      {"output": float("nan")}, {"output": float("inf")}):
            with self.subTest(usage=usage):
                span = self.single_span(assistant(usage=usage))
                self.assertEqual(span.tokens, 0)
                self.assertEqual(PiAdapter(self.root).collect_sessions()[0].total_tokens, 0)
                self.assertFalse(span.is_valid)
                self.assertIsNone(span.tps)

    def test_entry_and_response_duplicates_use_latest_usage_and_stable_identity(self):
        initial = assistant(tokens=2)
        same_entry = assistant(tokens=30)
        final = assistant("duplicate-response", tokens=140, responseId="response-a-1", content=[
            {"type": "thinking", "thinking": "final reasoning"},
            {"type": "toolCall", "id": "tool-1", "name": "synthetic_tool"},
        ])
        path = self.write_session([header(), user(), initial])
        adapter = PiAdapter(self.root)
        before = adapter.collect_sessions()[0]
        self.write_records(path, [same_entry, final], "a")
        after = adapter.read_session("pi-main")
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)
        self.assertEqual((spans[0].turn_id, spans[0].tokens), ("a-1", 140))
        self.assertEqual((after.assistant_messages, after.tool_calls, after.total_tokens), (1, 1, 140))
        self.assertEqual(sum(event.kind == "reasoning" for event in after.events), 1)
        first_id = next(event.event_id for event in before.events if event.kind == "assistant_message")
        final_id = next(event.event_id for event in after.events if event.kind == "assistant_message")
        self.assertEqual(first_id, final_id)

    def test_equal_timestamp_distinct_response_ids_and_providers_are_retained(self):
        self.write_session([header(), user(), assistant("a-1"), assistant("a-2"),
                            assistant("a-3", provider="other", responseId="response-a-1")])
        adapter = PiAdapter(self.root)
        self.assertEqual(len(adapter.collect()), 3)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual((timeline.assistant_messages, timeline.total_tokens), (3, 360))
        self.assertEqual(len({event.event_id for event in timeline.events}), len(timeline.events))

    def test_idless_distinct_records_keep_physical_line_identity_on_refresh(self):
        response = assistant()
        response.pop("id")
        response["message"].pop("responseId")
        records = [header(), user(), response, copy.deepcopy(response)]
        path = self.write_session(records)
        adapter = PiAdapter(self.root)
        before = adapter.collect_sessions()[0]
        self.assertEqual(len(adapter.collect()), 2)
        records[2]["message"]["usage"]["output"] = 140
        self.write_records(path, records)
        after = adapter.read_session("pi-main")
        self.assertEqual(after.total_tokens, 260)
        self.assertEqual([event.event_id for event in before.events],
                         [event.event_id for event in after.events])
        self.assertEqual(len({event.event_id for event in after.events}), len(after.events))

    def test_completion_cutoff_keeps_earlier_model_effort_and_full_prompt(self):
        records = canonical_records()[:5]
        records[4]["message"].pop("provider")
        records[4]["message"].pop("model")
        path = self.write_session(records, mtime=100)
        adapter = PiAdapter(self.root)
        span = adapter.collect()[0]
        timeline = adapter.collect_sessions()[0]
        for cutoff in (EPOCH + 3, EPOCH + 7):
            with self.subTest(cutoff=cutoff):
                retained = adapter.collect(min_timestamp=cutoff)
                self.assertEqual(retained, [span])
                self.assertEqual(retained[0].model, "openai/pi-test-model")
                self.assertEqual(retained[0].reasoning_effort, "high")
                retained_sessions = adapter.collect_sessions(min_timestamp=cutoff)
                self.assertEqual(retained_sessions, [timeline])
                self.assertEqual(retained_sessions[0].events[0].timestamp, EPOCH + 1)
        self.assertEqual(adapter.collect(min_timestamp=EPOCH + 7.001), [])
        self.assertEqual(adapter.collect_sessions(min_timestamp=EPOCH + 7.001), [])
        os.utime(path, (EPOCH + 1000, EPOCH + 1000))
        self.assertEqual(adapter.collect(min_timestamp=EPOCH + 100), [])
        self.assertEqual(adapter.collect_sessions(min_timestamp=EPOCH + 100), [])

    def test_fork_excludes_copied_requests_even_when_completion_is_after_creation(self):
        self.write_session(canonical_records(), "parent.jsonl")
        copied = assistant("copied-a", 15, 500, "copied-user", request_seconds=9)
        own = assistant("child-a", 20, 80, "child-user", request_seconds=10)
        own["message"].pop("provider")
        own["message"].pop("model")
        self.write_session([
            header("pi-child", 10, parentSession="pi-main"),
            entry("model_change", "model", provider="openai", modelId="pi-test-model"),
            entry("thinking_level_change", "effort", parent="model", thinkingLevel="high"),
            user("copied-user", 9, "effort"), copied,
            entry("message", "copied-result", 18, message={"role": "toolResult",
                  "timestamp": (EPOCH + 9) * 1000, "toolName": "inherited_tool", "content": "copied"}),
            user("child-user", 10, "effort"), own,
        ], "child.jsonl")
        adapter = PiAdapter(self.root)
        self.assertEqual(sum(span.tokens for span in adapter.collect()), 200)
        child = next(item for item in adapter.collect_sessions() if item.session_id == "pi-child")
        self.assertEqual((child.user_messages, child.assistant_messages, child.total_tokens), (1, 1, 80))
        self.assertEqual((child.model, child.reasoning_effort), ("openai/pi-test-model", "high"))
        self.assertEqual(child.created_at, EPOCH + 10)

    def test_fork_without_creation_evidence_is_not_attributed(self):
        for timestamp in (None, "invalid", "2026-10-03T10:00:00"):
            with self.subTest(timestamp=timestamp):
                self.write_session([header("child", parentSession="parent", timestamp=timestamp), user(), assistant()])
                adapter = PiAdapter(self.root)
                self.assertEqual(adapter.collect(), [])
                self.assertEqual(adapter.collect_sessions(), [])

    def test_malformed_records_and_truncated_tail_do_not_hide_later_records(self):
        junk = [[], None, "text", 42, True, {"type": []}, {"type": "unknown"},
                {"type": "message", "message": None}, {"type": "message", "message": []},
                entry("message", "bad-role", message={"role": []}),
                entry("model_change", "bad-model", provider=[], modelId={}),
                entry("thinking_level_change", "bad-effort", thinkingLevel=[])]
        path = self.write_session([header(), *junk, user(), assistant()])
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{not json}\n{"type":"message"')
        self.write_session([header("", timestamp="bad"), [], "broken"], "broken.jsonl")
        adapter = PiAdapter(self.root)
        self.assertEqual([span.tokens for span in adapter.collect()], [120])
        self.assertEqual(len(adapter.collect_sessions()), 1)
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n" + json.dumps(assistant("later", 20, 80)) + "\n")
        self.assertEqual([span.tokens for span in adapter.collect()], [120, 80])
        self.assertEqual(adapter.collect_sessions()[0].total_tokens, 200)

    def test_malformed_nested_shapes_do_not_discard_a_later_good_response(self):
        cases = [{"usage": None}, {"usage": []}, {"content": None}, {"content": {}},
                 {"content": [None, [], "bad", {"type": []}, {"type": "text", "text": []}]}]
        for fields in cases:
            with self.subTest(fields=fields):
                self.write_session([header(), user(), assistant("malformed", **fields),
                                    assistant("good", 20, 80)])
                adapter = PiAdapter(self.root)
                good = [span for span in adapter.collect() if span.turn_id == "good"]
                self.assertEqual(len(good), 1)
                self.assertEqual(good[0].tokens, 80)
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(sum(event.kind == "assistant_message" and event.tokens == 80
                                     for event in timeline.events), 1)

    def test_bad_message_times_use_valid_entry_time_and_skip_unusable_user_times(self):
        bad_times = [True, "1000", float("nan"), float("inf"), 1e308, None]
        for value in bad_times:
            with self.subTest(value=value):
                bad_user = user("bad-user", timestamp=value)
                bad_user["timestamp"] = "2026-10-03T10:00:01"
                self.write_session([header(), bad_user, user(), assistant(timestamp=value)])
                adapter = PiAdapter(self.root)
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.user_messages, 1)
                self.assertEqual(timeline.updated_at, EPOCH + 7)
                self.assertEqual(adapter.collect()[0].ended_at, EPOCH + 7)

    def test_unreadable_source_isolated_from_readable_session(self):
        blocked = self.write_session(canonical_records("blocked"), "blocked.jsonl", mtime=200)
        self.write_session(canonical_records("readable", 80), "readable.jsonl", mtime=100)
        original_open = Path.open

        def readable_open(path, *args, **kwargs):
            if path == blocked:
                raise PermissionError("synthetic unreadable source")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", readable_open):
            adapter = PiAdapter(self.root)
            self.assertEqual([(span.session_id, span.tokens) for span in adapter.collect()], [("readable", 80)])
            self.assertEqual([item.session_id for item in adapter.collect_sessions()], ["readable"])

    def test_header_and_user_only_sessions_preserve_activity_without_spans(self):
        self.write_session([header("empty", cwd="")], "empty.jsonl")
        self.write_session([header("user-only"), user(seconds=20)], "user.jsonl")
        adapter = PiAdapter(self.root)
        self.assertEqual(adapter.collect(), [])
        timelines = adapter.collect_sessions()
        self.assertEqual([item.session_id for item in timelines], ["user-only", "empty"])
        self.assertEqual(timelines[1].events, [])
        self.assertIsNone(timelines[1].cwd)
        self.assertEqual((timelines[1].created_at, timelines[1].updated_at), (EPOCH, EPOCH))
        self.assertEqual((timelines[0].created_at, timelines[0].updated_at), (EPOCH + 20, EPOCH + 20))
        self.assertEqual([item.session_id for item in adapter.collect_sessions(min_timestamp=EPOCH + 1)],
                         ["user-only"])

    def test_bounded_discovery_uses_mtime_but_final_sessions_use_activity_order(self):
        self.write_session([header("new-activity"), user(seconds=50), assistant(seconds=60)],
                           "a.jsonl", mtime=100)
        self.write_session([header("new-file"), user(seconds=1), assistant(seconds=7)],
                           "b.jsonl", mtime=300)
        self.write_session([header("middle"), user(seconds=10), assistant(seconds=20)],
                           "nested/c.jsonl", mtime=200)
        adapter = PiAdapter(self.root)
        self.assertEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "new-file")
        self.assertEqual([item.session_id for item in adapter.collect_sessions(max_sessions=2)],
                         ["middle", "new-file"])
        self.assertEqual([item.session_id for item in adapter.collect_sessions()],
                         ["new-activity", "middle", "new-file"])
        self.assertEqual([span.ended_at for span in adapter.collect()], [EPOCH + 7, EPOCH + 20, EPOCH + 60])
        for limit in (0, -1):
            self.assertEqual(adapter.collect(max_sessions=limit), [])
            self.assertEqual(adapter.collect_sessions(max_sessions=limit), [])

    def test_duplicate_session_header_uses_newest_source_once(self):
        self.write_session(canonical_records("same", 30), "older.jsonl", mtime=100)
        self.write_session(canonical_records("same", 80), "newer.jsonl", mtime=200)
        adapter = PiAdapter(self.root)
        self.assertEqual([span.tokens for span in adapter.collect()], [80])
        self.assertEqual(len(adapter.collect_sessions()), 1)
        self.assertEqual(adapter.read_session("same").total_tokens, 80)

    def test_pinned_refresh_survives_failure_and_newer_discovery_sources(self):
        path = self.write_session(canonical_records("pinned"), "nested/scout.jsonl", mtime=100)
        adapter = PiAdapter(self.root)
        initial = adapter.collect_sessions(max_sessions=1)[0]
        self.write_session(canonical_records("new-main"), "new.jsonl", mtime=300)
        self.write_session(canonical_records("new-worker"), "other/scout.jsonl", mtime=400)
        self.assertEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "new-worker")
        hidden = path.with_suffix(".hidden")
        path.rename(hidden)
        self.assertIsNone(adapter.read_session("pinned"))
        hidden.rename(path)
        self.write_records(path, [assistant(tokens=140, providerThinkingLevel="minimal")], "a")
        refreshed = adapter.read_session("pinned")
        self.assertEqual((refreshed.session_id, refreshed.total_tokens), ("pinned", 140))
        old_response = next(event for event in initial.events if event.kind == "assistant_message")
        new_response = next(event for event in refreshed.events if event.kind == "assistant_message")
        self.assertEqual(old_response.event_id, new_response.event_id)
        self.assertEqual(new_response.reasoning_effort, "minimal")
        encoded = json.dumps(user("same-time-user", 8))
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded[:20])
        self.assertEqual(adapter.read_session("pinned").user_messages, 1)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded[20:] + "\n")
        complete = adapter.read_session("pinned")
        self.assertEqual(complete.user_messages, 2)
        self.assertEqual(sum(event.event_id == "entry:same-time-user:user" for event in complete.events), 1)
        self.write_records(path, [header("wrong-id"), user(), assistant()])
        self.assertIsNone(adapter.read_session("pinned"))
        self.write_records(path, canonical_records("pinned"))
        self.assertEqual(adapter.read_session("pinned").session_id, "pinned")

    def test_unknown_pinned_session_discovers_header_not_filename(self):
        self.write_session(canonical_records("header-uuid"), "project/scout.jsonl")
        adapter = PiAdapter(self.root)
        self.assertEqual(adapter.read_session("header-uuid").session_id, "header-uuid")
        self.assertIsNone(adapter.read_session("scout"))
        self.assertIsNone(adapter.read_session("absent"))

    def test_exact_session_lookup_is_not_limited_to_recent_64_sources(self):
        for index in range(65):
            self.write_session(canonical_records(f"session-{index}", tokens=index + 1),
                               f"project/file-{index}.jsonl", mtime=100 + index)
        adapter = PiAdapter(self.root)
        oldest = adapter.read_session("session-0")
        self.assertEqual(oldest.session_id, "session-0")
        self.assertEqual(oldest.total_tokens, 1)
        self.write_session(canonical_records("session-0", tokens=999),
                           "project/replacement.jsonl", mtime=1000)
        self.assertEqual(adapter.read_session("session-0").total_tokens, 1)

    def test_pi_and_omp_native_sources_are_isolated_in_shared_root(self):
        self.write_session(canonical_records(), "pi.jsonl", mtime=100)
        omp_response = assistant(tokens=80, model="omp-test-model", duration=5000, ttft=1000,
                                 completedAt=(EPOCH + 7) * 1000)
        self.write_session([header("omp-main"),
                            entry("model_change", "model", model="openai/omp-test-model"),
                            user(parent="model"), omp_response], "omp.jsonl", mtime=200)
        pi = PiAdapter(self.root)
        omp = OMPAdapter(self.root)
        self.assertEqual([(span.session_id, span.tokens) for span in pi.collect()], [("pi-main", 120)])
        self.assertEqual([item.session_id for item in pi.collect_sessions()], ["pi-main"])
        self.assertEqual([(span.session_id, span.tokens) for span in omp.collect()], [("omp-main", 80)])
        self.assertEqual([item.session_id for item in omp.collect_sessions()], ["omp-main"])
        self.assertTrue(omp.collect()[0].is_valid)
        self.assertFalse(pi.collect()[0].is_valid)

    def test_omp_session_init_workers_are_isolated_from_pi_in_shared_root(self):
        self.write_session(canonical_records(), "pi.jsonl", mtime=100)
        response = assistant(tokens=80, duration=5000, ttft=1000,
                             completedAt=(EPOCH + 7) * 1000)
        response["message"].pop("provider")
        response["message"].pop("model")
        self.write_session([
            header("omp-worker"),
            entry("session_init", "init", resolvedModel="openai/seed-model",
                  task="synthetic prompt", tools=[]),
            user(parent="init"), response,
        ], "nested/omp-worker.jsonl", mtime=200)
        pi = PiAdapter(self.root)
        omp = OMPAdapter(self.root)
        self.assertTrue(pi.detect())
        self.assertTrue(omp.detect())
        self.assertEqual([item.session_id for item in pi.collect_sessions()], ["pi-main"])
        self.assertEqual([item.session_id for item in omp.collect_sessions()], ["omp-worker"])
        span = omp.collect()[0]
        self.assertEqual((span.model, span.tokens, span.is_valid), ("openai/seed-model", 80, True))
        self.assertIsNone(PiAdapter(self.root).read_session("omp-worker"))
        self.assertEqual(OMPAdapter(self.root).read_session("omp-worker").total_tokens, 80)

    def test_registry_detection_counts_excluded_pi_completion_as_activity(self):
        self.write_session(canonical_records())
        old = -40 * 86400
        self.write_session([
            header("old-omp", old - 10),
            entry("model_change", "omp-model", model="openai/old-model"),
            user("old-user", old - 6, "omp-model"),
            assistant("old-response", old, parent="old-user", request_seconds=old - 5,
                      duration=5000, ttft=1000, completedAt=(EPOCH + old) * 1000),
        ], "old-omp.jsonl")
        adapter = get_adapter("pi", root=self.root)
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.collect()[0].tokens, 120)
        with patch("tokenmon.adapters.time.time", return_value=EPOCH + 100):
            detected = detect_available_adapters(root=self.root)
        self.assertEqual([item.name for item in detected], ["pi"])

    def test_registry_detection_counts_native_user_only_pi_activity(self):
        self.write_session([header(),
                            entry("model_change", "model", provider="openai", modelId="pi-test-model"),
                            user(seconds=20, parent="model")])
        adapter = get_adapter("pi", root=self.root)
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.collect(), [])
        self.assertEqual(adapter.last_generation_timestamp(), EPOCH + 20)
        with patch("tokenmon.adapters.time.time", return_value=EPOCH + 100):
            detected = detect_available_adapters(root=self.root)
        self.assertIn("pi", [item.name for item in detected])


if __name__ == "__main__":
    unittest.main()
