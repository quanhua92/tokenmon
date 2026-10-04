"""Public OMP adapter regressions using only synthetic JSONL journals."""

import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tokenmon.adapters.omp import OMPAdapter


EPOCH = datetime(2026, 10, 3, 10, tzinfo=timezone.utc).timestamp()


def iso(seconds=0):
    return datetime.fromtimestamp(EPOCH + seconds, timezone.utc).isoformat()


def header(session_id="main-id", seconds=0, **fields):
    return {"type": "session", "version": 3, "id": session_id,
            "timestamp": iso(seconds), "cwd": "/synthetic/project", **fields}


def entry(kind, entry_id, seconds=0, parent=None, **fields):
    return {"type": kind, "id": entry_id, "parentId": parent,
            "timestamp": iso(seconds), **fields}


def user(entry_id="u-1", seconds=1, parent=None, **fields):
    return entry("message", entry_id, seconds, parent,
                 message={"role": "user", "timestamp": (EPOCH + seconds) * 1000,
                          "content": "synthetic prompt", **fields})


def assistant(entry_id="a-1", seconds=2, tokens=120, parent="u-1", **fields):
    message = {"role": "assistant", "provider": "openai", "model": "test-model",
               "api": "openai-responses", "responseId": "response-" + entry_id,
               "timestamp": (EPOCH + seconds) * 1000,
               "completedAt": (EPOCH + seconds + 5) * 1000,
               "duration": 5000, "ttft": 1000, "stopReason": "stop",
               "content": [{"type": "text", "text": "synthetic answer"}],
               "usage": {"output": tokens, "input": 900, "cacheRead": 800,
                         "cacheWrite": 700, "totalTokens": 2400}, **fields}
    return entry("message", entry_id, seconds + 5, parent, message=message)


def canonical_records(session_id="main-id", tokens=120):
    return [
        {"type": "title", "v": 1, "title": "synthetic title", "pad": " " * 30},
        header(session_id),
        entry("model_change", "model", model="openai/test-model"),
        entry("thinking_level_change", "effort", parent="model",
              thinkingLevel="high", configured="auto"),
        entry("service_tier_change", "tier", parent="effort",
              serviceTier={"openai": "priority"}),
        user(parent="tier"),
        assistant(tokens=tokens, content=[
            {"type": "text", "text": "synthetic answer"},
            {"type": "thinking", "thinking": "synthetic reasoning"},
            {"type": "toolCall", "id": "tool-1", "name": "synthetic_tool",
             "arguments": {"command": "never executed"}},
        ]),
        entry("message", "result", 8, "a-1", message={
            "role": "toolResult", "toolCallId": "tool-1", "toolName": "synthetic_tool",
            "timestamp": (EPOCH + 8) * 1000,
            "content": [{"type": "text", "text": "synthetic tool output"}],
            "details": {"usage": {"output": 999999}},
        }),
    ]


class TestOMPAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_session(self, records, relative="project/main.jsonl", mtime=None):
        path = self.root / "sessions" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write_records(path, records)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def write_records(self, path, records, mode="w"):
        with path.open(mode, encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def single_span(self, response):
        self.write_session([header(), user(), response])
        spans = OMPAdapter(self.root).collect()
        self.assertEqual(len(spans), 1)
        return spans[0]

    def test_main_and_nested_worker_have_independent_spans_and_timelines(self):
        self.write_session(canonical_records(), mtime=100)
        self.write_session(canonical_records("worker-id", 80),
                           "project/main-artifacts/scout.jsonl", mtime=200)
        adapter = OMPAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(adapter.name, "omp")
        self.assertEqual({span.session_id for span in spans}, {"main-id", "worker-id"})
        self.assertEqual(sum(span.tokens for span in spans if span.is_valid), 200)
        self.assertEqual(sum(span.duration for span in spans if span.is_valid), 8)
        self.assertEqual(sum(span.tokens for span in spans) / sum(span.duration for span in spans), 25)
        self.assertEqual({span.session_id: span.tps for span in spans},
                         {"main-id": 30, "worker-id": 20})
        for span in spans:
            self.assertTrue(span.is_valid)
            self.assertEqual(span.agent, "omp")
            self.assertEqual(span.model, "openai/test-model")
            self.assertEqual(span.turn_id, "a-1")
            self.assertEqual(span.started_at, EPOCH + 3)
            self.assertEqual(span.ended_at, EPOCH + 7)
            self.assertEqual(span.timing_source, "omp-ttft")
            self.assertEqual(span.reasoning_effort, "high")
            self.assertEqual(span.service_tier, "priority")
            self.assertEqual(span.speed_mode, "fast")
            self.assertIsNone(span.speed)
        timelines = adapter.collect_sessions()
        self.assertEqual(len(timelines), 2)
        for timeline in timelines:
            self.assertEqual(timeline.cwd, "/synthetic/project")
            self.assertEqual(timeline.model, "openai/test-model")
            self.assertEqual(timeline.user_messages, 1)
            self.assertEqual(timeline.assistant_messages, 1)
            self.assertEqual(timeline.tool_calls, 1)
            self.assertEqual(sum(event.kind == "reasoning" for event in timeline.events), 1)
            self.assertEqual(sum(event.kind == "tool_output" for event in timeline.events), 1)
            self.assertEqual(timeline.total_tokens, 120 if timeline.session_id == "main-id" else 80)
            self.assertEqual(timeline.created_at, EPOCH + 1)
            self.assertEqual(timeline.updated_at, EPOCH + 8)
            self.assertEqual(timeline.reasoning_effort, "high")
            self.assertEqual(timeline.service_tier, "priority")
            self.assertEqual(timeline.speed_mode, "fast")
            self.assertIsNone(timeline.speed)
            self.assertEqual([event.timestamp for event in timeline.events],
                             sorted(event.timestamp for event in timeline.events))
            self.assertEqual({event.kind for event in timeline.events},
                             {"user_message", "assistant_message", "reasoning", "tool_call", "tool_output"})
            for event in timeline.events:
                self.assertEqual(event.turn_id, "u-1")
                if event.kind in {"reasoning", "tool_call"}:
                    self.assertIsNone(event.tokens)
                if event.kind in {"assistant_message", "reasoning", "tool_call"}:
                    self.assertEqual(event.reasoning_effort, "high")
                    self.assertEqual(event.service_tier, "priority")
                    self.assertEqual(event.speed_mode, "fast")
                if event.kind in {"tool_call", "tool_output"}:
                    self.assertIn("synthetic_tool", event.summary)
            self.assertEqual(adapter.read_session(timeline.session_id).session_id, timeline.session_id)

    def test_stable_entry_and_response_dedup_use_final_payload(self):
        first = assistant(tokens=2, content=[{"type": "text", "text": "partial"}])
        same_entry = assistant(tokens=30, content=[{"type": "text", "text": "updated"}])
        final = assistant("new-entry", tokens=120, responseId="response-a-1", content=[
            {"type": "text", "text": "final"},
            {"type": "thinking", "thinking": "reasoning"},
            {"type": "toolCall", "id": "tool-1", "name": "synthetic_tool"},
        ])
        self.write_session([header(), user(), first, same_entry, final])
        adapter = OMPAdapter(self.root)
        spans = adapter.collect()
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].tokens, 120)
        self.assertEqual(spans[0].turn_id, "a-1")
        self.assertEqual(timeline.total_tokens, 120)
        self.assertEqual(timeline.assistant_messages, 1)
        self.assertEqual(timeline.tool_calls, 1)
        self.assertEqual(sum(event.kind == "reasoning" for event in timeline.events), 1)

    def test_distinct_response_ids_and_providers_are_not_deduplicated(self):
        responses = [assistant("a-1"), assistant("a-2"),
                     assistant("a-3", provider="other", responseId="response-a-1")]
        self.write_session([header(), user(), *responses])
        adapter = OMPAdapter(self.root)
        self.assertEqual(len(adapter.collect()), 3)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.assistant_messages, 3)
        self.assertEqual(timeline.total_tokens, 360)

    def test_idless_entries_use_physical_line_identity(self):
        response = assistant()
        response.pop("id")
        response["message"].pop("responseId")
        path = self.write_session([header(), user(), response, copy.deepcopy(response)])
        adapter = OMPAdapter(self.root)
        before = adapter.collect_sessions()[0]
        self.assertEqual(len(adapter.collect()), 2)
        assistants = [event for event in before.events if event.kind == "assistant_message"]
        self.assertNotEqual(assistants[0].event_id, assistants[1].event_id)
        records = [header(), user(), response, copy.deepcopy(response)]
        records[2]["message"]["usage"]["output"] = 130
        self.write_records(path, records)
        after = adapter.read_session("main-id")
        self.assertEqual([event.event_id for event in before.events],
                         [event.event_id for event in after.events])
        self.assertEqual(after.total_tokens, 250)

    def test_fork_ownership_uses_request_start_and_preserves_settings(self):
        parent = canonical_records()
        self.write_session(parent)
        copied = assistant("copied", seconds=8, tokens=500)
        copied["timestamp"] = iso(20)
        child = [header("worker-id", 10, parentSession="main-id"),
                 entry("model_change", "model", model="openai/test-model"),
                 entry("thinking_level_change", "effort", parent="model", thinkingLevel="high"),
                 entry("service_tier_change", "tier", parent="effort", serviceTier={"openai": "priority"}),
                 user("inherited-user", 9, "tier"), copied,
                 entry("message", "copied-result", 15, message={
                     "role": "toolResult", "timestamp": (EPOCH + 9) * 1000,
                     "toolName": "copied_tool", "content": "copied"}),
                 user("child-user", 10, "tier"),
                 assistant("child-a", 10, 80, "child-user")]
        self.write_session(child, "project/main-artifacts/scout.jsonl")
        adapter = OMPAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(sum(span.tokens for span in spans), 200)
        worker_spans = [span for span in spans if span.session_id == "worker-id"]
        self.assertEqual(len(worker_spans), 1)
        self.assertEqual(worker_spans[0].turn_id, "child-a")
        self.assertEqual(worker_spans[0].reasoning_effort, "high")
        self.assertEqual(worker_spans[0].service_tier, "priority")
        timeline = next(item for item in adapter.collect_sessions() if item.session_id == "worker-id")
        self.assertEqual(timeline.total_tokens, 80)
        self.assertEqual(timeline.user_messages, 1)
        self.assertEqual(timeline.assistant_messages, 1)
        self.assertEqual(timeline.created_at, EPOCH + 10)
        self.assertFalse(any(event.event_id.startswith("entry:copied") for event in timeline.events))

    def test_fork_recovers_request_start_before_attributing_copied_output(self):
        response = assistant("copied", seconds=5, duration=15000,
                             completedAt=(EPOCH + 20) * 1000)
        response["message"].pop("timestamp")
        response["timestamp"] = iso(20)
        self.write_session([header("parent"), response])
        self.write_session([header("child", 10, parentSession="parent"), response],
                           "project/child.jsonl")
        adapter = OMPAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual([(span.session_id, span.tokens, span.duration) for span in spans],
                         [("parent", 120, 14)])
        child = next(item for item in adapter.collect_sessions() if item.session_id == "child")
        self.assertEqual(child.total_tokens, 0)
        self.assertEqual(child.assistant_messages, 0)

    def test_fork_requires_request_evidence_and_includes_creation_boundary(self):
        for duration, expected_tokens in ((None, 0), (10000, 120), (15000, 0)):
            with self.subTest(duration=duration):
                response = assistant("response", completedAt=(EPOCH + 20) * 1000,
                                     duration=duration)
                response["message"].pop("timestamp")
                response["timestamp"] = iso(20)
                self.write_session([header("child", 10, parentSession="parent"), response])
                adapter = OMPAdapter(self.root)
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.total_tokens, expected_tokens)
                self.assertEqual(sum(span.tokens for span in adapter.collect()), expected_tokens)

    def test_fork_without_valid_creation_evidence_is_skipped(self):
        for timestamp in (None, "invalid", "2026-10-03T10:00:00"):
            with self.subTest(timestamp=timestamp):
                self.write_session([header("fork", parentSession="parent", timestamp=timestamp), user(), assistant()])
                adapter = OMPAdapter(self.root)
                with self.assertLogs("tokenmon.adapters.omp", level="WARNING"):
                    self.assertEqual(adapter.collect(), [])
                self.assertEqual(adapter.collect_sessions(), [])

    def test_missing_invalid_timing_is_explicitly_excluded(self):
        cases = [
            ("ttft", None), ("duration", None), ("ttft", True), ("duration", False),
            ("ttft", float("nan")), ("duration", float("inf")),
            ("ttft", -1), ("ttft", 5000), ("ttft", 6000),
            ("duration", 0), ("duration", -1), ("duration", "5000"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                response = assistant()
                if value is None:
                    response["message"].pop(field)
                else:
                    response["message"][field] = value
                span = self.single_span(response)
                self.assertFalse(span.is_valid)
                self.assertEqual(span.note, "unconfirmed_generation_timing")
                self.assertEqual(span.timing_source, "omp-unconfirmed")
                self.assertIsNone(span.tps)
                self.assertEqual(span.ended_at, EPOCH + 7)
                timeline = OMPAdapter(self.root).collect_sessions()[0]
                self.assertEqual(timeline.total_tokens, 120)
                self.assertEqual(timeline.updated_at, EPOCH + 7)

    def test_missing_completion_falls_back_to_request_plus_duration(self):
        response = assistant()
        response["message"].pop("completedAt")
        response["timestamp"] = iso(100)
        span = self.single_span(response)
        self.assertTrue(span.is_valid)
        self.assertEqual(span.started_at, EPOCH + 3)
        self.assertEqual(span.ended_at, EPOCH + 7)
        self.assertEqual(OMPAdapter(self.root).collect_sessions()[0].updated_at, EPOCH + 7)

    def test_invalid_completed_at_can_use_valid_request_duration(self):
        for value in (True, float("inf"), "invalid"):
            with self.subTest(value=value):
                span = self.single_span(assistant(completedAt=value))
                self.assertTrue(span.is_valid)
                self.assertEqual(span.ended_at, EPOCH + 7)
                self.assertEqual(span.timing_source, "omp-ttft")

    def test_completion_without_request_start_is_valid_when_completed_at_recorded(self):
        response = assistant(completedAt=(EPOCH + 20) * 1000)
        response["message"].pop("timestamp")
        span = self.single_span(response)
        self.assertTrue(span.is_valid)
        self.assertEqual(span.started_at, EPOCH + 16)
        self.assertEqual(span.ended_at, EPOCH + 20)

    def test_no_completion_evidence_does_not_borrow_prompt_or_entry_time(self):
        response = assistant()
        response["message"].pop("completedAt")
        response["message"]["timestamp"] = True
        response["timestamp"] = iso(30)
        span = self.single_span(response)
        self.assertFalse(span.is_valid)
        self.assertEqual(span.note, "unconfirmed_generation_timing")
        self.assertEqual(span.timing_source, "omp-unconfirmed")
        self.assertEqual(span.ended_at, EPOCH + 30)

    def test_unconfirmed_output_boundary_retains_activity_without_throughput(self):
        cases = [[], [{"type": "text", "text": ""}],
                 [{"type": "thinking", "thinking": ""}],
                 [{"type": "redactedThinking", "data": "redacted"}],
                 [{"type": "toolCall", "id": "t", "name": "synthetic_tool"},
                  {"type": "text", "text": "later text"}],
                 [{"type": "unknown"}, {"type": "text", "text": "later text"}]]
        for content in cases:
            with self.subTest(content=content):
                span = self.single_span(assistant(content=content))
                self.assertFalse(span.is_valid)
                self.assertEqual(span.note, "unconfirmed_output_boundary")
                self.assertEqual(span.timing_source, "omp-ttft")
                self.assertEqual(span.duration, 4)
                timeline = OMPAdapter(self.root).collect_sessions()[0]
                self.assertEqual(timeline.assistant_messages, 1)
                self.assertEqual(timeline.total_tokens, 120)

    def test_substantive_thinking_is_a_confirmed_output_boundary(self):
        span = self.single_span(assistant(content=[{"type": "thinking", "thinking": "synthetic reasoning"}]))
        self.assertTrue(span.is_valid)
        self.assertEqual(span.timing_source, "omp-ttft")

    def test_aborted_and_error_responses_are_incomplete(self):
        for reason in ("aborted", "error"):
            with self.subTest(reason=reason):
                span = self.single_span(assistant(stopReason=reason))
                self.assertFalse(span.is_valid)
                self.assertEqual(span.note, "incomplete_response")
                self.assertEqual(span.timing_source, "omp-ttft")
                self.assertEqual(span.ended_at, EPOCH + 7)
                self.assertIsNone(span.tps)
                self.assertEqual(OMPAdapter(self.root).collect_sessions()[0].total_tokens, 120)

    def test_shared_minimum_duration_token_and_tps_validation(self):
        for duration, tokens, valid, note in [
            (2000, 120, True, None), (1500, 120, False, "duration_under_1s"),
            (5000, 0, False, "zero_tokens"), (2000, 401, False, "unconfirmed_boundary_tps"),
            (2000, 400, True, None),
        ]:
            with self.subTest(duration=duration, tokens=tokens):
                response = assistant(tokens=tokens, duration=duration,
                                     completedAt=(EPOCH + 2) * 1000 + duration)
                span = self.single_span(response)
                self.assertEqual(span.is_valid, valid)
                self.assertEqual(span.note, note)
                self.assertEqual(span.timing_source, "omp-ttft")
                self.assertEqual(span.duration, (duration - 1000) / 1000)

    def test_bad_output_counts_are_zero_not_input_cache_or_total(self):
        for usage in ({}, {"output": True}, {"output": -1},
                      {"output": 1.5}, {"output": "120"}, {"output": float("nan")},
                      {"input": 900, "cacheRead": 800, "totalTokens": 2400}):
            with self.subTest(usage=usage):
                span = self.single_span(assistant(usage=usage))
                self.assertEqual(span.tokens, 0)
                self.assertFalse(span.is_valid)
                self.assertEqual(span.note, "zero_tokens")
                self.assertEqual(OMPAdapter(self.root).collect_sessions()[0].total_tokens, 0)

    def test_span_and_session_cutoffs_use_completion_equality_and_full_history(self):
        path = self.write_session(canonical_records(), mtime=100)
        adapter = OMPAdapter(self.root)
        expected = adapter.collect()[0]
        for cutoff in (EPOCH + 4, expected.ended_at):
            with self.subTest(cutoff=cutoff):
                self.assertEqual(adapter.collect(min_timestamp=cutoff), [expected])
                self.assertEqual(adapter.collect(min_timestamp=cutoff)[0].duration, 4)
        self.assertEqual(adapter.collect(min_timestamp=expected.ended_at + 0.001), [])
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(adapter.collect_sessions(min_timestamp=timeline.updated_at), [timeline])
        self.assertEqual(adapter.collect_sessions(min_timestamp=timeline.updated_at + 0.001), [])
        retained = adapter.collect_sessions(min_timestamp=EPOCH + 7)[0]
        self.assertEqual(retained.events, timeline.events)
        self.assertEqual(retained.events[0].timestamp, EPOCH + 1)
        os.utime(path, (EPOCH + 1000, EPOCH + 1000))
        self.assertEqual(adapter.collect(min_timestamp=EPOCH + 100), [])
        self.assertEqual(adapter.collect_sessions(min_timestamp=EPOCH + 100), [])

    def test_parent_linked_branches_preserve_historical_settings_and_turns(self):
        left = assistant("left-a", 2, parent="left-user")
        right = assistant("right-a", 10, parent="right-user")
        for response in (left, right):
            response["message"].pop("provider")
            response["message"].pop("model")
            response["message"]["content"].extend([
                {"type": "thinking", "thinking": "synthetic reasoning"},
                {"type": "toolCall", "id": "tool", "name": "synthetic_tool"}])
        self.write_session([
            header(), entry("model_change", "base", model="openai/left-model"),
            entry("thinking_level_change", "left-effort", parent="base", thinkingLevel="high"),
            entry("service_tier_change", "left-tier", parent="left-effort", serviceTier={"openai": "priority"}),
            user("left-user", 1, "left-tier"),
            entry("model_change", "right-model", parent="base", model="google/right-model"),
            entry("thinking_level_change", "right-effort", parent="right-model", thinkingLevel="low"),
            entry("service_tier_change", "right-tier", parent="right-effort", serviceTier={"google": "standard"}),
            user("right-user", 9, "right-tier"), left, right,
        ])
        adapter = OMPAdapter(self.root)
        spans = {span.turn_id: span for span in adapter.collect()}
        self.assertEqual((spans["left-a"].model, spans["left-a"].reasoning_effort, spans["left-a"].service_tier),
                         ("openai/left-model", "high", "priority"))
        self.assertEqual((spans["right-a"].model, spans["right-a"].reasoning_effort, spans["right-a"].service_tier),
                         ("google/right-model", "low", "standard"))
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.model, "google/right-model")
        self.assertEqual(timeline.reasoning_effort, "low")
        self.assertEqual(timeline.service_tier, "standard")
        for event in timeline.events:
            if event.event_id.startswith("entry:left-a:"):
                self.assertEqual(event.turn_id, "left-user")
                self.assertEqual(event.reasoning_effort, "high")
                self.assertEqual(event.service_tier, "priority")
            elif event.event_id.startswith("entry:right-a:"):
                self.assertEqual(event.turn_id, "right-user")
                self.assertEqual(event.reasoning_effort, "low")
                self.assertEqual(event.service_tier, "standard")

    def test_explicit_null_settings_reset_effort_and_tier(self):
        records = canonical_records()
        records.extend([
            entry("thinking_level_change", "reset-effort", 9, "result", thinkingLevel=None),
            entry("service_tier_change", "reset-tier", 9, "reset-effort", serviceTier=None),
            user("new-user", 10, "reset-tier"), assistant("new-a", 11, parent="new-user"),
        ])
        self.write_session(records)
        adapter = OMPAdapter(self.root)
        spans = {span.turn_id: span for span in adapter.collect()}
        self.assertEqual(spans["a-1"].reasoning_effort, "high")
        self.assertEqual(spans["a-1"].service_tier, "priority")
        self.assertIsNone(spans["new-a"].reasoning_effort)
        self.assertIsNone(spans["new-a"].service_tier)
        timeline = adapter.collect_sessions()[0]
        self.assertIsNone(timeline.reasoning_effort)
        self.assertIsNone(timeline.service_tier)
        self.assertIsNone(timeline.speed_mode)

    def test_explicit_metadata_precedence_and_latest_settings(self):
        records = canonical_records()
        records[6]["message"].update(effort="medium", service_tier="standard", speed="slow")
        records[6]["message"]["usage"].update(effort="max", service_tier="priority", speed="fast")
        records[6].update(effort="low", service_tier="default", speed="normal")
        path = self.write_session(records)
        adapter = OMPAdapter(self.root)
        span = adapter.collect()[0]
        self.assertEqual((span.reasoning_effort, span.service_tier, span.speed), ("max", "priority", "fast"))
        timeline = adapter.collect_sessions()[0]
        self.assertEqual((timeline.reasoning_effort, timeline.service_tier, timeline.speed),
                         ("max", "priority", "fast"))
        self.write_records(path, [entry("thinking_level_change", "later-effort", 9, thinkingLevel="low"),
                                  entry("service_tier_change", "later-tier", 9, serviceTier=None)], "a")
        latest = adapter.read_session("main-id")
        self.assertEqual(latest.reasoning_effort, "low")
        self.assertIsNone(latest.service_tier)
        self.assertEqual(latest.speed, "fast")

    def test_provider_routing_resolves_only_explicit_tier_families(self):
        routes = [
            ("openai", "openai-responses", "priority"),
            ("openai-codex", "openai-codex-responses", "priority"),
            ("anthropic", "anthropic-messages", "standard"),
            ("gateway", "anthropic-messages", "standard"),
            ("google", "google-generative-ai", "flex"),
            ("google-vertex", "google-vertex", "flex"),
            ("gateway", "aggregated", None), (None, None, None),
        ]
        for provider, api, tier in routes:
            with self.subTest(provider=provider, api=api):
                response = assistant(provider=provider, api=api, model="openai-looking-fast-high")
                self.write_session([header(), entry("service_tier_change", "tier",
                    serviceTier={"openai": "priority", "anthropic": "standard", "google": "flex"}), response])
                adapter = OMPAdapter(self.root)
                span = adapter.collect()[0]
                self.assertEqual(span.service_tier, tier)
                self.assertIsNone(span.reasoning_effort)
                self.assertIsNone(span.speed)
                self.assertEqual(adapter.collect_sessions()[0].service_tier, tier)

    def test_model_switch_recomputes_tier_and_map_replacement_drops_old_family(self):
        records = canonical_records()
        records.extend([entry("model_change", "google", 9, model="google/new-model")])
        path = self.write_session(records)
        adapter = OMPAdapter(self.root)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.model, "google/new-model")
        self.assertIsNone(timeline.service_tier)
        self.write_records(path, [entry("service_tier_change", "only-google", 10,
                                       serviceTier={"google": "flex"}),
                                  entry("model_change", "openai-again", 11,
                                        model="openai/test-model")], "a")
        self.assertIsNone(adapter.read_session("main-id").service_tier)

    def test_message_model_overrides_state_without_duplicate_provider_prefix(self):
        for model, expected in (("own-model", "openai/own-model"),
                                ("openai/own-model", "openai/own-model")):
            with self.subTest(model=model):
                self.write_session([header(), entry("model_change", "model", model="google/inherited"),
                                    user(parent="model"), assistant(model=model)])
                self.assertEqual(OMPAdapter(self.root).collect()[0].model, expected)
        span = self.single_span(assistant(provider=None, model="recorded-selector"))
        self.assertEqual(span.model, "recorded-selector")
        span = self.single_span(assistant(provider=None, model=None))
        self.assertEqual(span.model, "unknown")

    def test_older_scalar_service_tier_is_recorded_directly(self):
        self.write_session([
            header(), entry("service_tier_change", "legacy-tier", serviceTier="priority"),
            user(parent="legacy-tier"), assistant(provider="gateway"),
        ])
        span = OMPAdapter(self.root).collect()[0]
        self.assertEqual(span.service_tier, "priority")
        self.assertEqual(span.speed_mode, "fast")
        self.assertIsNone(span.speed)

    def test_session_init_seeds_model_without_duplicate_prompt(self):
        response = assistant()
        response["message"].pop("provider")
        response["message"].pop("model")
        self.write_session([header("worker-id"),
                            entry("session_init", "init", resolvedModel="openai/seed-model",
                                  task="synthetic prompt", tools=[]), user(parent="init"), response])
        adapter = OMPAdapter(self.root)
        self.assertEqual(adapter.collect()[0].model, "openai/seed-model")
        self.assertEqual(adapter.collect_sessions()[0].user_messages, 1)

    def test_utility_model_usage_is_not_another_generation(self):
        self.write_session([header(), user(), assistant(), entry("model_usage", "utility", 20,
            provider="openai", model="utility", purpose="title", usage={"output": 999999},
            duration=5000, ttft=1000)])
        adapter = OMPAdapter(self.root)
        self.assertEqual(len(adapter.collect()), 1)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.total_tokens, 120)
        self.assertEqual(timeline.updated_at, EPOCH + 7)

    def test_header_and_user_only_sources_have_no_fabricated_spans(self):
        self.write_session([header("empty", cwd="")], "empty.jsonl")
        self.write_session([header("user-only"), user(seconds=20)], "user.jsonl")
        adapter = OMPAdapter(self.root)
        self.assertEqual(adapter.collect(), [])
        timelines = adapter.collect_sessions()
        self.assertEqual([item.session_id for item in timelines], ["user-only", "empty"])
        empty = timelines[1]
        self.assertEqual(empty.events, [])
        self.assertEqual((empty.created_at, empty.updated_at), (EPOCH, EPOCH))
        self.assertIsNone(empty.cwd)
        self.assertEqual(timelines[0].user_messages, 1)
        self.assertEqual(timelines[0].created_at, EPOCH + 20)
        self.assertEqual(timelines[0].updated_at, EPOCH + 20)
        self.assertEqual(adapter.collect_sessions(min_timestamp=EPOCH), timelines)
        self.assertEqual([item.session_id for item in adapter.collect_sessions(min_timestamp=EPOCH + 0.001)],
                         ["user-only"])

    def test_timestamp_fallbacks_summary_and_tool_block_identity(self):
        prompt = user(content=[{"type": "text", "text": "x" * 80}])
        prompt["message"]["timestamp"] = False
        response = assistant(content=[{"type": "text", "text": "answer"},
                                      {"type": "redactedThinking", "data": "opaque"},
                                      {"type": "toolCall", "name": "synthetic_tool"}])
        result = entry("message", "result", 9, "a-1", message={
            "role": "toolResult", "toolName": "synthetic_tool", "content": "result",
            "timestamp": float("nan")})
        self.write_session([header(), prompt, response, result])
        timeline = OMPAdapter(self.root).collect_sessions()[0]
        self.assertEqual(timeline.events[0].timestamp, EPOCH + 1)
        self.assertIn("x" * 60, timeline.events[0].summary)
        self.assertNotIn("x" * 61, timeline.events[0].summary)
        self.assertEqual(timeline.events[-1].timestamp, EPOCH + 9)

    def test_unrenderable_numeric_times_do_not_hide_later_generation(self):
        self.write_session([
            header(), user(),
            assistant("bad-time", timestamp=1e308, completedAt=1e308),
            assistant("later", 20, 80),
        ])
        adapter = OMPAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual([span.tokens for span in spans if span.is_valid], [80])
        self.assertEqual(spans[0].note, "unconfirmed_generation_timing")
        timeline = adapter.collect_sessions()[0]
        self.assertEqual([event.timestamp for event in timeline.events],
                         [EPOCH + 1, EPOCH + 7, EPOCH + 25])
        self.assertEqual(timeline.assistant_messages, 2)

    def test_malformed_values_and_truncated_tail_do_not_hide_later_good_records(self):
        junk = [[], None, "text", 42, True,
                {"type": []}, {"type": "unknown", "id": "unknown"},
                {"type": "message", "message": None},
                {"type": "message", "message": []},
                entry("message", "bad-role", message={"role": []}),
                entry("model_change", "bad-model", model={}),
                entry("thinking_level_change", "bad-effort", thinkingLevel=[]),
                entry("service_tier_change", "bad-tier", serviceTier=[]),
                assistant("bad-usage", usage=None), assistant("bad-usage-list", usage=[])]
        path = self.write_session([header(), *junk, user(), assistant()])
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{not json}\n{"type":"message"')
        self.write_session([{"type": "session", "id": None}, [], "broken"], "broken.jsonl")
        adapter = OMPAdapter(self.root)
        self.assertEqual([span.tokens for span in adapter.collect()], [120])
        timelines = adapter.collect_sessions()
        self.assertEqual(len(timelines), 1)
        self.assertEqual(timelines[0].assistant_messages, 1)
        self.assertEqual(timelines[0].total_tokens, 120)
        # A new valid line after a malformed line is independently recoverable.
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n" + json.dumps(assistant("later", 20, 80)) + "\n")
        self.assertEqual([span.tokens for span in adapter.collect()], [120, 80])

    def test_malformed_content_blocks_preserve_later_valid_generation(self):
        for content in (None, {}, "text", [None, [], "bad", {"type": []}],
                        [{"type": "text", "text": []}]):
            with self.subTest(content=content):
                self.write_session([header(), user(), assistant("bad-content", content=content),
                                    assistant("good", 20, 80)])
                adapter = OMPAdapter(self.root)
                valid = [span for span in adapter.collect() if span.is_valid]
                self.assertEqual([span.turn_id for span in valid], ["good"])
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(sum(event.event_id == "entry:good:assistant" for event in timeline.events), 1)

    def test_naive_bad_and_missing_event_times_do_not_block_later_records(self):
        records = [header(timestamp="invalid")]
        for index, timestamp in enumerate(("2026-10-03T10:00:00", "invalid", None)):
            records.append(entry("message", "bad-" + str(index), timestamp=timestamp,
                                 message={"role": "user", "timestamp": True, "content": "bad time"}))
        records.extend([user(), assistant()])
        self.write_session(records)
        adapter = OMPAdapter(self.root)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.user_messages, 1)
        self.assertEqual(timeline.created_at, EPOCH + 1)
        self.assertEqual(len(adapter.collect()), 1)
        self.write_session([header(timestamp="2026-10-03T10:00:00")], "naive-only.jsonl")
        self.assertEqual(len(adapter.collect_sessions()), 1)

    def test_usable_header_can_follow_title_and_unusable_header(self):
        self.write_session([{"type": "title", "title": {"usage": {"output": 999999}}},
                            header("", timestamp="invalid"), header("usable"), user(), assistant()])
        adapter = OMPAdapter(self.root)
        self.assertEqual(adapter.collect()[0].session_id, "usable")
        self.assertEqual(adapter.collect_sessions()[0].session_id, "usable")
        self.write_session([user(), assistant()], "missing-header.jsonl")
        self.assertEqual(len(adapter.collect()), 1)

    def test_replacement_decoding_preserves_usable_records(self):
        path = self.write_session([header(), user(), assistant()])
        with path.open("ab") as stream:
            stream.write(b'{"type":"title","title":"invalid byte \xff"}\n')
        self.assertEqual(OMPAdapter(self.root).collect()[0].tokens, 120)

    def test_detect_empty_missing_and_non_directory_sources(self):
        adapter = OMPAdapter(self.root)
        self.assertFalse(adapter.detect())
        self.assertEqual(adapter.collect(), [])
        self.assertEqual(adapter.collect_sessions(), [])
        sessions = self.root / "sessions"
        sessions.write_text("not a directory", encoding="utf-8")
        self.assertFalse(adapter.detect())
        sessions.unlink()
        sessions.mkdir()
        self.assertTrue(adapter.detect())
        self.assertEqual(adapter.collect(), [])

    def test_nested_limits_are_newest_first_and_nonpositive_limits_are_empty(self):
        self.write_session([header("main"), user(), assistant()], "main.jsonl", mtime=100)
        self.write_session([header("worker-a"), user(), assistant()], "artifacts/a/scout.jsonl", mtime=300)
        self.write_session([header("worker-b"), user(), assistant()], "artifacts/b/scout.jsonl", mtime=200)
        adapter = OMPAdapter(self.root)
        for limit in (0, -1):
            with self.subTest(limit=limit):
                self.assertEqual(adapter.collect(max_sessions=limit), [])
                self.assertEqual(adapter.collect_sessions(max_sessions=limit), [])
        self.assertEqual({span.session_id for span in adapter.collect(max_sessions=2)}, {"worker-a", "worker-b"})
        self.assertEqual({item.session_id for item in adapter.collect_sessions(max_sessions=2)}, {"worker-a", "worker-b"})
        adapter.collect_sessions()
        self.assertEqual(adapter.read_session("worker-a").session_id, "worker-a")
        self.assertEqual(adapter.read_session("worker-b").session_id, "worker-b")

    def test_discovery_ties_are_deterministic_and_bound_precedes_parse(self):
        self.write_session([header("a"), user(), assistant()], "a.jsonl", mtime=100)
        self.write_session([header("b"), user(), assistant()], "b.jsonl", mtime=100)
        adapter = OMPAdapter(self.root)
        self.assertEqual(adapter.collect(max_sessions=1)[0].session_id, "a")
        self.assertEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "a")
        self.write_session(["unusable source"], "newer-broken.jsonl", mtime=200)
        self.assertEqual(adapter.collect(max_sessions=1), [])
        self.assertEqual(adapter.collect_sessions(max_sessions=1), [])

    def test_empty_dotfiles_and_directory_symlinks_are_not_discovered(self):
        self.write_session([header("good"), user(), assistant()], "good.jsonl", mtime=100)
        self.write_session([], "empty.jsonl", mtime=500)
        self.write_session([header("hidden"), user(), assistant()], ".hidden.jsonl", mtime=500)
        external = self.root / "outside"
        external.mkdir()
        self.write_records(external / "external.jsonl", [header("external"), user(), assistant()])
        (self.root / "sessions" / "linked").symlink_to(external, target_is_directory=True)
        adapter = OMPAdapter(self.root)
        self.assertEqual([span.session_id for span in adapter.collect(max_sessions=1)], ["good"])
        self.assertEqual([item.session_id for item in adapter.collect_sessions()], ["good"])

    def test_duplicate_header_id_keeps_newest_readable_source(self):
        self.write_session([header("same"), user(), assistant(tokens=30)], "older.jsonl", mtime=100)
        newest = self.write_session([header("same"), user(), assistant(tokens=80)], "newer.jsonl", mtime=200)
        adapter = OMPAdapter(self.root)
        self.assertEqual([span.tokens for span in adapter.collect()], [80])
        self.assertEqual(len(adapter.collect_sessions()), 1)
        self.assertEqual(adapter.read_session("same").total_tokens, 80)
        original_open = Path.open

        def readable_open(path, *args, **kwargs):
            if path == newest:
                raise PermissionError("synthetic unreadable source")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", readable_open):
            fresh = OMPAdapter(self.root)
            self.assertEqual([span.tokens for span in fresh.collect()], [30])
            self.assertEqual(fresh.collect_sessions()[0].total_tokens, 30)

    def test_pinned_refresh_survives_partial_line_failures_and_newer_sources(self):
        path = self.write_session(canonical_records("worker-id"), "artifacts/scout.jsonl", mtime=100)
        adapter = OMPAdapter(self.root)
        initial = adapter.collect_sessions(max_sessions=1)[0]
        self.write_session([header("newer"), user(), assistant()], "new-main.jsonl", mtime=300)
        self.write_session([header("newer-worker"), user(), assistant()], "other/scout.jsonl", mtime=400)
        self.assertEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "newer-worker")
        final = assistant(tokens=140, completedAt=(EPOCH + 8) * 1000, effort="max")
        self.write_records(path, [final], "a")
        refreshed = adapter.read_session("worker-id")
        self.assertEqual(refreshed.session_id, "worker-id")
        self.assertEqual(refreshed.total_tokens, 140)
        initial_assistant = next(event for event in initial.events if event.kind == "assistant_message")
        refreshed_assistant = next(event for event in refreshed.events if event.kind == "assistant_message")
        self.assertEqual(initial_assistant.event_id, refreshed_assistant.event_id)
        self.assertEqual(refreshed_assistant.reasoning_effort, "max")
        new_user = user("same-time-user", 8)
        encoded = json.dumps(new_user)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded[:20])
        partial = adapter.read_session("worker-id")
        self.assertEqual(partial.user_messages, 1)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded[20:] + "\n")
        complete = adapter.read_session("worker-id")
        self.assertEqual(complete.user_messages, 2)
        self.assertEqual(sum(event.event_id == "entry:same-time-user:user" for event in complete.events), 1)
        self.assertEqual(adapter.read_session("worker-id").events, complete.events)
        hidden = path.with_suffix(".hidden")
        path.rename(hidden)
        self.assertIsNone(adapter.read_session("worker-id"))
        hidden.rename(path)
        self.assertEqual(adapter.read_session("worker-id").total_tokens, 140)
        self.write_records(path, [header("wrong-id"), user(), assistant()])
        self.assertIsNone(adapter.read_session("worker-id"))
        self.write_records(path, canonical_records("worker-id"))
        self.assertEqual(adapter.read_session("worker-id").session_id, "worker-id")

    def test_pinned_unreadable_or_unusable_source_does_not_switch_sessions(self):
        path = self.write_session([header("pinned"), user(), assistant()], mtime=100)
        adapter = OMPAdapter(self.root)
        adapter.collect_sessions()
        self.write_session([header("other"), user(), assistant()], "other.jsonl", mtime=200)
        original_open = Path.open

        def readable_open(source, *args, **kwargs):
            if source == path:
                raise OSError("synthetic temporary failure")
            return original_open(source, *args, **kwargs)

        with patch.object(Path, "open", readable_open):
            self.assertIsNone(adapter.read_session("pinned"))
        self.assertEqual(adapter.read_session("pinned").session_id, "pinned")
        self.write_records(path, [[], "unusable"])
        self.assertIsNone(adapter.read_session("pinned"))
        self.write_records(path, [header("pinned"), user(), assistant()])
        self.assertEqual(adapter.read_session("pinned").session_id, "pinned")

    def test_unknown_pinned_id_uses_initial_discovery_by_header_id(self):
        self.write_session([header("header-uuid"), user(), assistant()], "scout.jsonl")
        adapter = OMPAdapter(self.root)
        self.assertEqual(adapter.read_session("header-uuid").session_id, "header-uuid")
        self.assertIsNone(adapter.read_session("scout"))
        self.assertIsNone(adapter.read_session("absent"))

    def test_exact_session_lookup_is_not_limited_to_recent_64_sources(self):
        for index in range(65):
            self.write_session(canonical_records(f"session-{index}", tokens=index + 1),
                               f"project/file-{index}.jsonl", mtime=100 + index)
        adapter = OMPAdapter(self.root)
        oldest = adapter.read_session("session-0")
        self.assertEqual(oldest.session_id, "session-0")
        self.assertEqual(oldest.total_tokens, 1)
        self.write_session(canonical_records("session-0", tokens=999),
                           "project/replacement.jsonl", mtime=1000)
        self.assertEqual(adapter.read_session("session-0").total_tokens, 1)

    def test_tool_result_keeps_its_own_historical_settings(self):
        records = canonical_records()[:-1]
        records.extend([
            entry("thinking_level_change", "result-effort", 8, "a-1", thinkingLevel="low"),
            entry("service_tier_change", "result-tier", 8, "result-effort",
                  serviceTier={"openai": "standard"}),
            entry("message", "result", 9, "result-tier", message={
                "role": "toolResult", "toolName": "synthetic_tool",
                "timestamp": (EPOCH + 9) * 1000, "content": "synthetic result"}),
        ])
        self.write_session(records)
        timeline = OMPAdapter(self.root).collect_sessions()[0]
        response = next(event for event in timeline.events if event.kind == "assistant_message")
        result = next(event for event in timeline.events if event.kind == "tool_output")
        self.assertEqual((response.reasoning_effort, response.service_tier), ("high", "priority"))
        self.assertEqual((result.reasoning_effort, result.service_tier), ("low", "standard"))
        self.assertEqual(result.turn_id, "u-1")
        self.assertEqual((timeline.reasoning_effort, timeline.service_tier), ("low", "standard"))

    def test_unknown_parent_uses_preceding_state_and_no_prompt_uses_source_turn(self):
        response = assistant(parent="unknown-parent")
        response["message"].pop("provider")
        response["message"].pop("model")
        self.write_session([
            header(), entry("model_change", "model", model="openai/recorded"),
            entry("thinking_level_change", "effort", parent="unknown", thinkingLevel="high"),
            response,
        ])
        adapter = OMPAdapter(self.root)
        span = adapter.collect()[0]
        self.assertEqual(span.model, "openai/recorded")
        self.assertEqual(span.reasoning_effort, "high")
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.events[0].turn_id, "a-1")
        self.assertEqual(timeline.user_messages, 0)

    def test_messages_without_any_timestamp_skip_events_but_keep_later_activity(self):
        bad = assistant("untimed", timestamp=None, completedAt=None, duration=None)
        bad["timestamp"] = None
        self.write_session([
            header(), entry("message", "untimed-user", timestamp=None,
                            message={"role": "user", "content": "no time"}),
            bad, user(), assistant(),
        ])
        adapter = OMPAdapter(self.root)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.user_messages, 1)
        self.assertEqual(timeline.assistant_messages, 1)
        self.assertEqual(timeline.total_tokens, 120)
        self.assertEqual([span.turn_id for span in adapter.collect() if span.is_valid], ["a-1"])

    def test_pinned_missing_path_does_not_rebind_to_duplicate_header(self):
        pinned = self.write_session([header("pinned"), user(), assistant(tokens=30)],
                                    "pinned.jsonl", mtime=100)
        adapter = OMPAdapter(self.root)
        adapter.collect_sessions(max_sessions=1)
        self.write_session([header("pinned"), user(), assistant(tokens=80)],
                           "replacement.jsonl", mtime=200)
        hidden = pinned.with_suffix(".hidden")
        pinned.rename(hidden)
        self.assertIsNone(adapter.read_session("pinned"))
        hidden.rename(pinned)
        self.assertEqual(adapter.read_session("pinned").total_tokens, 30)

    def test_jsonl_outside_sessions_is_not_a_telemetry_source(self):
        self.write_records(self.root / "unrelated.jsonl", [header("outside"), user(), assistant()])
        self.write_session([header("inside"), user(), assistant()])
        adapter = OMPAdapter(self.root)
        self.assertEqual([span.session_id for span in adapter.collect()], ["inside"])
        self.assertEqual([item.session_id for item in adapter.collect_sessions()], ["inside"])


    def test_nonrepresentable_integer_counts_and_timing_do_not_hide_later_response(self):
        huge = 10 ** 400
        cases = [
            ("output", {"usage": {"output": huge}}, "zero_tokens"),
            ("duration", {"duration": huge}, "unconfirmed_generation_timing"),
            ("ttft", {"ttft": huge}, "unconfirmed_generation_timing"),
            ("completion", {"completedAt": huge, "timestamp": huge},
             "unconfirmed_generation_timing"),
        ]
        for label, fields, note in cases:
            with self.subTest(field=label):
                self.write_session([header(), user(), assistant("huge", **fields),
                                    assistant("later-good", 20, 80)])
                adapter = OMPAdapter(self.root)
                spans = {span.turn_id: span for span in adapter.collect()}
                self.assertEqual(set(spans), {"huge", "later-good"})
                self.assertTrue(spans["later-good"].is_valid)
                self.assertEqual(spans["later-good"].tokens, 80)
                self.assertFalse(spans["huge"].is_valid)
                self.assertEqual(spans["huge"].note, note)
                if label == "output":
                    self.assertEqual(spans["huge"].tokens, 0)
                    self.assertEqual(spans["huge"].timing_source, "omp-ttft")
                    self.assertEqual(adapter.collect_sessions()[0].total_tokens, 80)
                else:
                    self.assertEqual(spans["huge"].timing_source, "omp-unconfirmed")
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.assistant_messages, 2)
                self.assertEqual(timeline.updated_at, EPOCH + 25)

    def test_missing_or_malformed_settings_do_not_act_as_explicit_null_resets(self):
        records = canonical_records()
        records.extend([
            entry("thinking_level_change", "missing-effort", 9, "result"),
            entry("service_tier_change", "missing-tier", 9, "missing-effort"),
            entry("thinking_level_change", "malformed-effort", 9, "missing-tier", thinkingLevel=[]),
            entry("service_tier_change", "malformed-tier", 9, "malformed-effort", serviceTier=[]),
            user("later-user", 10, "malformed-tier"),
            assistant("later-a", 11, parent="later-user"),
        ])
        self.write_session(records)
        adapter = OMPAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(len(spans), 2)
        for span in spans:
            self.assertEqual(span.reasoning_effort, "high")
            self.assertEqual(span.service_tier, "priority")
            self.assertEqual(span.speed_mode, "fast")
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.reasoning_effort, "high")
        self.assertEqual(timeline.service_tier, "priority")

    def test_unreadable_and_broken_sources_do_not_hide_readable_good_source(self):
        good = self.write_session([header("good"), user(), assistant(tokens=80)],
                                  "good.jsonl", mtime=100)
        unreadable = self.write_session([header("unreadable"), user(), assistant()],
                                        "unreadable.jsonl", mtime=300)
        broken = self.write_session([[], {"type": "session", "id": []}, "bad"],
                                    "broken.jsonl", mtime=200)
        with broken.open("a", encoding="utf-8") as stream:
            stream.write("{invalid json}\\n")
        original_open = Path.open

        def readable_open(path, *args, **kwargs):
            if path == unreadable:
                raise PermissionError("synthetic unreadable journal")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", readable_open):
            adapter = OMPAdapter(self.root)
            self.assertEqual([span.session_id for span in adapter.collect()], ["good"])
            timelines = adapter.collect_sessions()
            self.assertEqual([timeline.session_id for timeline in timelines], ["good"])
            self.assertEqual(timelines[0].total_tokens, 80)
            self.assertEqual(adapter.read_session("good").session_id, "good")
        self.assertTrue(good.exists())


class TestOMPRoots(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.home = Path(self.temp_dir.name).resolve()
        self.environment = patch.dict(os.environ, {"HOME": str(self.home)}, clear=True)
        self.environment.start()
        self.home_patch = patch.object(Path, "home", return_value=self.home)
        self.home_patch.start()
        self.platform_patch = patch("sys.platform", "linux")
        self.platform_patch.start()

    def tearDown(self):
        self.platform_patch.stop()
        self.home_patch.stop()
        self.environment.stop()
        self.temp_dir.cleanup()

    def test_native_default_and_config_directory_are_home_relative(self):
        self.assertEqual(OMPAdapter().root, self.home / ".omp" / "agent")
        with patch.dict(os.environ, {"PI_CONFIG_DIR": "custom"}):
            self.assertEqual(OMPAdapter().root, self.home / "custom" / "agent")

    def test_fictional_omp_home_does_not_override_native_directory(self):
        with patch.dict(os.environ, {"OMP_HOME": str(self.home / "fictional")}):
            self.assertEqual(OMPAdapter().root, self.home / ".omp" / "agent")

    def test_explicit_root_and_agent_dir_bypass_profiles(self):
        with patch.dict(os.environ, {"PI_CODING_AGENT_DIR": "~/env-agent",
                                     "OMP_PROFILE": "../../invalid", "PI_CONFIG_DIR": "other"}):
            self.assertEqual(OMPAdapter("~/explicit/../chosen").root, self.home / "chosen")
            self.assertEqual(OMPAdapter().root, self.home / "env-agent")
        relative = Path("relative-omp-root")
        self.assertEqual(OMPAdapter(relative).root, relative.resolve())

    def test_blank_agent_dir_does_not_override_native_root(self):
        for value in ("", "   "):
            with self.subTest(value=value), patch.dict(os.environ, {"PI_CODING_AGENT_DIR": value}):
                self.assertEqual(OMPAdapter().root, self.home / ".omp" / "agent")

    def test_omp_profile_precedes_pi_profile_and_blank_selects_default(self):
        with patch.dict(os.environ, {"PI_PROFILE": "legacy"}):
            self.assertEqual(OMPAdapter().root, self.home / ".omp" / "profiles" / "legacy" / "agent")
        with patch.dict(os.environ, {"PI_PROFILE": "legacy", "OMP_PROFILE": "named-1"}):
            self.assertEqual(OMPAdapter().root, self.home / ".omp" / "profiles" / "named-1" / "agent")
        for value in ("", "   ", "default"):
            with self.subTest(value=value), patch.dict(os.environ, {"PI_PROFILE": "legacy", "OMP_PROFILE": value}):
                self.assertEqual(OMPAdapter().root, self.home / ".omp" / "agent")

    def test_valid_profile_names_and_invalid_environment_fallback(self):
        for name in ("a", "team-1", "a_b.c", "a" * 64):
            with self.subTest(name=name), patch.dict(os.environ, {"OMP_PROFILE": name}):
                self.assertEqual(OMPAdapter().root, self.home / ".omp" / "profiles" / name / "agent")
        for name in (".", "..", "../escape", "/absolute", "UPPER", "trailing.", "a" * 65,
                     "con", "prn", "aux", "nul", "com1", "lpt9"):
            with self.subTest(name=name), patch.dict(os.environ, {"OMP_PROFILE": name}):
                with self.assertLogs("tokenmon.adapters.omp", level="WARNING"):
                    self.assertEqual(OMPAdapter().root, self.home / ".omp" / "agent")

    def test_xdg_migration_requires_existing_data_root_not_agent_subdirectory(self):
        xdg = self.home / "xdg"
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg)}):
            self.assertEqual(OMPAdapter().root, self.home / ".omp" / "agent")
            (xdg / "omp").mkdir(parents=True)
            self.assertEqual(OMPAdapter().root, xdg / "omp")
            (xdg / "omp" / "sessions").mkdir()
            self.assertTrue(OMPAdapter().detect())
            with patch.dict(os.environ, {"OMP_PROFILE": "team"}):
                self.assertEqual(OMPAdapter().root, self.home / ".omp" / "profiles" / "team" / "agent")
                profile = xdg / "omp" / "profiles" / "team"
                profile.mkdir(parents=True)
                self.assertEqual(OMPAdapter().root, profile)
                (profile / "sessions").mkdir()
                self.assertTrue(OMPAdapter().detect())

    def test_xdg_file_does_not_count_as_migrated_directory(self):
        xdg = self.home / "xdg"
        xdg.mkdir()
        (xdg / "omp").write_text("not a directory", encoding="utf-8")
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg)}):
            self.assertEqual(OMPAdapter().root, self.home / ".omp" / "agent")

    def test_explicit_root_and_agent_env_precede_existing_xdg_root(self):
        xdg = self.home / "xdg"
        (xdg / "omp").mkdir(parents=True)
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg),
                                     "PI_CODING_AGENT_DIR": str(self.home / "env")}):
            self.assertEqual(OMPAdapter().root, self.home / "env")
            self.assertEqual(OMPAdapter(self.home / "explicit").root, self.home / "explicit")

    def test_macos_honors_xdg_but_other_platform_uses_native_root(self):
        xdg = self.home / "xdg"
        (xdg / "omp").mkdir(parents=True)
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg)}):
            with patch("sys.platform", "darwin"):
                self.assertEqual(OMPAdapter().root, xdg / "omp")
            with patch("sys.platform", "win32"):
                self.assertEqual(OMPAdapter().root, self.home / ".omp" / "agent")

    def test_root_selection_does_not_create_directories(self):
        adapter = OMPAdapter()
        self.assertFalse(adapter.root.exists())
        self.assertFalse(adapter.detect())
        self.assertEqual(adapter.collect(), [])
        self.assertEqual(adapter.collect_sessions(), [])
        self.assertFalse(adapter.root.exists())


if __name__ == "__main__":
    unittest.main()
