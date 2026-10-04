"""Unit tests for tokenmon models, analyzer, and CodexAdapter."""

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from contextlib import closing

from tokenmon.adapters.codex import CodexAdapter
from tokenmon.analyzer import (
    analyze_agent_model_windows,
    analyze_windows,
    filter_by_window,
    summarize_spans,
)
from tokenmon.models import GenerationSpan


class TestModelsAndAnalyzer(unittest.TestCase):
    def test_generation_span_tps_calculation(self):
        span = GenerationSpan(
            agent="codex",
            session_id="s1",
            turn_id="t1",
            model="gpt-5",
            tokens=120,
            started_at=100.0,
            ended_at=102.0,
            timing_source="stream-log",
            is_valid=True,
        )
        self.assertEqual(span.duration, 2.0)
        self.assertAlmostEqual(span.tps, 60.0)

    def test_invalid_span_has_no_tps(self):
        span = GenerationSpan(
            agent="codex",
            session_id="s1",
            turn_id="t1",
            model="gpt-5",
            tokens=120,
            started_at=100.0,
            ended_at=102.0,
            is_valid=False,
            note="unconfirmed_tool_start",
        )
        self.assertIsNone(span.tps)

    def test_weighted_tps_calculation(self):
        # 100 tokens in 1s (100 TPS) + 100 tokens in 9s (11.1 TPS)
        # Weighted TPS should be 200 tokens / 10s = 20.0 TPS, NOT arithmetic mean (55.5)
        s1 = GenerationSpan("codex", "s1", "t1", "m", 100, 10.0, 11.0)
        s2 = GenerationSpan("codex", "s2", "t2", "m", 100, 20.0, 29.0)
        summary = summarize_spans([s1, s2], "all", "m")
        self.assertAlmostEqual(summary.weighted_tps, 20.0)
        self.assertEqual(summary.total_tokens, 200)
        self.assertAlmostEqual(summary.total_duration, 10.0)
        self.assertAlmostEqual(summary.median_tps, 55.555555, places=3)

    def test_agent_model_analysis_does_not_blend_sources(self):
        codex = GenerationSpan("codex", "s1", "t1", "shared", 100, 10.0, 12.0)
        claude = GenerationSpan("claude", "s2", "t2", "shared", 90, 20.0, 23.0)

        by_source = analyze_agent_model_windows([codex, claude], ["all"], now=30.0)

        self.assertEqual(list(by_source), [("claude", "shared"), ("codex", "shared")])
        self.assertEqual(by_source[("claude", "shared")][0].weighted_tps, 30.0)
        self.assertEqual(by_source[("codex", "shared")][0].weighted_tps, 50.0)
        self.assertEqual(analyze_windows([codex, claude], ["all"], now=30.0)["shared"][0].weighted_tps, 38.0)


class TestCodexAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_session(self, records, name="session"):
        sessions = self.root / "sessions"
        sessions.mkdir(exist_ok=True)
        path = sessions / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return path

    def _generation_records(self):
        start = int(datetime.fromisoformat("2026-10-03T10:00:02+00:00").timestamp() * 1000)
        return [
            {"type": "turn_context", "payload": {"turn_id": "turn-1", "model": "test-model"}, "timestamp": "2026-10-03T10:00:00Z"},
            {"type": "response_item", "payload": {"type": "reasoning", "id": "r-1"}, "timestamp": "2026-10-03T10:00:02Z"},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "id": "a-1"}, "timestamp": "2026-10-03T10:00:03Z"},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "Reasoning", "id": "r-1"}, "started_at_ms": start, "completed_at_ms": start + 1000}, "timestamp": "2026-10-03T10:00:03Z"},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "AgentMessage", "id": "a-1"}, "started_at_ms": start + 1000, "completed_at_ms": start + 2000}, "timestamp": "2026-10-03T10:00:04Z"},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {"output_tokens": 120}}}, "timestamp": "2026-10-03T10:00:04Z"},
        ]

    def test_timeline_token_count_and_duplicate_usage(self):
        records = self._generation_records()
        count = records[-1]
        usage = {"type": "token_usage_record", "payload": {"response_id": "resp-1", "usage": {"output_tokens": 120}}, "timestamp": "2026-10-03T10:00:04Z"}
        for usage_records in [[count], [usage], [count, usage], [usage, count]]:
            with self.subTest(usage_records=usage_records):
                self._write_session(records[:-1] + usage_records)
                timeline = CodexAdapter(self.root).collect_sessions()[0]
                self.assertEqual(timeline.total_tokens, 120)
                self.assertEqual(timeline.assistant_messages, 1)
                self.assertIsNone(next(event for event in timeline.events if event.kind == "reasoning").tokens)

    def test_malformed_records_do_not_discard_later_generation(self):
        junk = [
            [], None, "text", 42, True,
            {"type": [], "payload": {}, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "response_item", "payload": {"type": []}, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": []}}, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "event_msg", "payload": {"type": "token_count", "info": []}, "timestamp": "2026-10-03T10:00:01Z"},
        ]
        for invalid in junk:
            with self.subTest(invalid=invalid):
                records = self._generation_records()
                records.insert(1, invalid)
                self._write_session(records)
                adapter = CodexAdapter(self.root)
                spans = adapter.collect()
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].tokens, 120)
                self.assertEqual(spans[0].model, "test-model")
                self.assertEqual(adapter.collect_sessions()[0].assistant_messages, 1)

    def test_usage_without_output_does_not_overwrite_previous_turn(self):
        records = self._generation_records() + [
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-2"}, "timestamp": "2026-10-03T10:00:05Z"},
            {"type": "event_msg", "payload": {"type": "token_count", "info": {"last_token_usage": {"output_tokens": 200}}}, "timestamp": "2026-10-03T10:00:06Z"},
        ]
        self._write_session(records)
        self.assertEqual(CodexAdapter(self.root).collect_sessions()[0].total_tokens, 120)

    def test_repeated_cumulative_snapshot_does_not_consume_new_reasoning(self):
        for second_tokens in (120, 200):
            with self.subTest(second_tokens=second_tokens):
                records = self._generation_records()
                initial_count = records[-1]
                initial_count["payload"]["info"]["total_token_usage"] = {"output_tokens": 120, "input_tokens": 300}
                start = int(datetime.fromisoformat("2026-10-03T10:00:04+00:00").timestamp() * 1000)
                records += [
                    {"type": "event_msg", "timestamp": "2026-10-03T10:00:06Z", "payload": {
                        "type": "item_completed", "item": {"type": "Reasoning", "id": "r-2"},
                        "started_at_ms": start, "completed_at_ms": start + 2000}},
                    {"type": "response_item", "timestamp": "2026-10-03T10:00:06Z",
                     "payload": {"type": "reasoning", "id": "r-2"}},
                    {**initial_count, "timestamp": "2026-10-03T10:00:06Z"},
                    {"type": "event_msg", "timestamp": "2026-10-03T10:00:08Z", "payload": {
                        "type": "item_completed", "item": {"type": "AgentMessage", "id": "a-2"},
                        "started_at_ms": start + 2000, "completed_at_ms": start + 4000}},
                    {"type": "response_item", "timestamp": "2026-10-03T10:00:08Z",
                     "payload": {"type": "message", "role": "assistant", "id": "a-2"}},
                    {"type": "event_msg", "timestamp": "2026-10-03T10:00:08Z", "payload": {
                        "type": "token_count", "info": {"last_token_usage": {"output_tokens": second_tokens},
                        "total_token_usage": {"output_tokens": 120 + second_tokens, "input_tokens": 600}}}},
                ]
                self._write_session(records)
                adapter = CodexAdapter(self.root)
                spans = adapter.collect()
                self.assertEqual([s.tokens for s in spans], [120, second_tokens])
                self.assertEqual([s.duration for s in spans], [2.0, 4.0])
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.total_tokens, 120 + second_tokens)
                self.assertTrue(all(e.tokens is None for e in timeline.events if e.kind == "reasoning"))
                cutoff = start / 1000 + 0.5
                windowed = adapter.collect(min_timestamp=cutoff)
                self.assertEqual(len(windowed), 1)
                self.assertEqual(windowed[0].tokens, second_tokens)
                self.assertEqual(windowed[0].duration, 4.0)
                self.assertEqual(windowed[0].model, "test-model")

    def test_native_diagnostic_schema_and_start_before_cutoff(self):
        self._write_session(self._generation_records())
        started = datetime.fromisoformat("2026-10-03T10:00:02+00:00").timestamp()
        with closing(sqlite3.connect(self.root / "logs_1.sqlite")) as conn:
            conn.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, target TEXT, ts INTEGER, ts_nanos INTEGER, feedback_log_body TEXT)")
            conn.execute("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", (1, "codex_core::stream_events_utils", int(started), 125000000,
                         'Output item item_type="reasoning" item_id="r-1"'))
            conn.commit()
        adapter = CodexAdapter(self.root)
        spans = adapter.collect(min_timestamp=started + 1)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].timing_source, "stream-log")
        self.assertAlmostEqual(spans[0].started_at, started + 0.125)
        self.assertAlmostEqual(spans[0].duration, 1.875)

    def test_effort_and_tier_are_captured_per_output_and_not_guessed(self):
        records = self._generation_records()
        records[0]["payload"]["effort"] = "medium"
        settings = {"type": "event_msg", "timestamp": "2026-10-03T10:00:01Z", "payload": {
            "type": "thread_settings_applied", "thread_settings": {
                "reasoning_effort": "medium", "service_tier": "priority"}}}
        records.insert(1, settings)
        # Settings for a later request must not relabel output already in progress.
        records.insert(-1, {**settings, "timestamp": "2026-10-03T10:00:04Z", "payload": {
            "type": "thread_settings_applied", "thread_settings": {
                "reasoning_effort": "high", "service_tier": "default"}}})
        self._write_session(records)
        span = CodexAdapter(self.root).collect()[0]
        self.assertEqual(span.reasoning_effort, "medium")
        self.assertEqual(span.service_tier, "priority")
        self.assertEqual(span.speed_mode, "fast")
        timeline = CodexAdapter(self.root).collect_sessions()[0]
        self.assertEqual(timeline.reasoning_effort, "high")
        self.assertEqual(timeline.speed_mode, "standard")
        for event in timeline.events:
            if event.kind in {"assistant_message", "reasoning"}:
                self.assertEqual(event.reasoning_effort, "medium")
                self.assertEqual(event.speed_mode, "fast")
        self._write_session(self._generation_records())
        unknown = CodexAdapter(self.root).collect()[0]
        self.assertIsNone(unknown.reasoning_effort)
        self.assertIsNone(unknown.speed_mode)

    def test_session_cutoff_uses_event_time_and_keeps_boundary(self):
        self._write_session(self._generation_records(), "older")
        records = [{"type": "response_item", "payload": {"type": "message", "role": "assistant"}, "timestamp": "2026-10-03T10:00:05Z"}]
        self._write_session(records, "boundary")
        cutoff = datetime.fromisoformat("2026-10-03T10:00:05+00:00").timestamp()
        adapter = CodexAdapter(self.root)
        self.assertEqual(len(adapter.collect_sessions()), 2)
        self.assertEqual([timeline.session_id for timeline in adapter.collect_sessions(min_timestamp=cutoff)], ["boundary"])

    def test_detect_returns_true_when_files_present(self):
        adapter = CodexAdapter(self.root)
        self.assertFalse(adapter.detect())
        (self.root / "sessions").mkdir()
        self.assertTrue(adapter.detect())

    def test_collect_synthetic_session(self):
        sessions_dir = self.root / "sessions"
        sessions_dir.mkdir()
        session_file = sessions_dir / "task_1.jsonl"

        records = [
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-1"}, "timestamp": "2026-10-03T10:00:00Z"},
            {"type": "turn_context", "payload": {"turn_id": "turn-1", "model": "o3-mini"}, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "id": "item-1"}, "timestamp": "2026-10-03T10:00:02Z"},
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "started_at_ms": 1759485602000,
                    "completed_at_ms": 1759485605000,
                    "item": {"id": "item-1", "type": "AgentMessage"},
                },
                "timestamp": "2026-10-03T10:00:05Z",
            },
            {
                "type": "token_usage_record",
                "payload": {"response_id": "resp-1", "usage": {"output_tokens": 150}},
                "timestamp": "2026-10-03T10:00:05Z",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        adapter = CodexAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.model, "o3-mini")
        self.assertEqual(span.tokens, 150)
        self.assertAlmostEqual(span.duration, 3.0)
        self.assertAlmostEqual(span.tps, 50.0)
        self.assertTrue(span.is_valid)

    def test_tool_first_item_is_excluded_from_tps(self):
        sessions_dir = self.root / "sessions"
        sessions_dir.mkdir()
        session_file = sessions_dir / "task_tool.jsonl"

        records = [
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-tool"}, "timestamp": "2026-10-03T10:00:00Z"},
            {"type": "turn_context", "payload": {"turn_id": "turn-tool", "model": "o3-mini"}, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "response_item", "payload": {"type": "function_call", "name": "bash", "id": "fn-1"}, "timestamp": "2026-10-03T10:00:02Z"},
            {
                "type": "token_usage_record",
                "payload": {"response_id": "resp-tool", "usage": {"output_tokens": 80}},
                "timestamp": "2026-10-03T10:00:05Z",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        adapter = CodexAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)
        self.assertFalse(spans[0].is_valid)
        self.assertEqual(spans[0].note, "unconfirmed_tool_start")
        self.assertIsNone(spans[0].tps)

    def test_subsecond_duration_is_excluded(self):
        sessions_dir = self.root / "sessions"
        sessions_dir.mkdir()
        session_file = sessions_dir / "task_fast.jsonl"

        # 100 tokens in 0.2s (500 TPS) -> should be excluded by shared < 1s validator
        records = [
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-fast"}, "timestamp": "2026-10-03T10:00:00Z"},
            {"type": "turn_context", "payload": {"turn_id": "turn-fast", "model": "o3-mini"}, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "id": "item-fast"}, "timestamp": "2026-10-03T10:00:02Z"},
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-fast",
                    "started_at_ms": 1759485602000,
                    "completed_at_ms": 1759485602200,  # 0.20s duration
                    "item": {"id": "item-fast", "type": "AgentMessage"},
                },
                "timestamp": "2026-10-03T10:00:02.200Z",
            },
            {
                "type": "token_usage_record",
                "payload": {"response_id": "resp-fast", "usage": {"output_tokens": 100}},
                "timestamp": "2026-10-03T10:00:02.200Z",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        adapter = CodexAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)
        self.assertFalse(spans[0].is_valid)
        self.assertEqual(spans[0].note, "duration_under_1s")
        self.assertIsNone(spans[0].tps)


if __name__ == "__main__":
    unittest.main()
