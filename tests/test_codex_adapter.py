"""Unit tests for llm-monitor models, analyzer, and CodexAdapter."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from llm_monitor.adapters.codex import CodexAdapter
from llm_monitor.analyzer import analyze_windows, filter_by_window, summarize_spans
from llm_monitor.models import GenerationSpan


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


class TestCodexAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

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

