"""Unit tests for ClaudeAdapter."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tokenmon.adapters.claude import ClaudeAdapter


class TestClaudeAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_session(self, records, name="session"):
        projects = self.root / "projects" / "regression"
        projects.mkdir(parents=True, exist_ok=True)
        path = projects / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return path

    def _stream_records(self):
        return [
            {"type": "user", "message": {"content": "hello"}, "timestamp": "2026-10-03T10:00:00Z"},
            {"type": "assistant", "message": {"id": "a-1", "model": "test-model", "content": [{"type": "thinking"}, {"type": "tool_use", "id": "tool-1", "name": "Bash"}], "usage": {"output_tokens": 10}}, "timestamp": "2026-10-03T10:00:02Z"},
            {"type": "assistant", "message": {"id": "a-1", "model": "test-model", "content": [{"type": "thinking"}, {"type": "tool_use", "id": "tool-1", "name": "Bash"}], "usage": {"output_tokens": 100}}, "timestamp": "2026-10-03T10:00:04Z"},
        ]

    def test_stream_timeline_keeps_final_usage_and_unique_events(self):
        self._write_session(self._stream_records())
        adapter = ClaudeAdapter(self.root)
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.total_tokens, adapter.collect()[0].tokens)
        self.assertEqual(timeline.total_tokens, 100)
        self.assertEqual(timeline.assistant_messages, 1)
        self.assertEqual(timeline.tool_calls, 1)
        self.assertEqual(sum(event.kind == "reasoning" for event in timeline.events), 1)
        self.assertEqual(timeline.updated_at, datetime.fromisoformat("2026-10-03T10:00:04+00:00").timestamp())

    def test_malformed_records_do_not_discard_later_generation(self):
        junk = [
            [], None, "text", 42, True,
            {"type": [], "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "assistant", "message": None, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "assistant", "message": {"usage": []}, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "user", "message": None, "timestamp": "2026-10-03T10:00:01Z"},
        ]
        for invalid in junk:
            with self.subTest(invalid=invalid):
                records = self._stream_records()
                records.insert(1, invalid)
                self._write_session(records)
                adapter = ClaudeAdapter(self.root)
                spans = adapter.collect()
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].tokens, 100)
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.assistant_messages, 1)
                self.assertEqual(timeline.total_tokens, 100)

    def test_session_cutoff_uses_event_time_and_keeps_boundary(self):
        self._write_session(self._stream_records(), "older")
        self._write_session([{"type": "user", "message": {"content": "hello"}, "timestamp": "2026-10-03T10:00:05Z"}], "boundary")
        cutoff = datetime.fromisoformat("2026-10-03T10:00:05+00:00").timestamp()
        adapter = ClaudeAdapter(self.root)
        self.assertEqual(len(adapter.collect_sessions()), 2)
        self.assertEqual([timeline.session_id for timeline in adapter.collect_sessions(min_timestamp=cutoff)], ["boundary"])

    def test_detect_claude_projects(self):
        adapter = ClaudeAdapter(self.root)
        self.assertFalse(adapter.detect())
        (self.root / "projects").mkdir()
        self.assertTrue(adapter.detect())

    def test_collect_synthetic_claude_session(self):
        projects = self.root / "projects" / "test-project"
        projects.mkdir(parents=True)
        session_file = projects / "session-123.jsonl"

        records = [
            {
                "type": "user",
                "message": {"role": "user", "content": "hello world"},
                "timestamp": "2026-05-20T06:01:50.000Z",
                "uuid": "u-1",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg-1",
                    "role": "assistant",
                    "model": "claude-3-7-sonnet",
                    "content": [{"type": "text", "text": "Hi there!"}],
                    "usage": {"output_tokens": 100},
                },
                "timestamp": "2026-05-20T06:01:52.000Z",
                "uuid": "a-1",
            },
            {
                "type": "system",
                "subtype": "turn_duration",
                "durationMs": 2000,
                "timestamp": "2026-05-20T06:01:52.010Z",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        adapter = ClaudeAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)

        span = spans[0]
        self.assertEqual(span.agent, "claude")
        self.assertEqual(span.model, "claude-3-7-sonnet")
        self.assertEqual(span.tokens, 100)
        self.assertAlmostEqual(span.duration, 2.0)
        self.assertAlmostEqual(span.tps, 50.0)
        self.assertTrue(span.is_valid)

    def test_collect_claude_timeline(self):
        projects = self.root / "projects" / "test-project"
        projects.mkdir(parents=True)
        session_file = projects / "session-456.jsonl"

        records = [
            {
                "type": "user",
                "message": {"role": "user", "content": "run tests"},
                "timestamp": "2026-05-20T06:01:50.000Z",
                "uuid": "u-1",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg-tool",
                    "role": "assistant",
                    "model": "claude-3-7-sonnet",
                    "content": [
                        {"type": "thinking", "thinking": "Let me run bash tests"},
                        {"type": "tool_use", "name": "Bash", "id": "t-1"},
                    ],
                    "usage": {"output_tokens": 50},
                },
                "timestamp": "2026-05-20T06:01:52.000Z",
                "uuid": "a-tool",
            },
            {
                "type": "tool_result",
                "name": "Bash",
                "timestamp": "2026-05-20T06:01:53.000Z",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg-final",
                    "role": "assistant",
                    "model": "claude-3-7-sonnet",
                    "content": [{"type": "text", "text": "Tests passed!"}],
                    "usage": {"output_tokens": 30},
                },
                "timestamp": "2026-05-20T06:01:54.000Z",
                "uuid": "a-final",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        adapter = ClaudeAdapter(self.root)
        timelines = adapter.collect_sessions()
        self.assertEqual(len(timelines), 1)

        t = timelines[0]
        self.assertEqual(t.session_id, "session-456")
        self.assertEqual(t.agent, "claude")
        self.assertEqual(t.user_messages, 1)
        self.assertEqual(t.assistant_messages, 2)
        self.assertEqual(t.tool_calls, 1)
        self.assertEqual(t.total_tokens, 80)

    def test_timeline_cwd_from_records(self):
        projects = self.root / "projects" / "test-cwd"
        projects.mkdir(parents=True)
        with (projects / "session-cwd.jsonl").open("w") as f:
            f.write(json.dumps({"type": "summary", "cwd": "/tmp/demo-project"}) + "\n")
            f.write(
                json.dumps(
                    {
                        "type": "user",
                        "cwd": "/tmp/other",
                        "message": {"role": "user", "content": "hi"},
                        "timestamp": "2026-05-20T06:01:50.000Z",
                        "uuid": "u-1",
                    }
                )
                + "\n"
            )

        t = ClaudeAdapter(self.root).collect_sessions()[0]
        self.assertEqual(t.cwd, "/tmp/demo-project")  # first one wins, even on a record with no timestamp

    def test_timeline_cwd_missing_is_none(self):
        projects = self.root / "projects" / "test-nocwd"
        projects.mkdir(parents=True)
        with (projects / "session-nocwd.jsonl").open("w") as f:
            f.write(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": "hi"},
                        "timestamp": "2026-05-20T06:01:50.000Z",
                        "uuid": "u-1",
                    }
                )
                + "\n"
            )

        t = ClaudeAdapter(self.root).collect_sessions()[0]
        self.assertIsNone(t.cwd)

    def test_duration_under_1s_is_excluded(self):
        projects = self.root / "projects" / "test-fast"
        projects.mkdir(parents=True)
        session_file = projects / "session-fast.jsonl"

        records = [
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
                "timestamp": "2026-05-20T06:01:50.000Z",
                "uuid": "u-fast",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg-fast",
                    "role": "assistant",
                    "model": "claude-3-7-sonnet",
                    "content": [{"type": "text", "text": "Quick"}],
                    "usage": {"output_tokens": 10},
                },
                "timestamp": "2026-05-20T06:01:50.400Z",
                "uuid": "a-fast",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        adapter = ClaudeAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)
        self.assertFalse(spans[0].is_valid)
        self.assertEqual(spans[0].note, "duration_under_1s")
        self.assertIsNone(spans[0].tps)

    def test_burst_chunks_under_1s_excluded(self):
        # Reproduces 43,000 TPS / 30,000 TPS burst flush bug where chunks arrive in 0.01s
        projects = self.root / "projects" / "test-burst"
        projects.mkdir(parents=True)
        session_file = projects / "session-burst.jsonl"

        records = [
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
                "timestamp": "2026-06-13T19:28:10.000Z",
                "uuid": "u-burst",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg-burst",
                    "role": "assistant",
                    "model": "glm-5.2",
                    "content": [{"type": "text", "text": "chunk1"}],
                    "usage": {"output_tokens": 44},
                },
                "timestamp": "2026-06-13T19:28:20.000Z",
                "uuid": "a-1",
            },
            {
                "type": "assistant",
                "message": {
                    "id": "msg-burst",
                    "role": "assistant",
                    "model": "glm-5.2",
                    "content": [{"type": "text", "text": "chunk2"}],
                    "usage": {"output_tokens": 44},
                },
                "timestamp": "2026-06-13T19:28:20.010Z",
                "uuid": "a-2",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        adapter = ClaudeAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)
        self.assertFalse(spans[0].is_valid)
        self.assertEqual(spans[0].note, "duration_under_1s")
        self.assertIsNone(spans[0].tps)


if __name__ == "__main__":
    unittest.main()
