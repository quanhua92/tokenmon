"""Unit tests for ClaudeAdapter."""

import json
import tempfile
import unittest
from pathlib import Path

from llm_monitor.adapters.claude import ClaudeAdapter


class TestClaudeAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

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


if __name__ == "__main__":
    unittest.main()

