"""Unit tests for interactive terminal shell (cmd.Cmd)."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_monitor.adapters.codex import CodexAdapter
from llm_monitor.terminal import MonitorShell


class TestTerminalShell(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        sessions_dir = self.root / "sessions"
        sessions_dir.mkdir()
        session_file = sessions_dir / "shell_test.jsonl"
        records = [
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-1"}, "timestamp": "2026-10-03T10:00:00Z"},
            {"type": "turn_context", "payload": {"turn_id": "turn-1", "model": "gpt-5"}, "timestamp": "2026-10-03T10:00:01Z"},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "id": "item-1"}, "timestamp": "2026-10-03T10:00:02Z"},
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "started_at_ms": 1759485602000,
                    "completed_at_ms": 1759485604000,
                    "item": {"id": "item-1", "type": "AgentMessage"},
                },
                "timestamp": "2026-10-03T10:00:04Z",
            },
            {
                "type": "token_usage_record",
                "payload": {"response_id": "resp-1", "usage": {"output_tokens": 120}},
                "timestamp": "2026-10-03T10:00:04Z",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        self.adapter = CodexAdapter(self.root)
        self.shell = MonitorShell([self.adapter])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_shell_summary_command(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            self.shell.do_summary("all")
            output = fake_out.getvalue()
            self.assertIn("gpt-5", output)
            self.assertIn("60.0", output)

    def test_shell_sessions_command(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            self.shell.do_sessions("")
            output = fake_out.getvalue()
            self.assertIn("shell_test", output)
            self.assertIn("gpt-5", output)

    def test_shell_timeline_command(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            self.shell.do_timeline("latest")
            output = fake_out.getvalue()
            self.assertIn("Session Timeline", output)
            self.assertIn("USER", output)

    def test_shell_recent_command(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            self.shell.do_recent("5")
            output = fake_out.getvalue()
            self.assertIn("60.0 TPS", output)

    def test_shell_exit_command(self):
        self.assertTrue(self.shell.do_exit(""))

    def test_shell_agents_lists_registry_and_active_adapter(self):
        with patch("sys.stdout", new=io.StringIO()) as output:
            self.shell.do_agents("")
        text = output.getvalue()
        self.assertIn("codex", text)
        self.assertIn("[Active]", text)
        self.assertIn("claude", text)
        self.assertIn("antigravity", text)


if __name__ == "__main__":
    unittest.main()
