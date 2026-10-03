"""Integration tests for llm-monitor CLI."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestCLIIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        sessions_dir = self.root / "sessions"
        sessions_dir.mkdir()
        session_file = sessions_dir / "cli_test.jsonl"
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
                "payload": {"response_id": "resp-cli", "usage": {"output_tokens": 120}},
                "timestamp": "2026-10-03T10:00:04Z",
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_table_output(self):
        cmd = [
            "python3",
            "-m",
            "llm_monitor",
            "codex",
            "--home",
            str(self.root),
            "--window",
            "all",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertIn("gpt-5", res.stdout)
        self.assertIn("60.0", res.stdout)

    def test_cli_json_output(self):
        cmd = [
            "python3",
            "-m",
            "llm_monitor",
            "codex",
            "--home",
            str(self.root),
            "--json",
            "--all",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertIn("meta", data)
        self.assertIn("summary", data)
        self.assertIn("models", data)
        self.assertIn("recent_streams", data)
        self.assertIn("recent_sessions", data)
        self.assertIn("all", data["summary"])
        self.assertEqual(data["summary"]["all"]["weighted_tps"], 60.0)
        self.assertIn("gpt-5", data["models"])
        self.assertEqual(len(data["recent_streams"]), 1)

    def test_cli_sessions_list(self):
        cmd = [
            "python3",
            "-m",
            "llm_monitor",
            "codex",
            "--home",
            str(self.root),
            "--sessions",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertIn("Active & Recent Sessions", res.stdout)
        self.assertIn("cli_test", res.stdout)
        self.assertIn("gpt-5", res.stdout)

    def test_cli_timeline_view(self):
        cmd = [
            "python3",
            "-m",
            "llm_monitor",
            "codex",
            "--home",
            str(self.root),
            "--timeline",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertIn("Session Timeline", res.stdout)
        self.assertIn("USER", res.stdout)
        self.assertIn("ASSISTANT", res.stdout)

    def test_cli_timeline_window_export(self):
        cmd = [
            "python3",
            "-m",
            "llm_monitor",
            "codex",
            "--home",
            str(self.root),
            "--timeline",
            "--window",
            "1d",
            "--json",
            "--all",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["session_id"], "cli_test")
        self.assertIn("events", data[0])

    def test_cli_compact_and_wide_flags(self):
        # Test --compact flag
        cmd_compact = [
            "python3",
            "-m",
            "llm_monitor",
            "codex",
            "--home",
            str(self.root),
            "--compact",
        ]
        res_compact = subprocess.run(cmd_compact, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_compact.returncode, 0)
        self.assertIn("TPS", res_compact.stdout)
        self.assertNotIn("Time (s)", res_compact.stdout)

        # Test --wide flag
        cmd_wide = [
            "python3",
            "-m",
            "llm_monitor",
            "codex",
            "--home",
            str(self.root),
            "--wide",
        ]
        res_wide = subprocess.run(cmd_wide, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_wide.returncode, 0)
        self.assertIn("Time (s)", res_wide.stdout)
        self.assertIn("Range", res_wide.stdout)


if __name__ == "__main__":
    unittest.main()


