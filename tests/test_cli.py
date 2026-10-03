"""Integration tests for tokenmon CLI."""

import json
import io
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from tokenmon.cli import main
from tokenmon.models import SessionTimeline


class TestCLIIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        sessions_dir = self.root / "sessions"
        sessions_dir.mkdir()
        session_file = sessions_dir / "cli_test.jsonl"
        # Keep default rolling-window tests valid on any CI date.
        started_at = int(time.time()) - 60

        def timestamp(offset):
            return datetime.fromtimestamp(started_at + offset, tz=timezone.utc).isoformat()

        records = [
            {"type": "session_meta", "payload": {"id": "cli_test", "cwd": "/tmp/demo-project"}, "timestamp": timestamp(-1)},
            {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-1"}, "timestamp": timestamp(0)},
            {"type": "turn_context", "payload": {"turn_id": "turn-1", "model": "gpt-5"}, "timestamp": timestamp(1)},
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "id": "item-1"}, "timestamp": timestamp(2)},
            {
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "turn_id": "turn-1",
                    "started_at_ms": (started_at + 2) * 1000,
                    "completed_at_ms": (started_at + 4) * 1000,
                    "item": {"id": "item-1", "type": "AgentMessage"},
                },
                "timestamp": timestamp(4),
            },
            {
                "type": "token_usage_record",
                "payload": {"response_id": "resp-cli", "usage": {"output_tokens": 120}},
                "timestamp": timestamp(4),
            },
        ]
        with session_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multi_adapter_sessions_are_globally_sorted(self):
        older = SessionTimeline("older", "codex", "m", 1.0, 1.0)
        newer = SessionTimeline("newer", "claude", "m", 2.0, 2.0)
        adapters = [Mock(name="older_adapter"), Mock(name="newer_adapter")]
        adapters[0].collect_sessions.return_value = [older]
        adapters[1].collect_sessions.return_value = [newer]
        for args, expected in [
            (["logs", "--json", "--all"], "newer"),
            (["ps", "--json", "--all"], ["newer", "older"]),
            (["logs", "--window", "all", "--json"], ["newer", "older"]),
        ]:
            with self.subTest(args=args), patch.object(sys, "argv", ["tokenmon"] + args), patch("tokenmon.cli.detect_available_adapters", return_value=adapters), patch("sys.stdout", new=io.StringIO()) as output:
                self.assertEqual(main(), 0)
                result = json.loads(output.getvalue())
            selected = [timeline["session_id"] for timeline in result] if isinstance(result, list) else result["session_id"]
            self.assertEqual(selected, expected)

    def test_cli_table_output(self):
        # Test bare alias to stats
        cmd = [
            sys.executable,
            "-m",
            "tokenmon",
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

        # Test explicit stats subcommand
        cmd_stats = [
            sys.executable,
            "-m",
            "tokenmon",
            "stats",
            "codex",
            "--home",
            str(self.root),
            "--window",
            "all",
        ]
        res_stats = subprocess.run(cmd_stats, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_stats.returncode, 0)
        self.assertIn("gpt-5", res_stats.stdout)

    def test_cli_json_output(self):
        cmd = [
            sys.executable,
            "-m",
            "tokenmon",
            "stats",
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
            sys.executable,
            "-m",
            "tokenmon",
            "sessions",
            "codex",
            "--home",
            str(self.root),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertIn("Active & Recent Sessions", res.stdout)
        self.assertIn("cli_test", res.stdout)
        self.assertIn("gpt-5", res.stdout)

        # Test Docker alias 'ps'
        cmd_ps = [
            sys.executable,
            "-m",
            "tokenmon",
            "ps",
            "codex",
            "--home",
            str(self.root),
        ]
        res_ps = subprocess.run(cmd_ps, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_ps.returncode, 0)
        self.assertIn("Active & Recent Sessions", res_ps.stdout)

    def test_cli_timeline_view(self):
        cmd = [
            sys.executable,
            "-m",
            "tokenmon",
            "timeline",
            "cli_test",
            "--home",
            str(self.root),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertIn("Session Timeline", res.stdout)
        self.assertIn("USER", res.stdout)
        self.assertIn("ASSISTANT", res.stdout)

        # Test Docker alias 'logs' (default to latest)
        cmd_logs = [
            sys.executable,
            "-m",
            "tokenmon",
            "logs",
            "--home",
            str(self.root),
        ]
        res_logs = subprocess.run(cmd_logs, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_logs.returncode, 0)
        self.assertIn("Session Timeline", res_logs.stdout)

    def test_cli_timeline_window_export(self):
        cmd = [
            sys.executable,
            "-m",
            "tokenmon",
            "timeline",
            "--window",
            "1d",
            "--json",
            "--all",
            "--home",
            str(self.root),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["session_id"], "cli_test")
        self.assertIn("events", data[0])

    def test_cli_compact_and_wide_flags(self):
        # Test --compact flag with stats
        cmd_compact = [
            sys.executable,
            "-m",
            "tokenmon",
            "stats",
            "codex",
            "--home",
            str(self.root),
            "--compact",
        ]
        res_compact = subprocess.run(cmd_compact, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_compact.returncode, 0)
        self.assertIn("TPS", res_compact.stdout)
        self.assertNotIn("Time (s)", res_compact.stdout)

        # Test --wide flag with stats
        cmd_wide = [
            sys.executable,
            "-m",
            "tokenmon",
            "stats",
            "codex",
            "--home",
            str(self.root),
            "--wide",
        ]
        res_wide = subprocess.run(cmd_wide, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_wide.returncode, 0)
        self.assertIn("Time (s)", res_wide.stdout)
        self.assertIn("Range", res_wide.stdout)

    def test_cli_json_includes_session_cwd(self):
        base = [sys.executable, "-m", "tokenmon"]
        tail = ["codex", "--home", str(self.root), "--json", "--all"]

        res = subprocess.run(base + ["ps"] + tail, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertEqual(json.loads(res.stdout)[0]["cwd"], "/tmp/demo-project")

        res = subprocess.run(base + ["logs"] + tail, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertEqual(json.loads(res.stdout)["cwd"], "/tmp/demo-project")

        res = subprocess.run(base + ["stats"] + tail, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertEqual(json.loads(res.stdout)["recent_sessions"][0]["cwd"], "/tmp/demo-project")

    def test_cli_all_lists_each_adapter_once(self):
        (self.root / "conversations").mkdir()  # makes the Antigravity adapter detectable
        cmd = [sys.executable, "-m", "tokenmon", "stats", "all", "--home", str(self.root), "--json", "--all"]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        agents = json.loads(res.stdout)["meta"]["agents"]
        self.assertEqual(len(agents), len(set(agents)))
        self.assertIn("antigravity", agents)

    def test_cli_ascii_banner(self):
        # Human mode should display the ASCII banner
        cmd = [
            sys.executable,
            "-m",
            "tokenmon",
            "stats",
            "codex",
            "--home",
            str(self.root),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertIn("TokenMon", res.stdout)
        self.assertIn("Agent TPS", res.stdout)

        # JSON mode must NOT output the ASCII banner (must be clean JSON)
        cmd_json = [
            sys.executable,
            "-m",
            "tokenmon",
            "stats",
            "codex",
            "--home",
            str(self.root),
            "--json",
        ]
        res_json = subprocess.run(cmd_json, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_json.returncode, 0)
        self.assertNotIn("TokenMon", res_json.stdout)
        self.assertNotIn("Agent TPS", res_json.stdout)
        # Verify valid JSON parse
        parsed = json.loads(res_json.stdout)
        self.assertIn("meta", parsed)

    def test_cli_corrupted_session_handling(self):
        # Create a corrupted / radically unparseable session file
        sessions_dir = self.root / "sessions"
        corrupted_file = sessions_dir / "corrupted_session.jsonl"
        with corrupted_file.open("w") as f:
            f.write("{\x00\xffCORRUPTED_NON_JSON_DATA\n")
            f.write('{"type": "totally_unknown_schema", "payload": 12345}\n')
            f.write("GARBAGE_UNPARSEABLE_BYTES\n")

        # The CLI should not crash; it should log/skip corrupted file and parse valid session
        cmd = [
            sys.executable,
            "-m",
            "tokenmon",
            "stats",
            "codex",
            "--home",
            str(self.root),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res.returncode, 0)
        self.assertIn("gpt-5", res.stdout)

        # Also test sessions command with corrupted file present
        cmd_ps = [
            sys.executable,
            "-m",
            "tokenmon",
            "ps",
            "codex",
            "--home",
            str(self.root),
        ]
        res_ps = subprocess.run(cmd_ps, capture_output=True, text=True, env={"PYTHONPATH": "src"})
        self.assertEqual(res_ps.returncode, 0)
        self.assertIn("cli_test", res_ps.stdout)


if __name__ == "__main__":
    unittest.main()
