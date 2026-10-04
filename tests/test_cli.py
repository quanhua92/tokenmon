"""Integration tests for tokenmon CLI."""

import json
import io
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
from unittest.mock import Mock, patch

from tokenmon.cli import configuration_metadata, format_recent_span, format_session_duration, format_sessions_table, format_session_card, format_timeline_view, format_table, main
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, WindowSummary


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

    def test_cli_watch_routes_explicit_alias_and_default_stats_options(self):
        for prefix, watch_option in product((["stats"], ["top"], []), ("--watch", "-w")):
            with self.subTest(prefix=prefix, watch_option=watch_option), \
                    patch.object(sys, "argv", ["tokenmon"] + prefix + [
                        "codex", "--home", str(self.root), watch_option, "--window", "1d",
                        "--interval", "5", "--recent", "2", "--tasks", "7", "--wide"]), \
                    patch("tokenmon.live.watch_stats", return_value=0) as watch, \
                    patch("sys.stdout", new=io.StringIO()) as output:
                self.assertEqual(main(), 0)
                self.assertEqual(output.getvalue(), "")
                adapters = watch.call_args.args[0]
                self.assertEqual(adapters[0].root, self.root.resolve())
                self.assertEqual(watch.call_args.kwargs, {
                    "interval": 5.0, "window": "1d", "include_all": False,
                    "tasks": 7, "recent": 2, "compact": False,
                })

    def test_short_watch_runs_complete_dashboard(self):
        with patch.object(sys, "argv", ["tokenmon", "stats", "codex", "-w", "--window", "1d",
                                       "--home", str(self.root)]), \
                patch("tokenmon.live.time.sleep", side_effect=KeyboardInterrupt), \
                patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(main(), 0)
        self.assertIn("Recent 1 Generation Streams", output.getvalue())
        self.assertIn("Recent Sessions & User Interactions", output.getvalue())
        self.assertIn("60.0 TPS", output.getvalue())
        self.assertIn("Exited watch mode", output.getvalue())

    def test_cli_follow_routes_aliases_and_session_selection_without_history(self):
        for command, target in [("logs", "codex"), ("log", "cli_t"), ("timeline", "latest")]:
            with self.subTest(command=command), \
                    patch.object(sys, "argv", ["tokenmon", command, target, "--home", str(self.root),
                                              "-a", "codex", "-f", "--interval", "3"]), \
                    patch("tokenmon.live.follow_session", return_value=0) as follow, \
                    patch("sys.stdout", new=io.StringIO()) as output:
                # The positional adapter shorthand applies only without --agent.
                if target == "codex":
                    sys.argv = ["tokenmon", command, target, "--home", str(self.root), "-f", "--interval", "3"]
                self.assertEqual(main(), 0)
                self.assertEqual(output.getvalue(), "")
                adapter, initial = follow.call_args.args
                self.assertEqual(adapter.name, "codex")
                self.assertEqual(initial.session_id, "cli_test")
                self.assertTrue(initial.events)
                self.assertEqual(follow.call_args.kwargs, {"interval": 3.0})

    def test_cli_rejects_removed_window_shortcut_and_invalid_live_options(self):
        cases = [
            ["sessions", "-w", "1d"], ["logs", "-w", "1d"],
            ["stats", "--watch", "--json"], ["stats", "-w", "--json"], ["logs", "--follow", "--json"],
            ["logs", "-f", "--window", "1d"], ["logs", "window", "-f"],
        ]
        for command in ("stats", "logs"):
            for interval in ("0", "-2", "nan", "inf", "abc"):
                cases.append([command, "--interval", interval])
        for arguments in cases:
            with self.subTest(arguments=arguments), \
                    patch.object(sys, "argv", ["tokenmon"] + arguments), \
                    patch("sys.stderr", new=io.StringIO()), \
                    patch("sys.stdout", new=io.StringIO()) as output, \
                    patch("tokenmon.cli.detect_available_adapters") as detect:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(output.getvalue(), "")
                detect.assert_not_called()

    def test_follow_not_found_returns_error_without_banner(self):
        with patch.object(sys, "argv", ["tokenmon", "logs", "missing", "--agent", "codex",
                                       "--home", str(self.root), "-f"]), \
                patch("sys.stdout", new=io.StringIO()) as output, \
                patch("sys.stderr", new=io.StringIO()) as errors:
            self.assertEqual(main(), 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("not found", errors.getvalue())

    def test_json_event_shape_does_not_expose_internal_identity(self):
        with patch.object(sys, "argv", ["tokenmon", "logs", "codex", "--home", str(self.root), "--json"]), \
                patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(main(), 0)
        events = json.loads(output.getvalue())["events"]
        self.assertTrue(events)
        self.assertEqual(set(events[0]), {"timestamp", "kind", "turn_id", "summary", "tokens", "duration",
                                              "reasoning_effort", "service_tier", "speed", "speed_mode"})

    def test_follow_subprocess_starts_at_end_flushes_new_events_and_exits_on_sigint(self):
        cmd = [sys.executable, "-m", "tokenmon", "logs", "codex", "--home", str(self.root),
               "-f", "--interval", "0.01"]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env={"PYTHONPATH": "src"}) as process:
            try:
                self.assertTrue(select.select([process.stdout], [], [], 5)[0], "follow header timed out")
                header = process.stdout.readline()
                self.assertIn("Following session cli_test", header)
                with (self.root / "sessions" / "cli_test.jsonl").open("a") as stream:
                    stream.write(json.dumps({
                        "type": "response_item",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "payload": {"type": "function_call", "name": "synthetic-live-tool"},
                    }) + "\n")
                self.assertTrue(select.select([process.stdout], [], [], 5)[0], "new event timed out")
                event = process.stdout.readline()
                self.assertIn("synthetic-live-tool", event)
                process.send_signal(signal.SIGINT)
                remaining, errors = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, errors)
                output = header + event + remaining
                self.assertEqual(output.count("synthetic-live-tool"), 1)
                self.assertNotIn("Assistant message", output)
                self.assertNotIn("User started turn", output)
                self.assertNotIn("\033[2J", output)
                self.assertEqual(errors, "")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)

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
        self.assertNotIn("│ Time", res_compact.stdout)

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
        self.assertIn("│ Time", res_wide.stdout)
        self.assertIn("Range", res_wide.stdout)

    def test_stats_duration_uses_hours_minutes_and_seconds(self):
        summary = WindowSummary("7d", "m", 1, 1, 0, 2749137, 50026.5, 54.95, 55.5, 2.1, 394.0)
        output = format_table([summary], compact=False)
        self.assertIn("13h 53m 46s", output)
        self.assertIn("│ Time", output)
        self.assertNotIn("50026.5", output)
        self.assertEqual(format_session_duration(72605.9), "20h 10m 05s")
        self.assertEqual(format_session_duration(1940.7), "32m 20s")
        self.assertEqual(format_session_duration(0), "0s")

    def test_recent_stream_shows_effort_and_fast_mode_in_both_layouts(self):
        span = GenerationSpan("codex", "s", "t", "gpt-6.1-sol", 100, 1, 3,
                              reasoning_effort="medium", service_tier="priority")
        for compact in (True, False):
            with self.subTest(compact=compact):
                output = format_recent_span(span, compact=compact)
                self.assertIn("gpt-6.1-sol medium fast", output)
                self.assertIn("50.0 TPS", output)

    def test_recent_stream_json_contains_optional_metadata(self):
        path = self.root / "sessions" / "cli_test.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        context = next(record for record in records if record["type"] == "turn_context")
        context["payload"].update(effort="medium", service_tier="priority")
        path.write_text("\n".join(json.dumps(record) for record in records))
        with patch.object(sys, "argv", ["tokenmon", "stats", "codex", "--home", str(self.root), "--json"]), \
                patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(main(), 0)
        stream = json.loads(output.getvalue())["recent_streams"][0]
        self.assertEqual(stream["reasoning_effort"], "medium")
        self.assertEqual(stream["service_tier"], "priority")
        self.assertIsNone(stream["speed"])
        self.assertEqual(stream["speed_mode"], "fast")

    def test_recorded_metadata_reaches_all_json_views(self):
        path = self.root / "sessions" / "cli_test.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        next(r for r in records if r["type"] == "turn_context")["payload"].update(
            effort="medium", service_tier="priority")
        path.write_text("\n".join(json.dumps(r) for r in records))
        expected = {"reasoning_effort": "medium", "service_tier": "priority",
                    "speed": None, "speed_mode": "fast"}
        for command in (["stats"], ["ps"], ["logs"], ["logs", "window", "--window", "1d", "--agent", "codex"]):
            args = command + ([] if "--agent" in command else ["codex"])
            with self.subTest(command=command), patch.object(sys, "argv", ["tokenmon"] + args + [
                    "--home", str(self.root), "--json"]), patch("sys.stdout", new=io.StringIO()) as output:
                self.assertEqual(main(), 0)
                data = json.loads(output.getvalue())
                if command == ["stats"]:
                    values = [data["recent_streams"][0], data["recent_sessions"][0],
                              data["summary"]["1d"], data["models"]["gpt-5"]["1d"]]
                    self.assertEqual(data["summary"]["1d"]["configurations"], [expected])
                elif command == ["ps"]:
                    values = data
                else:
                    timeline = data[0] if isinstance(data, list) else data
                    values = [timeline] + [e for e in timeline["events"] if e["kind"] == "assistant_message"]
                for value in values:
                    self.assertEqual({key: value[key] for key in expected}, expected)

    def test_mixed_aggregate_configurations_and_excluded_spans(self):
        spans = [GenerationSpan("codex", "s", "t", "m", 100, 1, 3,
                                reasoning_effort=effort, service_tier="priority")
                 for effort in ("medium", "high")]
        spans.append(GenerationSpan("codex", "s", "t", "m", 0, 1, 3,
                                    reasoning_effort="low", note="invalid"))
        result = configuration_metadata(spans)
        self.assertIsNone(result["reasoning_effort"])
        self.assertEqual(result["speed_mode"], "fast")
        self.assertEqual([c["reasoning_effort"] for c in result["configurations"]], ["high", "medium"])
        self.assertEqual(configuration_metadata([])["configurations"], [])

    def test_session_and_timeline_human_views_show_recorded_settings(self):
        event = TimelineEvent(2, "reasoning", "t", "Thinking", reasoning_effort="low", service_tier="default")
        timeline = SessionTimeline("s", "codex", "m", 1, 3, [event],
                                   reasoning_effort="medium", service_tier="priority")
        for compact in (True, False):
            self.assertIn("m medium fast", format_sessions_table([timeline], 4, compact))
        self.assertIn("m medium fast", format_session_card(timeline, 4))
        text = format_timeline_view(timeline, 4)
        self.assertIn("m medium fast", text)
        self.assertIn("[low standard]", text)

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


class TestOMPCLIIntegration(unittest.TestCase):
    """Exercise OMP through the public CLI with isolated synthetic journals."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.started_at = datetime(2026, 10, 3, 10, tzinfo=timezone.utc).timestamp()
        self.main_path = self.root / "sessions" / "project" / "main.jsonl"
        self.worker_path = self.main_path.parent / "main.artifacts" / "scout.jsonl"
        self.write_records(self.main_path, self.session_records("main-id", 0, 120, tools=True))
        self.write_records(self.worker_path, self.session_records("worker-id", 20, 80))
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"

    def timestamp(self, offset):
        return datetime.fromtimestamp(self.started_at + offset, timezone.utc).isoformat()

    def write_records(self, path, records, mode="w"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def session_records(self, session_id, offset, tokens, tools=False):
        prefix = session_id + "-"
        header = {"type": "session", "version": 3, "id": session_id,
                  "timestamp": self.timestamp(offset), "cwd": "/synthetic/omp-project"}
        if session_id == "worker-id":
            header["parentSession"] = "main-id"
        content = [{"type": "text", "text": "synthetic response"}]
        if tools:
            content += [{"type": "thinking", "thinking": "synthetic reasoning"},
                        {"type": "toolCall", "id": "read-1", "name": "synthetic-read",
                         "arguments": {"path": "not-executed"}}]
        records = [
            {"type": "title", "title": "synthetic title"},
            header,
            {"type": "model_change", "id": prefix + "model", "parentId": None,
             "model": "openai/test-model"},
            {"type": "thinking_level_change", "id": prefix + "effort",
             "parentId": prefix + "model", "thinkingLevel": "high"},
            {"type": "service_tier_change", "id": prefix + "tier",
             "parentId": prefix + "effort", "serviceTier": {"openai": "priority"}},
            {"type": "message", "id": prefix + "user", "parentId": prefix + "tier",
             "timestamp": self.timestamp(offset + 1),
             "message": {"role": "user", "content": "synthetic prompt",
                         "timestamp": (self.started_at + offset + 1) * 1000}},
            {"type": "message", "id": prefix + "assistant", "parentId": prefix + "user",
             "timestamp": self.timestamp(offset + 9),
             "message": {"role": "assistant", "provider": "openai", "model": "test-model",
                         "responseId": prefix + "response", "content": content,
                         "usage": {"output": tokens, "input": 9999, "totalTokens": 999999},
                         "timestamp": (self.started_at + offset + 2) * 1000,
                         "duration": 5000, "ttft": 1000,
                         "completedAt": (self.started_at + offset + 7) * 1000,
                         "stopReason": "toolUse" if tools else "stop"}},
        ]
        if tools:
            records += [
                {"type": "message", "id": prefix + "tool", "parentId": prefix + "assistant",
                 "timestamp": self.timestamp(offset + 8),
                 "message": {"role": "toolResult", "toolCallId": "read-1",
                             "toolName": "synthetic-read", "content": [{"type": "text", "text": "synthetic result"}],
                             "timestamp": (self.started_at + offset + 8) * 1000,
                             "details": {"usage": {"output": 999999}}}},
                {"type": "model_usage", "id": prefix + "utility",
                 "usage": {"output": 999999}, "model": "test-model"},
            ]
        return records

    def run_cli(self, arguments, now=None):
        if now is None:
            command = [sys.executable, "-m", "tokenmon"]
        else:
            # Inject only the clock; registry discovery and CLI execution remain real.
            script = (
                "import sys; from unittest.mock import patch; from tokenmon.cli import main; "
                f"clock = patch('time.time', return_value={now!r}); clock.start(); "
                "sys.exit(main())"
            )
            command = [sys.executable, "-c", script]
        result = subprocess.run(command + arguments + ["--home", str(self.root)],
                                capture_output=True, text=True, env=self.env, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return result.stdout

    def assert_metadata(self, value):
        self.assertEqual({key: value[key] for key in (
            "reasoning_effort", "service_tier", "speed", "speed_mode")}, {
                "reasoning_effort": "high", "service_tier": "priority",
                "speed": None, "speed_mode": "fast",
            })

    def assert_totals(self, data, window="all"):
        summary = data["summary"][window]
        self.assertEqual(summary["total_spans"], 2)
        self.assertEqual(summary["valid_spans"], 2)
        self.assertEqual(summary["excluded_spans"], 0)
        self.assertEqual(summary["total_tokens"], 200)
        self.assertEqual(summary["total_duration_seconds"], 8.0)
        self.assertEqual(summary["weighted_tps"], 25.0)
        self.assert_metadata(summary)
        self.assertEqual(summary["configurations"], [{
            "reasoning_effort": "high", "service_tier": "priority",
            "speed": None, "speed_mode": "fast",
        }])

    def test_stats_json_counts_main_and_nested_worker_once(self):
        data = json.loads(self.run_cli(["stats", "omp", "--all", "--json"]))
        self.assertEqual(set(data), {"meta", "summary", "models", "recent_streams", "recent_sessions"})
        self.assertEqual(data["meta"]["agents"], ["omp"])
        self.assertEqual(data["meta"]["inspected_spans"], 2)
        self.assert_totals(data)
        self.assertEqual(set(data["models"]), {"openai/test-model"})
        self.assertEqual(data["models"]["openai/test-model"]["all"], data["summary"]["all"])
        streams = {stream["session_id"]: stream for stream in data["recent_streams"]}
        self.assertEqual(set(streams), {"main-id", "worker-id"})
        for session_id, tokens, tps, offset in (("main-id", 120, 30.0, 0), ("worker-id", 80, 20.0, 20)):
            with self.subTest(session_id=session_id):
                stream = streams[session_id]
                self.assertEqual(stream["agent"], "omp")
                self.assertEqual(stream["model"], "openai/test-model")
                self.assertEqual(stream["tokens"], tokens)
                self.assertEqual(stream["duration"], 4.0)
                self.assertEqual(stream["tps"], tps)
                self.assertEqual(stream["timestamp"], self.started_at + offset + 7)
                self.assertEqual(stream["timing_source"], "omp-ttft")
                self.assert_metadata(stream)
        self.assertEqual({card["session_id"] for card in data["recent_sessions"]}, set(streams))
        for card in data["recent_sessions"]:
            self.assertEqual(card["cwd"], "/synthetic/omp-project")
            self.assert_metadata(card)
        self.assertNotIn("event_id", json.dumps(data))

    def test_sessions_json_and_exact_header_id_timelines(self):
        now = self.started_at + 60
        sessions = json.loads(self.run_cli(["sessions", "omp", "--all", "--json"], now=now))
        self.assertEqual([session["session_id"] for session in sessions], ["worker-id", "main-id"])
        self.assertEqual(json.loads(self.run_cli(["ps", "omp", "--all", "--json"], now=now)), sessions)
        for session_id, tokens, duration, kinds in [
            ("main-id", 120, 7.0, ["user_message", "assistant_message", "reasoning", "tool_call", "tool_output"]),
            ("worker-id", 80, 6.0, ["user_message", "assistant_message"]),
        ]:
            with self.subTest(session_id=session_id):
                timeline = json.loads(self.run_cli(
                    ["logs", session_id, "--agent", "omp", "--all", "--json"], now=now))
                session = next(item for item in sessions if item["session_id"] == session_id)
                self.assertEqual({key: timeline[key] for key in session}, session)
                self.assertEqual(timeline["agent"], "omp")
                self.assertEqual(timeline["model"], "openai/test-model")
                self.assertEqual(timeline["cwd"], "/synthetic/omp-project")
                self.assertEqual(timeline["total_tokens"], tokens)
                self.assertEqual(timeline["duration_seconds"], duration)
                self.assertEqual(timeline["user_messages"], 1)
                self.assertEqual(timeline["assistant_messages"], 1)
                self.assertEqual(timeline["tool_calls"], int(session_id == "main-id"))
                self.assertEqual(sorted(event["kind"] for event in timeline["events"]), sorted(kinds))
                self.assertEqual(sum(event["tokens"] or 0 for event in timeline["events"]), tokens)
                self.assert_metadata(timeline)
                for event in timeline["events"]:
                    self.assertEqual(set(event), {"timestamp", "kind", "turn_id", "summary", "tokens",
                                                  "duration", "reasoning_effort", "service_tier", "speed", "speed_mode"})
                    self.assert_metadata(event)
                    if event["kind"] != "assistant_message":
                        self.assertIsNone(event["tokens"])
                if session_id == "main-id":
                    tool_events = [event for event in timeline["events"] if event["kind"].startswith("tool_")]
                    self.assertTrue(all("synthetic-read" in event["summary"] for event in tool_events))

    def test_shorthand_bare_agent_and_all_registry_behavior(self):
        latest = json.loads(self.run_cli(["logs", "omp", "--all", "--json"]))
        self.assertEqual(latest["session_id"], "worker-id")
        self.assertEqual(latest["total_tokens"], 80)
        self.assert_totals(json.loads(self.run_cli(["omp", "--all", "--json"])))
        human = self.run_cli(["omp", "--all"])
        self.assertIn("openai/test-model", human)
        self.assertIn("25.0", human)
        all_agents = json.loads(self.run_cli(["stats", "all", "--all", "--json"]))
        self.assertEqual(all_agents["meta"]["agents"].count("omp"), 1)
        self.assertEqual(len(all_agents["meta"]["agents"]), len(set(all_agents["meta"]["agents"])))
        self.assert_totals(all_agents)
        self.assertEqual({stream["agent"] for stream in all_agents["recent_streams"]}, {"omp"})

    def test_auto_detection_uses_synthetic_root_and_injected_current_time(self):
        now = self.started_at + 60
        data = json.loads(self.run_cli(["stats", "--json"], now=now))
        self.assertEqual(data["meta"]["agents"], ["omp"])
        self.assertEqual(data["meta"]["timestamp"], now)
        self.assert_totals(data, window="1d")
        self.assertEqual({stream["session_id"] for stream in data["recent_streams"]}, {"main-id", "worker-id"})

    def test_worker_follow_subprocess_emits_equal_time_append_once_and_stops(self):
        command = [sys.executable, "-m", "tokenmon", "logs", "worker-id", "--agent", "omp",
                   "--home", str(self.root), "--all", "-f", "--interval", "0.01"]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=self.env) as process:
            try:
                self.assertTrue(select.select([process.stdout], [], [], 5)[0], "OMP follow header timed out")
                header = process.stdout.readline()
                self.assertIn("Following session worker-id (omp)", header)
                self.write_records(self.worker_path, [{
                    "type": "message", "id": "live-result", "parentId": "worker-id-assistant",
                    "timestamp": self.timestamp(27),
                    "message": {"role": "toolResult", "toolCallId": "live-call",
                                "toolName": "equal-time-live-tool", "content": [],
                                "timestamp": (self.started_at + 27) * 1000},
                }], "a")
                self.assertTrue(select.select([process.stdout], [], [], 5)[0], "OMP appended event timed out")
                event = process.stdout.readline()
                self.assertIn("equal-time-live-tool", event)
                process.send_signal(signal.SIGINT)
                remaining, errors = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, errors)
                text = header + event + remaining
                self.assertEqual(text.count("equal-time-live-tool"), 1)
                self.assertIn("Stopped following session", text)
                self.assertNotIn("synthetic prompt", text)
                self.assertNotIn("Assistant response", text)
                self.assertNotIn("TokenMon", text)
                self.assertNotIn("\033[2J", text)
                self.assertEqual(errors, "")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
