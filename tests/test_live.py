"""Live-mode regressions using synthetic telemetry and interruptible fake polling."""

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, call, patch

from tokenmon.adapters.antigravity import AntigravityAdapter
from tokenmon.adapters.claude import ClaudeAdapter
from tokenmon.adapters.codex import CodexAdapter
from tokenmon.adapters.omp import OMPAdapter
from tokenmon.adapters.pi import PiAdapter
from tokenmon.live import follow_session, watch_stats
from tokenmon.models import SessionTimeline, TimelineEvent, create_span


class TestLiveModes(unittest.TestCase):
    def test_follow_baselines_history_and_survives_missing_or_failed_reads(self):
        old = TimelineEvent(1000, "user_message", "turn", "existing prompt", event_id="old")
        new = TimelineEvent(1000, "tool_call", "turn", "new tool", event_id="new")
        initial = SessionTimeline("pinned", "codex", "m", 1000, 1000, [old])
        refreshed = replace(initial, events=[old, new])
        adapter = Mock()
        adapter.name = "codex"
        adapter.read_session.side_effect = [None, OSError("busy"), refreshed, refreshed]
        with patch("tokenmon.live.time.sleep", side_effect=[None] * 4 + [KeyboardInterrupt]), \
                patch("sys.stdout", new=io.StringIO()) as output, \
                self.assertLogs("tokenmon.live", level="WARNING"):
            self.assertEqual(follow_session(adapter, initial), 0)
        text = output.getvalue()
        self.assertNotIn("existing prompt", text)
        self.assertEqual(text.count("new tool"), 1)
        self.assertNotIn("\033[2J", text)
        self.assertEqual(adapter.read_session.call_args_list, [call("pinned")] * 4)

    def test_follow_keeps_distinct_identical_events_without_source_ids(self):
        old = TimelineEvent(1000, "tool_call", "turn", "same tool")
        initial = SessionTimeline("s", "codex", "m", 1000, 1000, [old])
        adapter = Mock()
        adapter.read_session.return_value = replace(initial, events=[old, old])
        with patch("tokenmon.live.time.sleep", side_effect=[None, None, KeyboardInterrupt]), \
                patch("sys.stdout", new=io.StringIO()) as output:
            follow_session(adapter, initial)
        self.assertEqual(output.getvalue().count("same tool"), 1)

    def test_watch_refreshes_cutoffs_and_prints_complete_dashboard(self):
        span = create_span(agent="codex", session_id="session", turn_id="turn", model="m",
                           tokens=120, started_at=99996, ended_at=99998, timing_source="item")
        timeline = SessionTimeline("session", "codex", "m", 99990, 99998)
        adapter = Mock()
        adapter.name = "codex"
        adapter.collect.return_value = [span]
        adapter.collect_sessions.return_value = [timeline]
        with patch("tokenmon.live.time.time", side_effect=[100000, 100002]), \
                patch("tokenmon.live.time.sleep", side_effect=[None, KeyboardInterrupt]) as sleep, \
                patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(watch_stats([adapter], window="1d", tasks=7, recent=1, compact=True), 0)
        text = output.getvalue()
        self.assertEqual(text.count("Recent 1 Generation Streams"), 2)
        self.assertEqual(text.count("Recent Sessions & User Interactions"), 2)
        self.assertIn("60.0 TPS", text)
        self.assertIn("│ 1d", text)
        self.assertNotIn("\033[2J", text)
        self.assertEqual(adapter.collect.call_args_list, [
            call(max_sessions=7, min_timestamp=13600),
            call(max_sessions=7, min_timestamp=13602),
        ])
        self.assertEqual(adapter.collect_sessions.call_args_list, [
            call(max_sessions=5, min_timestamp=None), call(max_sessions=5, min_timestamp=None),
        ])
        self.assertEqual(sleep.call_args_list, [call(2.0), call(2.0)])

    def test_watch_shows_sessions_without_spans_and_clears_only_a_terminal(self):
        adapter = Mock()
        adapter.name = "codex"
        adapter.collect.return_value = []
        adapter.collect_sessions.return_value = [SessionTimeline("session", "codex", "m", 1, 2)]
        output = io.StringIO()
        with patch("tokenmon.live.time.sleep", side_effect=KeyboardInterrupt), \
                patch("sys.stdout", new=output), patch.object(output, "isatty", return_value=True):
            watch_stats([adapter], include_all=True, recent=0)
        self.assertIn("No generation output streams", output.getvalue())
        self.assertIn("Recent Sessions & User Interactions", output.getvalue())
        self.assertIn("\033[2J\033[H", output.getvalue())
        adapter.collect.assert_called_once_with(max_sessions=64, min_timestamp=None)

    def test_failed_adapter_does_not_hide_other_adapter_data(self):
        broken, good = Mock(), Mock()
        broken.name, good.name = "broken", "codex"
        broken.collect.side_effect = OSError("busy")
        broken.collect_sessions.side_effect = OSError("busy")
        good.collect.return_value = []
        good.collect_sessions.return_value = [SessionTimeline("healthy", "codex", "m", 1, 2)]
        with patch("tokenmon.live.time.sleep", side_effect=KeyboardInterrupt), \
                patch("sys.stdout", new=io.StringIO()) as output, \
                self.assertLogs("tokenmon.cli", level="WARNING"):
            watch_stats([broken, good])
        self.assertIn("healthy", output.getvalue())


class TestFollowAdapterSources(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_records(self, path, records, mode="w"):
        with path.open(mode) as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def test_codex_partial_records_usage_updates_and_pinned_source(self):
        directory = self.root / "sessions"
        directory.mkdir()
        path = directory / "pinned.jsonl"
        self.write_records(path, [
            {"timestamp": "2026-10-03T10:00:00Z", "type": "turn_context",
             "payload": {"turn_id": "t", "model": "m"}},
            {"timestamp": "2026-10-03T10:00:01Z", "type": "response_item",
             "payload": {"type": "message", "role": "assistant", "id": "a"}},
        ])
        adapter = CodexAdapter(self.root)
        initial = adapter.collect_sessions(max_sessions=1)[0]
        tool = {"timestamp": "2026-10-03T10:00:01Z", "type": "response_item",
                "payload": {"type": "function_call", "name": "new-tool"}}
        encoded = json.dumps(tool)
        count = 0

        def poll(_interval):
            nonlocal count
            count += 1
            if count == 1:
                # A new latest session must not change the pinned target.
                self.write_records(directory / "newer.jsonl", [
                    {"timestamp": "2026-10-04T10:00:00Z", "type": "event_msg",
                     "payload": {"type": "task_started", "turn_id": "other"}},
                ])
                self.write_records(path, [
                    {"timestamp": "2026-10-03T10:00:02Z", "type": "token_usage_record",
                     "payload": {"usage": {"output_tokens": 120}}},
                    [], {"payload": []},
                ], "a")
                with path.open("a") as stream:
                    stream.write(encoded[:20])
            elif count == 2:
                with path.open("a") as stream:
                    stream.write(encoded[20:] + "\n")
            elif count == 4:
                raise KeyboardInterrupt

        with patch("tokenmon.live.time.sleep", side_effect=poll), \
                patch("sys.stdout", new=io.StringIO()) as output, \
                patch.object(adapter, "_discover_sessions", side_effect=AssertionError("rediscovery")):
            follow_session(adapter, initial)
        text = output.getvalue()
        self.assertEqual(text.count("new-tool"), 1)
        self.assertNotIn("Assistant message", text)
        self.assertNotIn("User started turn", text)
        refreshed = adapter.read_session("pinned")
        self.assertEqual(refreshed.events[0].tokens, 120)
        self.assertEqual(initial.events[0].event_id, refreshed.events[0].event_id)

    def test_claude_streamed_message_updates_are_not_new_events(self):
        directory = self.root / "projects" / "project"
        directory.mkdir(parents=True)
        path = directory / "pinned.jsonl"
        chunk = {"type": "assistant", "uuid": "chunk-1", "timestamp": "2026-10-03T10:00:00Z",
                 "message": {"id": "message-1", "model": "m", "usage": {"output_tokens": 2},
                             "content": [{"type": "thinking", "thinking": "synthetic"}]}}
        self.write_records(path, [chunk])
        adapter = ClaudeAdapter(self.root)
        initial = adapter.collect_sessions(max_sessions=1)[0]
        final = {**chunk, "uuid": "chunk-2", "timestamp": "2026-10-03T10:00:02Z",
                 "message": {**chunk["message"], "usage": {"output_tokens": 120},
                             "content": chunk["message"]["content"] + [
                                 {"type": "tool_use", "id": "tool-1", "name": "new-tool"}]}}

        poll_count = 0

        def poll(_interval):
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                self.write_records(path, [None, {"type": "assistant", "message": []}, final], "a")
            elif poll_count == 3:
                raise KeyboardInterrupt

        with patch("tokenmon.live.time.sleep", side_effect=poll), \
                patch("sys.stdout", new=io.StringIO()) as output:
            follow_session(adapter, initial)
        text = output.getvalue()
        self.assertEqual(text.count("new-tool"), 1)
        self.assertNotIn("Assistant response", text)
        self.assertNotIn("Thinking / reasoning", text)
        with patch.object(adapter, "_discover_session_files", side_effect=AssertionError("rediscovery")):
            refreshed = adapter.read_session("pinned")
        assistant = next(e for e in refreshed.events if e.kind == "assistant_message")
        self.assertEqual(assistant.tokens, 120)
        self.assertEqual(assistant.event_id, initial.events[-1].event_id)

    def test_antigravity_step_updates_do_not_repeat_existing_events(self):
        directory = self.root / "conversations"
        directory.mkdir()
        path = directory / "pinned.db"
        # Metadata field 1 contains a Timestamp proto with seconds=1000.
        metadata = b"\x0a\x03\x08\xe8\x07"
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("CREATE TABLE steps (idx integer, step_type integer, metadata blob, step_payload blob)")
            conn.execute("INSERT INTO steps VALUES (?, ?, ?, ?)", (1, 15, metadata, None))
            conn.commit()
        adapter = AntigravityAdapter(self.root)
        initial = adapter.collect_sessions(max_sessions=1)[0]
        with closing(sqlite3.connect(path)) as conn:
            # Usage field 9 contains field 3 output tokens. Updating step 1 is not an append.
            conn.execute("UPDATE steps SET metadata = ? WHERE idx = ?", (metadata + b"\x4a\x02\x18\x78", 1))
            conn.execute("INSERT INTO steps VALUES (?, ?, ?, ?)", (2, 14, metadata, None))
            conn.commit()
        with patch("tokenmon.live.time.sleep", side_effect=[None, None, KeyboardInterrupt]), \
                patch("sys.stdout", new=io.StringIO()) as output:
            follow_session(adapter, initial)
        self.assertEqual(output.getvalue().count("User message"), 1)
        self.assertNotIn("Assistant response", output.getvalue())
        self.assertEqual(adapter.read_session("pinned").total_tokens, 120)
        self.assertIsNone(adapter.read_session("missing"))


class TestOMPLiveSources(unittest.TestCase):
    """Poll real OMP journals without sleeps, network, or real transcripts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.started_at = datetime(2026, 10, 3, 10, tzinfo=timezone.utc).timestamp()
        self.main_path = self.root / "sessions" / "project" / "main.jsonl"
        self.worker_path = self.main_path.parent / "main.artifacts" / "scout.jsonl"

    def timestamp(self, offset):
        return datetime.fromtimestamp(self.started_at + offset, timezone.utc).isoformat()

    def write_records(self, path, records, mode="w"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def session_records(self, session_id, offset, tokens=80, parent=None):
        header = {"type": "session", "version": 3, "id": session_id,
                  "timestamp": self.timestamp(offset), "cwd": "/synthetic/omp-project"}
        if parent:
            header["parentSession"] = parent
        prefix = session_id + "-"
        return [
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
             "message": {"role": "user", "content": "baseline prompt",
                         "timestamp": (self.started_at + offset + 1) * 1000}},
            {"type": "message", "id": prefix + "assistant", "parentId": prefix + "user",
             "timestamp": self.timestamp(offset + 7),
             "message": {"role": "assistant", "provider": "openai", "model": "test-model",
                         "responseId": prefix + "response",
                         "content": [{"type": "text", "text": "baseline response"},
                                     {"type": "thinking", "thinking": "baseline reasoning"},
                                     {"type": "toolCall", "id": "baseline-call", "name": "baseline-tool",
                                      "arguments": {}}],
                         "usage": {"output": tokens},
                         "timestamp": (self.started_at + offset + 2) * 1000,
                         "completedAt": (self.started_at + offset + 7) * 1000,
                         "duration": 5000, "ttft": 1000, "stopReason": "toolUse"}},
        ]

    def test_follow_pins_worker_through_updates_partial_lines_and_unavailable_source(self):
        self.write_records(self.main_path, self.session_records("main-id", 0, tokens=120))
        worker_records = self.session_records("worker-id", 10, parent="main-id")
        self.write_records(self.worker_path, worker_records)
        os.utime(self.main_path, (100, 100))
        os.utime(self.worker_path, (200, 200))
        adapter = OMPAdapter(self.root)
        initial = adapter.collect_sessions(max_sessions=1)[0]
        self.assertEqual(initial.session_id, "worker-id")
        self.assertEqual(initial.total_tokens, 80)
        baseline_ids = {event.event_id for event in initial.events}
        assistant = worker_records[-1]
        updated = {
            **assistant, "timestamp": self.timestamp(18),
            "message": {**assistant["message"], "usage": {"output": 160},
                        "completedAt": (self.started_at + 18) * 1000,
                        "reasoning_effort": "low", "service_tier": "default", "speed": "slow",
                        "content": [{**assistant["message"]["content"][0], "text": "updated baseline response"}]
                                   + assistant["message"]["content"][1:]},
        }
        equal_time_user = {
            "type": "message", "id": "equal-time-user", "parentId": "worker-id-assistant",
            "timestamp": self.timestamp(17),
            "message": {"role": "user", "content": "equal-time-new-prompt",
                        "timestamp": (self.started_at + 17) * 1000},
        }
        partial_tool = {
            "type": "message", "id": "partial-result", "parentId": "equal-time-user",
            "timestamp": self.timestamp(17),
            "message": {"role": "toolResult", "toolCallId": "partial-call",
                        "toolName": "partial-line-tool", "content": [],
                        "timestamp": (self.started_at + 17) * 1000},
        }
        restored_tool = {
            "type": "message", "id": "restored-result", "parentId": "partial-result",
            "timestamp": self.timestamp(19),
            "message": {"role": "toolResult", "toolCallId": "restored-call",
                        "toolName": "restored-source-tool", "content": [],
                        "timestamp": (self.started_at + 19) * 1000},
        }
        encoded = json.dumps(partial_tool)
        hidden_path = self.worker_path.with_suffix(".temporarily-unavailable")
        polls = 0
        snapshots = []

        def poll(_interval):
            nonlocal polls
            polls += 1
            snapshots.append(output.getvalue())
            if polls == 1:
                self.write_records(self.worker_path, [updated, equal_time_user], "a")
            elif polls == 2:
                with self.worker_path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded[:len(encoded) // 2])
                # Both main and nested worker sources overtake the original bounded selection.
                newer_main = self.main_path.parent / "newer.jsonl"
                newer_worker = self.main_path.parent / "newer.artifacts" / "scout.jsonl"
                self.write_records(newer_main, self.session_records("newer-main", 100))
                self.write_records(newer_worker, self.session_records("newer-worker", 200, parent="newer-main"))
                os.utime(self.worker_path, (300, 300))
                os.utime(newer_main, (400, 400))
                os.utime(newer_worker, (500, 500))
            elif polls == 3:
                with self.worker_path.open("a", encoding="utf-8") as stream:
                    stream.write(encoded[len(encoded) // 2:] + "\n")
                os.utime(self.worker_path, (300, 300))
            elif polls == 4:
                self.worker_path.rename(hidden_path)
            elif polls == 5:
                hidden_path.rename(self.worker_path)
                self.write_records(self.worker_path, [restored_tool], "a")
                os.utime(self.worker_path, (300, 300))
            elif polls == 7:
                raise KeyboardInterrupt

        with patch("tokenmon.live.time.sleep", side_effect=poll), \
                patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(follow_session(adapter, initial, interval=0.01), 0)
        text = output.getvalue()
        self.assertEqual(text.count("Following session worker-id (omp)"), 1)
        for marker in ("equal-time-new-prompt", "partial-line-tool", "restored-source-tool"):
            self.assertEqual(text.count(marker), 1, marker)
        self.assertIn("equal-time-new-prompt", snapshots[1])
        self.assertNotIn("partial-line-tool", snapshots[2])
        self.assertIn("partial-line-tool", snapshots[3])
        self.assertEqual(snapshots[3], snapshots[4], "missing source must retain the follow baseline")
        self.assertIn("restored-source-tool", snapshots[5])
        self.assertNotIn("baseline prompt", text)
        self.assertNotIn("baseline response", text)
        self.assertNotIn("baseline-tool", text)
        self.assertNotIn("Assistant response", text)
        self.assertNotIn("Thinking / reasoning", text)
        self.assertNotIn("newer-main", text)
        self.assertNotIn("newer-worker", text)
        self.assertNotIn("\033[2J", text)
        self.assertIn("Stopped following session", text)
        self.assertEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "newer-worker")
        refreshed = adapter.read_session("worker-id")
        self.assertIsNotNone(refreshed)
        self.assertTrue(baseline_ids.issubset({event.event_id for event in refreshed.events}))
        final_assistant = next(event for event in refreshed.events if event.kind == "assistant_message")
        initial_assistant = next(event for event in initial.events if event.kind == "assistant_message")
        self.assertEqual(final_assistant.event_id, initial_assistant.event_id)
        self.assertEqual(final_assistant.tokens, 160)
        self.assertEqual(final_assistant.timestamp, self.started_at + 18)
        self.assertEqual(final_assistant.reasoning_effort, "low")
        self.assertEqual(final_assistant.service_tier, "default")
        self.assertEqual(final_assistant.speed, "slow")
        self.assertEqual(refreshed.total_tokens, 160)

    def test_watch_renders_omp_streams_and_user_only_session_cards_each_refresh(self):
        self.write_records(self.main_path, self.session_records("main-id", 0, tokens=120))
        self.write_records(self.worker_path, self.session_records("worker-id", 10, parent="main-id"))
        user_path = self.root / "sessions" / "project" / "user-only.jsonl"
        self.write_records(user_path, [
            {"type": "session", "id": "user-only-id", "timestamp": self.timestamp(20),
             "cwd": "/synthetic/omp-user-only"},
            {"type": "model_change", "id": "user-model",
             "model": "openai/test-model"},
            {"type": "message", "id": "user-only-message", "parentId": "user-model",
             "timestamp": self.timestamp(21),
             "message": {"role": "user", "content": "user-only-dashboard-prompt",
                         "timestamp": (self.started_at + 21) * 1000}},
        ])
        adapter = OMPAdapter(self.root)
        with patch("tokenmon.live.time.time", side_effect=[self.started_at + 30, self.started_at + 32]), \
                patch("tokenmon.live.time.sleep", side_effect=[None, KeyboardInterrupt]), \
                patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(watch_stats([adapter], window="1d", recent=2, compact=False), 0)
        text = output.getvalue()
        self.assertIn("30.0 TPS", text)
        self.assertIn("20.0 TPS", text)
        self.assertIn("openai/test-model", text)
        self.assertIn("user-only-id", text)
        self.assertIn("Exited watch mode", text)
        self.assertNotIn("\033[2J", text)
        user_session = adapter.read_session("user-only-id")
        self.assertEqual(user_session.user_messages, 1)
        self.assertEqual(user_session.assistant_messages, 0)
        self.assertEqual(user_session.total_tokens, 0)
        self.assertEqual([span.session_id for span in adapter.collect()], ["main-id", "worker-id"])


class TestPiLiveSources(unittest.TestCase):
    """Refresh real Pi journals; only the polling clock is controlled."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.started_at = datetime(2026, 10, 3, 10, tzinfo=timezone.utc).timestamp()
        self.path = self.root / "sessions" / "project" / "pi.jsonl"

    def timestamp(self, offset):
        return datetime.fromtimestamp(self.started_at + offset, timezone.utc).isoformat()

    def write_records(self, path, records, mode="w"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")

    def session_records(self, session_id, offset=0, assistant=True):
        prefix = session_id + "-"
        records = [
            {"type": "session", "id": session_id, "timestamp": self.timestamp(offset),
             "cwd": "/synthetic/pi-project"},
            {"type": "model_change", "id": prefix + "model",
             "provider": "openai", "modelId": "test-model"},
            {"type": "thinking_level_change", "id": prefix + "effort",
             "parentId": prefix + "model", "thinkingLevel": "high"},
            {"type": "message", "id": prefix + "user", "parentId": prefix + "effort",
             "timestamp": self.timestamp(offset + 1),
             "message": {"role": "user", "content": "historical pi prompt",
                         "timestamp": (self.started_at + offset + 1) * 1000}},
        ]
        if assistant:
            records.append({
                "type": "message", "id": prefix + "assistant", "parentId": prefix + "user",
                "timestamp": self.timestamp(offset + 7),
                "message": {"role": "assistant", "provider": "openai", "model": "test-model",
                            "api": "openai-responses", "responseId": prefix + "response",
                            "thinkingLevel": "low", "providerThinkingLevel": "medium",
                            "content": [{"type": "text", "text": "historical pi answer"}],
                            "timestamp": (self.started_at + offset + 2) * 1000,
                            "usage": {"output": 80}, "stopReason": "stop"},
            })
        return records

    def test_follow_equal_time_identities_stay_pinned_through_updates_and_read_failure(self):
        records = self.session_records("pi-pinned-id")
        self.write_records(self.path, records)
        adapter = PiAdapter(self.root)
        initial = adapter.collect_sessions(max_sessions=1)[0]
        self.assertEqual(initial.session_id, "pi-pinned-id")
        baseline_ids = {event.event_id for event in initial.events}
        assistant = records[-1]
        update = {
            **assistant, "timestamp": self.timestamp(8),
            "message": {**assistant["message"], "usage": {"output": 160},
                        "providerThinkingLevel": "high"},
        }
        new_user = {
            "type": "message", "id": "equal-user", "parentId": "pi-pinned-id-assistant",
            "timestamp": self.timestamp(7),
            "message": {"role": "user", "content": "equal-time-pi-prompt",
                        "timestamp": (self.started_at + 7) * 1000},
        }
        new_tool = {
            "type": "message", "id": "equal-tool", "parentId": "equal-user",
            "timestamp": self.timestamp(7),
            "message": {"role": "toolResult", "toolName": "restored-pi-tool",
                        "toolCallId": "call", "content": [],
                        "timestamp": (self.started_at + 7) * 1000},
        }
        hidden = self.path.with_suffix(".unavailable")
        polls = 0
        snapshots = []

        def poll(_interval):
            nonlocal polls
            polls += 1
            snapshots.append(output.getvalue())
            if polls == 1:
                self.write_records(self.path, [new_user], "a")
            elif polls == 2:
                self.write_records(self.path, [update], "a")
                newer = self.path.with_name("newer.jsonl")
                self.write_records(newer, self.session_records("pi-newer-id", 100))
                os.utime(self.path, (100, 100))
                os.utime(newer, (200, 200))
            elif polls == 3:
                self.path.rename(hidden)
            elif polls == 4:
                hidden.rename(self.path)
                self.write_records(self.path, [new_tool], "a")
                os.utime(self.path, (100, 100))
            elif polls == 6:
                raise KeyboardInterrupt

        with patch("tokenmon.live.time.sleep", side_effect=poll), \
                patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(follow_session(adapter, initial, interval=0.01), 0)
        text = output.getvalue()
        self.assertEqual(text.count("Following session pi-pinned-id (pi)"), 1)
        self.assertEqual(text.count("equal-time-pi-prompt"), 1)
        self.assertEqual(text.count("restored-pi-tool"), 1)
        self.assertIn("equal-time-pi-prompt", snapshots[1])
        self.assertEqual(snapshots[1], snapshots[2], "metadata updates must not replay events")
        self.assertEqual(snapshots[2], snapshots[3], "missing source must retain baseline")
        for marker in ("historical pi prompt", "historical pi answer", "Assistant response",
                       "pi-newer-id", "\033[2J"):
            self.assertNotIn(marker, text)
        self.assertEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "pi-newer-id")
        refreshed = adapter.read_session("pi-pinned-id")
        self.assertIsNotNone(refreshed)
        self.assertTrue(baseline_ids.issubset({event.event_id for event in refreshed.events}))
        self.assertEqual(refreshed.total_tokens, 160)
        final_assistant = next(event for event in refreshed.events if event.kind == "assistant_message")
        self.assertEqual(final_assistant.reasoning_effort, "high")
        self.assertEqual(final_assistant.timestamp, self.started_at + 8)

    def test_watch_shows_pi_and_user_only_cards_without_measurable_generation(self):
        self.write_records(self.path, self.session_records("pi-response-id"))
        self.write_records(self.path.with_name("user-only.jsonl"),
                           self.session_records("pi-user-only-id", 10, assistant=False))
        adapter = PiAdapter(self.root)
        with patch("tokenmon.live.time.time", side_effect=[self.started_at + 30, self.started_at + 32]), \
                patch("tokenmon.live.time.sleep", side_effect=[None, KeyboardInterrupt]), \
                patch("sys.stdout", new=io.StringIO()) as output:
            self.assertEqual(watch_stats([adapter], window="1d", recent=2, compact=False), 0)
        text = output.getvalue()
        self.assertEqual(text.count("Recent Sessions & User Interactions"), 2)
        self.assertEqual(text.count("\033[1mpi-response-"), 2)
        self.assertEqual(text.count("\033[1mpi-user-only"), 2)
        self.assertNotIn("Recent 1 Generation Streams", text)
        self.assertNotIn("80.0 TPS", text)
        self.assertIn("Exited watch mode", text)
        spans = adapter.collect()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].tokens, 80)
        self.assertIsNone(spans[0].tps)
        self.assertEqual(spans[0].timing_source, "pi-unconfirmed")
        self.assertEqual(spans[0].note, "unconfirmed_generation_timing")
        user_session = adapter.read_session("pi-user-only-id")
        self.assertEqual(user_session.user_messages, 1)
        self.assertEqual(user_session.assistant_messages, 0)
        self.assertEqual(user_session.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
