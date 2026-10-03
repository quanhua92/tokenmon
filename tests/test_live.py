"""Live-mode regressions using synthetic telemetry and interruptible fake polling."""

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, call, patch

from tokenmon.adapters.antigravity import AntigravityAdapter
from tokenmon.adapters.claude import ClaudeAdapter
from tokenmon.adapters.codex import CodexAdapter
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


if __name__ == "__main__":
    unittest.main()
