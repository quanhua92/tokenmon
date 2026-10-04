"""Supported Antigravity worker conversations use the ordinary SQLite step schema.

Local native schema evidence: conversation_summaries has conversation_id,
last_modified_time, parent_conversation_id, nesting_depth, and agent_name;
conversations/<conversation_id>.db has steps and optional gen_metadata.
Native conversations also have SQLite WAL sidecars. A summary row is not
required for a persisted source. Parent association does not establish that
its steps are copied history. These fixtures intentionally contain only
synthetic protobuf timestamps, counts, model/configuration identifiers, and
no transcript content. Opaque parent_references/subtrajectory blobs are not
assigned an invented worker format.
"""

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tokenmon.adapters.antigravity import AntigravityAdapter, open_ro_db


def _varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number, value):
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _metadata(start, end, tokens, end_field=7):
    return (_field(1, _field(1, start)) +
            _field(end_field, _field(1, end)) +
            _field(9, _field(3, tokens)))


def _settings(model, last_index=0, **values):
    entry = _field(19, model.encode()) + _field(20,
        _field(1, b"last_step_index") + _field(2, str(last_index).encode()))
    for key, value in values.items():
        entry += _field(20, _field(1, key.encode()) + _field(2, value.encode()))
    return _field(1, entry)


class TestAntigravitySubagents(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.conversations = self.root / "conversations"
        self.conversations.mkdir()
        self.adapter = AntigravityAdapter(self.root)

    def _conversation(self, session_id, rows, *, modified=100, settings=None, payload=True):
        path = self.conversations / f"{session_id}.db"
        with closing(sqlite3.connect(path)) as conn:
            suffix = ", step_payload BLOB" if payload else ""
            conn.execute(f"CREATE TABLE steps (idx INTEGER PRIMARY KEY, step_type INTEGER, metadata BLOB{suffix})")
            conn.executemany("INSERT INTO steps (idx, step_type, metadata) VALUES (?, ?, ?)", rows)
            if settings is not None:
                conn.execute("CREATE TABLE gen_metadata (data BLOB)")
                conn.executemany("INSERT INTO gen_metadata VALUES (?)", [(value,) for value in settings])
            conn.commit()
        os.utime(path, (modified, modified))
        return path

    def _index(self, rows):
        """Rows are (canonical conversation ID, modification seconds, parent ID)."""
        path = self.root / "conversation_summaries.db"
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("CREATE TABLE conversation_summaries (conversation_id TEXT PRIMARY KEY, "
                         "last_modified_time TEXT, workspace_uris TEXT, parent_conversation_id TEXT, "
                         "nesting_depth INTEGER, agent_name TEXT)")
            for session_id, modified, parent in rows:
                stamp = datetime.fromtimestamp(modified, timezone.utc).isoformat()
                conn.execute("INSERT INTO conversation_summaries VALUES (?, ?, ?, ?, ?, ?)",
                             (session_id, stamp, "[]", parent, int(bool(parent)), "scout" if parent else ""))
            conn.commit()
        return path

    def test_unindexed_worker_counts_independently_and_tool_usage_is_not_output(self):
        main = self._conversation("main-id", [
            (1, 15, _metadata(2000, 2004, 120)),
            (2, 132, _metadata(2010, 2014, 80000)),
        ], modified=50, settings=[_settings("main-model", reasoning_effort="high")])
        worker = self._conversation("worker-id", [(1, 15, _metadata(2010, 2014, 80))],
                                    modified=2500, settings=[_settings("worker-model", service_tier="priority")])
        summary = self._index([("main-id", 2020, "")])
        original_bytes = {path: path.read_bytes() for path in (main, worker, summary)}

        spans = self.adapter.collect(max_sessions=2)
        self.assertEqual([span.session_id for span in spans], ["main-id", "worker-id"])
        self.assertEqual([span.tokens for span in spans], [120, 80])
        self.assertEqual([span.duration for span in spans], [4, 4])
        self.assertEqual([span.tps for span in spans], [30, 20])
        self.assertEqual(sum(span.tokens for span in spans) / sum(span.duration for span in spans), 25)
        self.assertTrue(all(span.is_valid and span.timing_source == "agy-step-proto" for span in spans))
        self.assertEqual(spans[0].reasoning_effort, "high")
        self.assertEqual(spans[1].model, "worker-model")
        self.assertEqual(spans[1].service_tier, "priority")
        self.assertEqual(spans[1].speed_mode, "fast")
        self.assertIsNone(spans[1].speed)
        self.assertIsNone(spans[1].reasoning_effort)

        sessions = {session.session_id: session for session in self.adapter.collect_sessions(max_sessions=2)}
        self.assertEqual(set(sessions), {"main-id", "worker-id"})
        self.assertEqual([event.kind for event in sessions["main-id"].events], ["assistant_message", "tool_call"])
        self.assertIsNone(sessions["main-id"].events[1].tokens)
        self.assertEqual(sum(event.tokens or 0 for session in sessions.values() for event in session.events), 200)
        self.assertIsNone(sessions["worker-id"].cwd)
        self.assertEqual(self.adapter.read_session("worker-id").session_id, "worker-id")
        self.assertEqual({path: path.read_bytes() for path in original_bytes}, original_bytes)

    def test_recent_unindexed_worker_outranks_indexed_main_under_one_limit(self):
        self._conversation("main-id", [(1, 15, _metadata(1000, 1004, 120))], modified=9000)
        worker = self._conversation("worker-id", [(1, 15, _metadata(2000, 2004, 80))], modified=2500)
        self._index([("main-id", 1005, "")])
        with patch("tokenmon.adapters.antigravity.open_ro_db", wraps=open_ro_db) as opened:
            spans = self.adapter.collect(max_sessions=1)
        self.assertEqual([span.session_id for span in spans], ["worker-id"])
        conversation_reads = [call.args[0] for call in opened.call_args_list
                              if call.args[0].parent == self.conversations]
        self.assertEqual(conversation_reads, [worker])
        self.assertEqual([session.session_id for session in self.adapter.collect_sessions(max_sessions=1)], ["worker-id"])
        self.assertEqual(self.adapter.last_generation_timestamp(), 2004)

    def test_index_order_merges_with_worker_recency_and_ignores_missing_rows(self):
        self._conversation("main-new", [(1, 15, _metadata(3000, 3004, 120))], modified=10)
        self._conversation("main-old", [(1, 15, _metadata(1000, 1004, 120))], modified=9000)
        self._conversation("worker-id", [(1, 15, _metadata(2000, 2004, 80))], modified=2000)
        self._index([("main-new", 3005, ""), ("main-old", 1005, ""), ("missing-id", 4000, "")])
        self.assertEqual({span.session_id for span in self.adapter.collect(max_sessions=2)}, {"main-new", "worker-id"})
        self.assertEqual({span.session_id for span in self.adapter.collect(max_sessions=3)},
                         {"main-new", "main-old", "worker-id"})

    def test_cutoff_uses_recorded_times_not_stale_source_or_summary_times(self):
        worker = self._conversation("worker-id", [(1, 15, _metadata(2000, 2004, 80))], modified=100)
        self._index([("worker-id", 100, "main-id")])
        self.assertEqual([span.session_id for span in self.adapter.collect(min_timestamp=2004)], ["worker-id"])
        self.assertEqual(self.adapter.collect(min_timestamp=2004.001), [])
        self.assertEqual(self.adapter.read_session("worker-id").events[0].duration, 4)
        self.assertEqual(worker.stat().st_mtime, 100)

    def test_completed_worker_activity_uses_end_not_generation_start(self):
        self._conversation("worker-id", [(1, 15, _metadata(2000, 2200, 80))])
        sessions = self.adapter.collect_sessions(min_timestamp=2200)
        self.assertEqual([session.session_id for session in sessions], ["worker-id"])
        worker = sessions[0]
        self.assertEqual(worker.session_duration, 200)
        self.assertEqual(worker.idle_time(2201), 1)
        self.assertEqual(worker.status(2201), "Active")
        self.assertEqual(self.adapter.collect_sessions(min_timestamp=2200.001), [])

    def test_unindexed_stale_mtime_does_not_hide_recent_output_and_touch_is_not_activity(self):
        old = self._conversation("old-worker", [(1, 15, _metadata(1000, 1004, 40))], modified=10)
        self._conversation("recent-worker", [(1, 15, _metadata(2000, 2004, 80))], modified=20)
        self.assertEqual([span.session_id for span in self.adapter.collect(max_sessions=2, min_timestamp=2004)], ["recent-worker"])
        self.assertEqual([session.session_id for session in self.adapter.collect_sessions(max_sessions=2, min_timestamp=2000)], ["recent-worker"])
        os.utime(old, (9000, 9000))
        self.assertEqual(self.adapter.collect(max_sessions=1, min_timestamp=2000), [])
        self.assertEqual(self.adapter.collect_sessions(max_sessions=1, min_timestamp=2000), [])

    def test_parent_association_and_equal_usage_do_not_collapse_distinct_worker_sessions(self):
        rows = [(1, 15, _metadata(2000, 2004, 80))]
        for session_id in ("parent-a", "parent-b", "worker-a", "worker-b"):
            self._conversation(session_id, rows, modified=3000)
        self._index([("parent-a", 3000, ""), ("parent-b", 2999, ""),
                     ("worker-a", 2998, "parent-a"), ("worker-b", 2997, "parent-b")])
        spans = self.adapter.collect(max_sessions=4)
        self.assertEqual(len(spans), 4)
        self.assertEqual(sum(span.tokens for span in spans), 320)
        self.assertEqual({span.session_id for span in spans}, {"parent-a", "parent-b", "worker-a", "worker-b"})
        for session_id in ("worker-a", "worker-b"):
            self.assertEqual(self.adapter.read_session(session_id).session_id, session_id)

    def test_bad_worker_database_and_protobuf_do_not_discard_other_workers(self):
        self._conversation("main-id", [(1, 15, _metadata(1000, 1004, 120))], modified=1000)
        self._conversation("worker-id", [
            (1, 15, 123),
            (2, 15, b"\x0a\xff"),
            (3, 15, _metadata(2000, 2004, 80, end_field=8)),
            (4, 132, _metadata(2010, 2014, 80000)),
        ], modified=2000)
        bad = self.conversations / "broken-worker.db"
        bad.write_bytes(b"not a SQLite database")
        os.utime(bad, (3000, 3000))
        self._index([("main-id", 1005, "")])
        with self.assertLogs("tokenmon.adapters.antigravity", level="ERROR"):
            spans = self.adapter.collect(max_sessions=3)
        self.assertEqual([(span.session_id, span.turn_id, span.tokens) for span in spans],
                         [("main-id", "1", 120), ("worker-id", "3", 80)])
        self.assertEqual({session.session_id for session in self.adapter.collect_sessions(max_sessions=3)},
                         {"main-id", "worker-id"})
        self.assertIsNone(self.adapter.read_session("broken-worker"))

    def test_optional_payload_settings_and_summary_columns_do_not_hide_worker(self):
        self._conversation("worker-id", [(1, 15, _metadata(2000, 2004, 80))], payload=False)
        with closing(sqlite3.connect(self.root / "conversation_summaries.db")) as conn:
            conn.execute("CREATE TABLE conversation_summaries (conversation_id TEXT)")
            conn.execute("INSERT INTO conversation_summaries VALUES ('worker-id')")
            conn.commit()
        span = self.adapter.collect()[0]
        self.assertEqual(span.tokens, 80)
        self.assertIsNone(span.reasoning_effort)
        timeline = self.adapter.collect_sessions()[0]
        self.assertEqual(timeline.session_id, "worker-id")
        self.assertEqual(timeline.events[0].tokens, 80)
        self.assertIsNone(timeline.cwd)
        self.assertEqual(self.adapter.read_session("worker-id").events[0].event_id, "step:1:assistant")

    def test_pinned_worker_remains_exact_outside_order_limit_and_through_read_failure(self):
        path = self._conversation("worker-id", [(1, 15, _metadata(2000, 2004, 80))], modified=2000)
        initial = self.adapter.collect_sessions(max_sessions=1)[0]
        self._conversation("main-new", [(1, 15, _metadata(3000, 3004, 120))], modified=3000)
        self._conversation("worker-id-shadow", [(1, 15, _metadata(4000, 4004, 40))], modified=4000)
        self._index([("main-new", 3005, "")])
        self.assertEqual(self.adapter.collect_sessions(max_sessions=1)[0].session_id, "worker-id-shadow")
        self.assertEqual(self.adapter.read_session(initial.session_id), initial)
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("UPDATE steps SET metadata = ? WHERE idx = 1", (_metadata(2000, 2004, 100),))
            conn.execute("INSERT INTO steps VALUES (2, 15, ?, NULL)", (_metadata(2000, 2004, 40),))
            conn.commit()
        refreshed = self.adapter.read_session(initial.session_id)
        self.assertEqual([event.event_id for event in refreshed.events], ["step:1:assistant", "step:2:assistant"])
        self.assertEqual([event.tokens for event in refreshed.events], [100, 40])
        saved = path.read_bytes()
        hidden = path.with_suffix(".saved")
        path.rename(hidden)
        self.assertIsNone(self.adapter.read_session(initial.session_id))
        self.assertEqual(self.adapter.parse_session_timeline(initial.session_id).session_id, initial.session_id)
        self.assertEqual(self.adapter.parse_session_timeline(initial.session_id).events, [])
        hidden.rename(path)
        path.write_bytes(b"unreadable database")
        self.assertIsNone(self.adapter.read_session(initial.session_id))
        path.write_bytes(saved)
        self.assertEqual(self.adapter.read_session(initial.session_id), refreshed)
        self.assertIsNone(self.adapter.read_session("worker"))

    def test_live_unindexed_worker_wal_recency_is_visible_without_writing_sources(self):
        self._conversation("main-id", [(1, 15, _metadata(1000, 1004, 120))], modified=9000)
        worker = self._conversation("worker-id", [(1, 15, _metadata(2000, 2004, 80))], modified=100)
        self._index([("main-id", 1500, "")])
        with closing(sqlite3.connect(worker)) as writer:
            writer.execute("PRAGMA journal_mode = WAL")
            writer.execute("INSERT INTO steps VALUES (2, 15, ?, NULL)", (_metadata(2005, 2009, 40),))
            writer.commit()
            wal = worker.with_name(worker.name + "-wal")
            self.assertGreater(wal.stat().st_size, 0)
            os.utime(worker, (100, 100))
            os.utime(wal, (3000, 3000))
            original = {path: path.read_bytes() for path in (worker, wal)}
            spans = self.adapter.collect(max_sessions=1)
            self.assertEqual([(span.session_id, span.turn_id) for span in spans], [("worker-id", "1"), ("worker-id", "2")])
            timeline = self.adapter.collect_sessions(max_sessions=1)[0]
            self.assertEqual(timeline.session_id, "worker-id")
            self.assertEqual([event.tokens for event in timeline.events], [80, 40])
            self.assertEqual({path: path.read_bytes() for path in original}, original)

    def test_nonpositive_limits_do_not_open_sources(self):
        self._conversation("worker-id", [(1, 15, _metadata(2000, 2004, 80))])
        self._index([("worker-id", 2005, "main-id")])
        with patch("tokenmon.adapters.antigravity.open_ro_db", wraps=open_ro_db) as opened:
            for limit in (0, -1):
                self.assertEqual(self.adapter.collect(max_sessions=limit), [])
                self.assertEqual(self.adapter.collect_sessions(max_sessions=limit), [])
        self.assertEqual(opened.call_count, 0)


if __name__ == "__main__":
    unittest.main()
