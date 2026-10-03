"""Tests for Antigravity (agy) adapter."""

import sqlite3
import unittest
from pathlib import Path
import tempfile
import shutil

from llm_monitor.adapters.antigravity import (
    AntigravityAdapter,
    parse_proto_fields,
    parse_proto_timestamp,
)


class TestAntigravityAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.conv_dir = self.temp_dir / "conversations"
        self.conv_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_proto_varint_and_fields(self):
        # Varint encoding of 150 is 0x96 0x01
        data = bytes([0x08, 0x96, 0x01])  # field 1, varint 150
        fields = parse_proto_fields(data)
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0], (1, "varint", 150))

    def test_detect_false_when_empty(self):
        adapter = AntigravityAdapter(root=self.temp_dir / "nonexistent")
        self.assertFalse(adapter.detect())

    def test_detect_true_when_conv_exists(self):
        adapter = AntigravityAdapter(root=self.temp_dir)
        self.assertTrue(adapter.detect())

    def test_collect_empty_session(self):
        db_path = self.conv_dir / "test_session.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE steps (idx integer, step_type integer, metadata blob);")
        conn.commit()
        conn.close()

        adapter = AntigravityAdapter(root=self.temp_dir)
        spans = adapter.collect()
        self.assertEqual(len(spans), 0)

        timelines = adapter.collect_sessions()
        self.assertEqual(len(timelines), 0)

    def _make_summary_db(self, rows):
        conn = sqlite3.connect(str(self.temp_dir / "conversation_summaries.db"))
        conn.execute("CREATE TABLE conversation_summaries (conversation_id text, workspace_uris text);")
        conn.executemany("INSERT INTO conversation_summaries VALUES (?, ?)", rows)
        conn.commit()
        conn.close()

    def test_workspace_path_from_summaries(self):
        self._make_summary_db(
            [
                ("a", '["file:///tmp/my%20project"]'),
                ("b", ""),
                ("c", "not json"),
                ("d", '["https://example.com", "file:///tmp/second"]'),
            ]
        )
        adapter = AntigravityAdapter(root=self.temp_dir)
        self.assertEqual(adapter._workspace_path("a"), "/tmp/my project")  # URL-decoded
        self.assertIsNone(adapter._workspace_path("b"))  # empty value
        self.assertIsNone(adapter._workspace_path("c"))  # unparseable value
        self.assertEqual(adapter._workspace_path("d"), "/tmp/second")  # skips non-file URIs
        self.assertIsNone(adapter._workspace_path("missing"))  # no row

    def test_workspace_path_without_summary_db(self):
        adapter = AntigravityAdapter(root=self.temp_dir)
        self.assertIsNone(adapter._workspace_path("anything"))

    def test_session_cutoff_uses_event_time_and_keeps_boundary(self):
        # Minimal Timestamp protos: seconds 1000 and 2000, wrapped in metadata field 1.
        for session_id, timestamp in [("older", b"\x08\xe8\x07"), ("boundary", b"\x08\xd0\x0f")]:
            with sqlite3.connect(self.conv_dir / f"{session_id}.db") as conn:
                conn.execute("CREATE TABLE steps (idx integer, step_type integer, metadata blob, step_payload blob)")
                conn.execute("INSERT INTO steps VALUES (?, ?, ?, ?)", (1, 14, b"\x0a\x03" + timestamp, None))
        adapter = AntigravityAdapter(root=self.temp_dir)
        self.assertEqual(len(adapter.collect_sessions()), 2)
        self.assertEqual([timeline.session_id for timeline in adapter.collect_sessions(min_timestamp=2000)], ["boundary"])


if __name__ == "__main__":
    unittest.main()
