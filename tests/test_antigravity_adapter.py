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


if __name__ == "__main__":
    unittest.main()
