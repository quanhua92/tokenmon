"""Tests for Antigravity (agy) adapter."""

import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
import tempfile
import shutil

from tokenmon.adapters.antigravity import (
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
            with closing(sqlite3.connect(self.conv_dir / f"{session_id}.db")) as conn:
                conn.execute("CREATE TABLE steps (idx integer, step_type integer, metadata blob, step_payload blob)")
                conn.execute("INSERT INTO steps VALUES (?, ?, ?, ?)", (1, 14, b"\x0a\x03" + timestamp, None))
                conn.commit()
        adapter = AntigravityAdapter(root=self.temp_dir)
        self.assertEqual(len(adapter.collect_sessions()), 2)
        self.assertEqual([timeline.session_id for timeline in adapter.collect_sessions(min_timestamp=2000)], ["boundary"])

    @staticmethod
    def _varint(value):
        result = bytearray()
        while value > 127:
            result.append((value & 127) | 128)
            value >>= 7
        result.append(value)
        return bytes(result)

    def _field(self, number, value):
        if isinstance(value, int):
            return self._varint(number << 3) + self._varint(value)
        return self._varint((number << 3) | 2) + self._varint(len(value)) + value

    def _metadata(self, start, end, tokens, end_field=7):
        return (self._field(1, self._field(1, start)) +
                self._field(end_field, self._field(1, end)) +
                self._field(9, self._field(3, tokens)))

    def test_model_mapping_carries_forward_until_next_boundary(self):
        with closing(sqlite3.connect(self.conv_dir / "models.db")) as conn:
            conn.execute("CREATE TABLE steps (idx integer, step_type integer, metadata blob)")
            conn.execute("CREATE TABLE gen_metadata (data blob)")
            for last_index, model in [(0, "model-a"), (3, "model-b")]:
                entry = self._field(19, model.encode()) + self._field(20,
                    self._field(1, b"last_step_index") + self._field(2, str(last_index).encode()))
                conn.execute("INSERT INTO gen_metadata VALUES (?)", (self._field(1, entry),))
            for index in (1, 2, 4, 5):
                conn.execute("INSERT INTO steps VALUES (?, ?, ?)", (index, 15, self._metadata(1000, 1010, 200)))
            conn.commit()
        spans = AntigravityAdapter(self.temp_dir).collect()
        self.assertEqual([s.model for s in spans], ["model-a", "model-a", "model-b", "model-b"])
        self.assertTrue(all(s.tps == 20 for s in spans))

    def test_invalid_spans_remain_counted_and_later_good_steps_survive(self):
        records = [
            (1, 15, 123),  # malformed nested object, not protobuf bytes
            (2, 15, b"\x0a\xff"),  # truncated protobuf
            (3, 15, self._metadata(1000, 999, 100)),
            (4, 15, self._metadata(1000, 1010, 0)),
            (5, 15, self._metadata(1000, 1001, 401)),
            (6, 132, self._metadata(1000, 1010, 500)),  # tools are not generation
            (7, 15, self._metadata(1000, 1002, 500)),  # evidenced 250 TPS is allowed
            (8, 15, self._metadata(2000, 2002, 200, end_field=8)),
        ]
        with closing(sqlite3.connect(self.conv_dir / "validation.db")) as conn:
            conn.execute("CREATE TABLE steps (idx integer, step_type integer, metadata blob)")
            conn.executemany("INSERT INTO steps VALUES (?, ?, ?)", records)
            conn.commit()
        adapter = AntigravityAdapter(self.temp_dir)
        spans = adapter.collect()
        self.assertEqual(len(spans), 5)
        self.assertCountEqual([s.note for s in spans if not s.is_valid],
                         ["invalid_duration", "zero_tokens", "unconfirmed_boundary_tps"])
        self.assertEqual([s.tps for s in spans if s.is_valid], [250, 100])
        self.assertEqual([s.turn_id for s in adapter.collect(min_timestamp=2002)], ["8"])

    def test_unusable_field_seven_falls_back_to_field_eight(self):
        metadata = self._metadata(1000, 1002, 200, end_field=8) + self._field(7, b"invalid")
        with closing(sqlite3.connect(self.conv_dir / "fallback.db")) as conn:
            conn.execute("CREATE TABLE steps (idx integer, step_type integer, metadata blob)")
            conn.execute("INSERT INTO steps VALUES (?, ?, ?)", (1, 15, metadata))
            conn.commit()
        span = AntigravityAdapter(self.temp_dir).collect()[0]
        self.assertEqual(span.duration, 2)
        self.assertEqual(span.tps, 100)

    def test_effort_metadata_is_optional_and_follows_model_boundary(self):
        with closing(sqlite3.connect(self.conv_dir / "effort.db")) as conn:
            conn.execute("CREATE TABLE steps (idx integer, step_type integer, metadata blob, step_payload blob)")
            conn.execute("CREATE TABLE gen_metadata (data blob)")
            for last_index, model, effort in [(0, "model-a", "high"), (3, "model-b", None)]:
                entry = self._field(19, model.encode()) + self._field(20,
                    self._field(1, b"last_step_index") + self._field(2, str(last_index).encode()))
                if effort:
                    entry += self._field(20, self._field(1, b"reasoning_effort") + self._field(2, effort.encode()))
                conn.execute("INSERT INTO gen_metadata VALUES (?)", (self._field(1, entry),))
            for index in (1, 2, 4, 5):
                conn.execute("INSERT INTO steps VALUES (?, ?, ?, ?)", (index, 15, self._metadata(1000, 1010, 200), None))
            conn.commit()
        spans = AntigravityAdapter(self.temp_dir).collect()
        self.assertEqual([s.reasoning_effort for s in spans], ["high", "high", None, None])
        self.assertTrue(all(s.speed_mode is None for s in spans))
        timeline = AntigravityAdapter(self.temp_dir).collect_sessions()[0]
        self.assertEqual([e.reasoning_effort for e in timeline.events], ["high", "high", None, None])
        self.assertIsNone(timeline.reasoning_effort)
        self.assertEqual(timeline.model, "model-b")


if __name__ == "__main__":
    unittest.main()
