"""Native Codex worker ownership, discovery and pinned-read regressions."""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from tokenmon.adapters.codex import CodexAdapter
from tokenmon.analyzer import summarize_spans


class TestCodexSubagents(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.started = datetime(2026, 10, 3, 10, tzinfo=timezone.utc).timestamp()

    def record(self, kind, payload, offset=0):
        return {"type": kind, "payload": payload,
                "timestamp": datetime.fromtimestamp(self.started + offset, timezone.utc).isoformat()}

    def header(self, session_id, *, parent=None, **metadata):
        payload = {"id": session_id, "session_id": "root-session-id", "cwd": "/synthetic/project",
                   "source": "cli" if parent is None else {
                       "subagent": {"thread_spawn": {"parent_thread_id": parent, "depth": 1}}}}
        payload.update(metadata)
        return self.record("session_meta", payload)

    def generation(self, item_id, tokens, offset=0, *, cumulative=None):
        start_ms = int((self.started + offset) * 1000)
        return [
            self.record("turn_context", {"turn_id": item_id, "model": "test-model"}, offset),
            self.record("event_msg", {"type": "task_started", "turn_id": item_id}, offset),
            self.record("response_item", {"type": "message", "role": "assistant", "id": item_id}, offset + 2),
            self.record("event_msg", {"type": "item_completed", "item": {
                "type": "AgentMessage", "id": item_id}, "started_at_ms": start_ms,
                "completed_at_ms": start_ms + 2000}, offset + 2),
            self.record("event_msg", {"type": "token_count", "info": {
                "last_token_usage": {"output_tokens": tokens},
                "total_token_usage": {"output_tokens": tokens if cumulative is None else cumulative}}}, offset + 2),
        ]

    def write(self, name, records, *, mtime=None):
        path = self.root / "sessions" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    def index(self, rows):
        with closing(sqlite3.connect(self.root / "state_1.sqlite")) as conn:
            conn.execute("CREATE TABLE threads (id TEXT, rollout_path TEXT, model TEXT, archived INTEGER, updated_at INTEGER)")
            conn.executemany("INSERT INTO threads VALUES (?, ?, ?, ?, ?)",
                             [(session_id, str(path), "indexed-model", archived, updated)
                              for session_id, path, archived, updated in rows])
            conn.commit()

    def test_indexed_main_and_unindexed_worker_count_output_once(self):
        main_records = [self.header("main-id")] + self.generation("main-output", 120)
        # Parent tool results may carry worker usage, but are not a response generation.
        main_records.append(self.record("response_item", {"type": "function_call_output",
            "call_id": "spawn-call", "usage": {"output_tokens": 80000}}, 15))
        main = self.write("rollout-main.jsonl", main_records, mtime=self.started + 15)
        self.write("nested/rollout-worker.jsonl", [self.header("worker-id", parent="main-id")]
                   + self.generation("worker-output", 80, 10), mtime=self.started + 20)
        archived = self.write("rollout-archived.jsonl", [self.header("archived-id")]
                              + self.generation("archived-output", 200, 20), mtime=self.started + 30)
        self.index([("main-id", main, 0, int(self.started + 15)),
                    ("archived-id", archived, 1, int(self.started + 30))])
        adapter = CodexAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual([(s.session_id, s.tokens, s.duration) for s in spans],
                         [("main-id", 120, 2.0), ("worker-id", 80, 2.0)])
        summary = summarize_spans(spans, "all", "test-model")
        self.assertEqual(summary.total_tokens, 200)
        self.assertEqual(summary.weighted_tps, 50.0)
        self.assertEqual({t.session_id: t.total_tokens for t in adapter.collect_sessions()},
                         {"main-id": 120, "worker-id": 80})
        self.assertEqual([s.session_id for s in adapter.collect(max_sessions=1)], ["worker-id"])
        self.assertEqual(adapter.collect(max_sessions=0), [])
        self.assertEqual(adapter.collect_sessions(max_sessions=-1), [])

    def test_rollout_names_and_root_session_ids_do_not_replace_worker_identity(self):
        for index, session_id in enumerate(("worker-a", "worker-b")):
            self.write(f"parent-{index}/rollout-scout.jsonl", [self.header(session_id, parent=f"parent-{index}")]
                       + self.generation(f"output-{index}", 80, 10), mtime=self.started + index)
        adapter = CodexAdapter(self.root)
        self.assertEqual({s.session_id for s in adapter.collect()}, {"worker-a", "worker-b"})
        self.assertEqual([t.session_id for t in adapter.collect_sessions(max_sessions=1)], ["worker-b"])
        self.assertEqual(adapter.read_session("worker-a").session_id, "worker-a")
        self.assertIsNone(adapter.read_session("root-session-id"))

    def test_ordinal_boundary_excludes_restamped_parent_activity_and_counters(self):
        copied = self.generation("parent-output", 80)
        copied.insert(1, self.record("event_msg", {"type": "thread_settings_applied",
            "thread_settings": {"reasoning_effort": "high", "service_tier": "priority"}}))
        boundary = 1 + len(copied)
        owned = self.generation("worker-output", 80, 10)
        records = [self.header("worker-id", parent="main-id", forked_from_id="main-id",
                               history_mode="paginated", subagent_history_start_ordinal=boundary)] + copied + owned
        # The native recorder restamps copied records. Ownership must not use their dates.
        for ordinal, record in enumerate(records):
            record["ordinal"] = ordinal
            if ordinal < boundary:
                record["timestamp"] = records[-1]["timestamp"]
        self.write("rollout-worker.jsonl", records)
        adapter = CodexAdapter(self.root)
        spans = adapter.collect()
        self.assertEqual([(s.session_id, s.tokens, s.turn_id) for s in spans],
                         [("worker-id", 80, "worker-output")])
        self.assertTrue(spans[0].is_valid)
        self.assertEqual(spans[0].duration, 2.0)
        self.assertEqual(spans[0].reasoning_effort, "high")
        self.assertEqual(spans[0].service_tier, "priority")
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.total_tokens, 80)
        self.assertEqual(timeline.assistant_messages, 1)
        self.assertEqual(timeline.user_messages, 1)
        self.assertTrue(all(e.turn_id == "worker-output" for e in timeline.events))
        self.assertEqual(timeline.created_at, self.started + 10)

    def test_unconfirmed_record_ordinals_cannot_charge_inherited_output(self):
        copied = self.generation("unconfirmed-output", 120)
        for record in copied:
            record["ordinal"] = True
        owned = self.generation("worker-output", 80, 10)
        for ordinal, record in enumerate(owned, 20):
            record["ordinal"] = ordinal
        records = [self.header("worker-id", parent="main-id",
                               subagent_history_start_ordinal=20)] + copied + owned
        self.write("rollout-worker.jsonl", records)
        adapter = CodexAdapter(self.root)
        self.assertEqual([s.tokens for s in adapter.collect()], [80])
        self.assertEqual(adapter.collect_sessions()[0].total_tokens, 80)

    def test_parent_association_without_copy_boundary_does_not_remove_output(self):
        records = [self.header("worker-id", parent="main-id", parent_thread_id="main-id",
                               timestamp="2026-10-03T10:01:00Z")] + self.generation("worker-output", 80)
        self.write("rollout-worker.jsonl", records)
        self.assertEqual(CodexAdapter(self.root).collect()[0].tokens, 80)
        self.assertEqual(CodexAdapter(self.root).collect_sessions()[0].total_tokens, 80)

    def test_legacy_copied_fork_without_boundary_is_not_measured(self):
        records = [self.header("worker-id", parent="main-id", forked_from_id="main-id")]
        records += self.generation("ambiguous-output", 120)
        self.write("rollout-worker.jsonl", records)
        adapter = CodexAdapter(self.root)
        with self.assertLogs("tokenmon.adapters.codex", level="WARNING"):
            self.assertEqual(adapter.collect(), [])
            self.assertEqual(adapter.collect_sessions(), [])
        with self.assertLogs("tokenmon.adapters.codex", level="WARNING"):
            timeline = adapter.read_session("worker-id")
        self.assertEqual(timeline.session_id, "worker-id")
        self.assertEqual(timeline.total_tokens, 0)
        self.assertEqual(timeline.events, [])

    def test_cutoff_uses_completion_not_source_freshness(self):
        main = self.write("rollout-main.jsonl", [self.header("main-id")]
                          + self.generation("main-output", 120), mtime=self.started - 100)
        self.write("rollout-worker.jsonl", [self.header("worker-id", parent="main-id")]
                   + self.generation("worker-output", 80, 10), mtime=self.started - 100)
        self.index([("main-id", main, 0, int(self.started - 100))])
        adapter = CodexAdapter(self.root)
        cutoff = self.started + 12
        spans = adapter.collect(min_timestamp=cutoff)
        self.assertEqual([(s.session_id, s.tokens, s.duration) for s in spans], [("worker-id", 80, 2.0)])
        self.assertEqual([t.session_id for t in adapter.collect_sessions(min_timestamp=cutoff)], ["worker-id"])
        self.assertEqual(adapter.collect(min_timestamp=cutoff + 0.01), [])
        self.assertEqual(adapter.collect_sessions(min_timestamp=cutoff + 0.01), [])

    def test_pinned_worker_survives_recent_bounds_disappearance_and_partial_append(self):
        records = [self.header("worker-id", parent="main-id")] + self.generation("worker-output", 80)
        path = self.write("rollout-worker.jsonl", records, mtime=self.started)
        adapter = CodexAdapter(self.root)
        baseline = adapter.collect_sessions()[0]
        for number in range(40):
            self.write(f"newer-{number}.jsonl", [self.header(f"newer-{number}")]
                       + self.generation(f"output-{number}", 20, 20), mtime=self.started + 50 + number)
        self.assertNotEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "worker-id")
        self.assertEqual(adapter.read_session("worker-id").events, baseline.events)
        # A fresh adapter resolves an exact older native ID without the recent-session cap.
        self.assertEqual(CodexAdapter(self.root).read_session("worker-id").events, baseline.events)
        hidden = path.with_suffix(".hidden")
        path.rename(hidden)
        self.assertIsNone(adapter.read_session("worker-id"))
        hidden.rename(path)
        new_record = self.record("response_item", {"type": "function_call", "name": "synthetic_tool", "id": "new-call"}, 2)
        line = json.dumps(new_record)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line[:len(line) // 2])
        self.assertEqual(adapter.read_session("worker-id").events, baseline.events)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line[len(line) // 2:] + "\n")
        refreshed = adapter.read_session("worker-id")
        self.assertEqual(len(refreshed.events), len(baseline.events) + 1)
        self.assertEqual(refreshed.events[-1].kind, "tool_call")
        self.assertEqual([e.event_id for e in refreshed.events[:-1]], [e.event_id for e in baseline.events])
        path.write_text(json.dumps(self.header("replacement-id")) + "\n", encoding="utf-8")
        self.assertIsNone(adapter.read_session("worker-id"))

    def test_worker_diagnostics_keep_start_before_cutoff_and_cumulative_duplicates(self):
        records = [self.header("worker-id", parent="main-id")] + self.generation("worker-output", 80)
        # A duplicated cumulative snapshot must not consume the next reasoning output.
        records += [self.record("response_item", {"type": "reasoning", "id": "next-reasoning"}, 4),
                    {**records[-1], "timestamp": self.record("unused", {}, 4)["timestamp"]}]
        records += self.generation("next-output", 120, 4, cumulative=200)[2:]
        self.write("rollout-worker.jsonl", records)
        with closing(sqlite3.connect(self.root / "logs_1.sqlite")) as conn:
            conn.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, target TEXT, ts INTEGER, ts_nanos INTEGER, feedback_log_body TEXT)")
            conn.executemany("INSERT INTO logs VALUES (?, ?, ?, ?, ?)", [
                (1, "codex_core::stream_events_utils", int(self.started), 125000000,
                 'Output item item_type="message" item_id="worker-output"'),
                (2, "codex_core::stream_events_utils", int(self.started + 4), 0,
                 'Output item item_type="reasoning" item_id="next-reasoning"'),
            ])
            conn.commit()
        adapter = CodexAdapter(self.root)
        spans = adapter.collect(min_timestamp=self.started + 1)
        self.assertEqual([s.tokens for s in spans], [80, 120])
        self.assertEqual([s.timing_source for s in spans], ["stream-log", "stream-log"])
        self.assertEqual([s.duration for s in spans], [1.875, 2.0])
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.total_tokens, 200)
        self.assertTrue(all(e.tokens is None for e in timeline.events if e.kind == "reasoning"))

    def test_native_worker_id_is_selectable_in_actual_cli_json(self):
        main = self.write("rollout-main.jsonl", [self.header("main-id")] + self.generation("main-output", 120))
        self.write("nested/rollout-worker.jsonl", [self.header("worker-id", parent="main-id")]
                   + self.generation("worker-output", 80, 10))
        self.index([("main-id", main, 0, int(self.started + 2))])
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                           PYTHONDONTWRITEBYTECODE="1")
        for arguments in (["ps", "codex"], ["logs", "worker-id", "--agent", "codex"]):
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, "-B", "-m", "tokenmon", *arguments,
                                         "--home", str(self.root), "--all", "--json"],
                                        capture_output=True, text=True, env=environment, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout)
                if arguments[0] == "ps":
                    self.assertEqual({t["session_id"]: t["total_tokens"] for t in value},
                                     {"main-id": 120, "worker-id": 80})
                else:
                    self.assertEqual(value["session_id"], "worker-id")
                    self.assertEqual(value["total_tokens"], 80)
                    self.assertTrue(all("event_id" not in e for e in value["events"]))
