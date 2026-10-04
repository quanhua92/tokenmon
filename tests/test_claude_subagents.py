"""Claude's native nested worker streams, using synthetic telemetry only."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokenmon.adapters.claude import ClaudeAdapter, parse_iso_timestamp


class TestClaudeSubagents(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name).resolve()
        self.adapter = ClaudeAdapter(self.root)

    def _records(self, tokens, parent="parent-one", worker=False):
        # Native workers use their parent's sessionId and their own agentId.
        common = {"sessionId": parent, "cwd": "/synthetic/worker" if worker else "/synthetic/main"}
        if worker:
            common.update(isSidechain=True, agentId="same")
        return [
            {**common, "type": "user", "uuid": "user-one", "timestamp": "2026-10-03T10:00:00Z",
             "message": {"role": "user", "content": "synthetic prompt"}},
            {**common, "type": "assistant", "uuid": "chunk-one", "timestamp": "2026-10-03T10:00:02Z",
             "message": {"id": "response-one", "model": "test-model", "role": "assistant",
                         "content": [{"type": "thinking"}, {"type": "tool_use", "id": "tool-one", "name": "SyntheticTool"}],
                         "output_config": {"effort": "high"}, "usage": {"output_tokens": 10, "service_tier": "standard", "speed": "fast"}}},
            {**common, "type": "assistant", "uuid": "chunk-two", "timestamp": "2026-10-03T10:00:04Z",
             "message": {"id": "response-one", "model": "test-model", "role": "assistant",
                         "content": [{"type": "thinking"}, {"type": "tool_use", "id": "tool-one", "name": "SyntheticTool"}],
                         "usage": {"output_tokens": tokens}}},
        ]

    def _write(self, relative, records, mtime=100):
        path = self.root / "projects" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        os.utime(path, (mtime, mtime))
        return path

    def _worker(self, parent, tokens=80, mtime=100, project="bucket"):
        return self._write(f"{project}/{parent}/subagents/agent-same.jsonl", self._records(tokens, parent, worker=True), mtime)

    def test_main_and_workers_have_independent_usage_identity_and_metadata(self):
        parent_records = self._records(40)
        parent_records.append({
            "type": "user", "sessionId": "parent-one", "uuid": "worker-result", "timestamp": "2026-10-03T10:00:05Z",
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-one", "content": "synthetic result"}]},
            "toolUseResult": {"agentId": "same", "totalTokens": 999999, "totalDurationMs": 999999,
                              "usage": {"output_tokens": 999999}},
        })
        self._write("bucket/parent-one.jsonl", parent_records)
        self._worker("parent-one", 80)
        self._worker("parent-two", 120)
        spans = self.adapter.collect()
        sessions = {session.session_id: session for session in self.adapter.collect_sessions()}
        expected_ids = {"parent-one", "agent-same@bucket/parent-one", "agent-same@bucket/parent-two"}
        self.assertEqual(set(sessions), expected_ids)
        self.assertEqual({span.session_id for span in spans}, expected_ids)
        self.assertEqual(sorted(span.tokens for span in spans), [40, 80, 120])
        self.assertTrue(all(span.is_valid for span in spans))
        self.assertEqual(sum(span.tokens for span in spans), 240)
        self.assertEqual(sum(span.duration for span in spans), 6)
        self.assertEqual(sum(span.tokens for span in spans) / sum(span.duration for span in spans), 40)
        for span in spans:
            session = sessions[span.session_id]
            self.assertEqual(session.total_tokens, span.tokens)
            self.assertEqual(session.assistant_messages, 1)
            self.assertEqual(session.tool_calls, 1)
            self.assertEqual(sum(event.kind == "reasoning" for event in session.events), 1)
            self.assertEqual(span.timing_source, "chunk-stream")
            self.assertEqual(session.model, "test-model")
            self.assertEqual(session.reasoning_effort, "high")
            self.assertEqual(session.service_tier, "standard")
            self.assertEqual(session.speed, "fast")
            self.assertEqual(span.reasoning_effort, "high")
            for event in session.events:
                if event.kind in {"assistant_message", "reasoning", "tool_call"}:
                    self.assertEqual(event.reasoning_effort, "high")
                    self.assertEqual(event.service_tier, "standard")
                    self.assertEqual(event.speed, "fast")
        self.assertEqual(sessions["parent-one"].cwd, "/synthetic/main")
        self.assertEqual(sessions["agent-same@bucket/parent-one"].cwd, "/synthetic/worker")
        self.assertEqual(self.adapter.read_session("agent-same@bucket/parent-two").total_tokens, 120)

    def test_distinct_worker_responses_with_matching_times_and_usage_are_not_deduplicated(self):
        records = self._records(80, worker=True)
        second_response = self._records(80, worker=True)[1:]
        for record in second_response:
            record["uuid"] = "second-" + record["uuid"]
            record["message"]["id"] = "response-two"
        self._write("bucket/parent-one/subagents/agent-same.jsonl", records + second_response)
        spans = self.adapter.collect()
        session = self.adapter.collect_sessions()[0]
        self.assertEqual({span.turn_id for span in spans}, {"response-one", "response-two"})
        self.assertEqual([span.tokens for span in spans], [80, 80])
        self.assertTrue(all(span.is_valid for span in spans))
        self.assertEqual(session.total_tokens, 160)
        self.assertEqual(session.assistant_messages, 2)
        self.assertEqual(session.tool_calls, 2)

    def test_worker_identity_is_scoped_by_project_as_well_as_parent(self):
        self._worker("parent-one", 80, project="bucket-one")
        self._worker("parent-one", 120, project="bucket-two")
        sessions = self.adapter.collect_sessions()
        self.assertEqual({s.session_id for s in sessions}, {
            "agent-same@bucket-one/parent-one", "agent-same@bucket-two/parent-one"})
        self.assertEqual(self.adapter.read_session("agent-same@bucket-one/parent-one").total_tokens, 80)
        self.assertEqual(self.adapter.read_session("agent-same@bucket-two/parent-one").total_tokens, 120)
        self.assertIsNone(self.adapter.read_session("parent-one"))

    def test_shared_limits_are_newest_first_with_deterministic_ties(self):
        self._write("bucket/main.jsonl", self._records(40), mtime=100)
        self._worker("parent-one", 80, mtime=200)
        self._worker("parent-two", 120, mtime=200)
        self.assertEqual([s.session_id for s in self.adapter.collect_sessions(max_sessions=1)], ["agent-same@bucket/parent-one"])
        self.assertEqual({s.session_id for s in self.adapter.collect(max_sessions=2)}, {
            "agent-same@bucket/parent-one", "agent-same@bucket/parent-two"})
        for limit in (0, -1):
            self.assertEqual(self.adapter.collect(max_sessions=limit), [])
            self.assertEqual(self.adapter.collect_sessions(max_sessions=limit), [])

    def test_cutoff_uses_recorded_completion_and_retains_preceding_context(self):
        path = self._worker("parent-one", 80, mtime=1)
        span = self.adapter.collect()[0]
        cutoff = parse_iso_timestamp("2026-10-03T10:00:03Z")
        self.assertEqual(self.adapter.collect(min_timestamp=cutoff), [span])
        self.assertEqual(span.duration, 2)
        self.assertEqual(span.reasoning_effort, "high")
        self.assertEqual(self.adapter.collect(min_timestamp=span.ended_at), [span])
        self.assertEqual(self.adapter.collect(min_timestamp=span.ended_at + 0.001), [])
        sessions = self.adapter.collect_sessions(min_timestamp=span.ended_at)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].user_messages, 1)
        self.assertEqual(sessions[0].total_tokens, 80)
        os.utime(path, (span.ended_at + 100, span.ended_at + 100))
        self.assertEqual(self.adapter.collect_sessions(min_timestamp=span.ended_at + 0.001), [])

    def test_pinned_worker_refresh_survives_limits_missing_source_and_final_update(self):
        path = self._worker("parent-one", 80)
        session = self.adapter.collect_sessions(max_sessions=1)[0]
        event_ids = [event.event_id for event in session.events]
        self._worker("parent-two", 120, mtime=200)
        self._write("bucket/new-main.jsonl", self._records(40), mtime=300)
        self.assertEqual([s.session_id for s in self.adapter.collect_sessions(max_sessions=1)], ["new-main"])
        self.assertEqual(self.adapter.read_session(session.session_id).total_tokens, 80)
        hidden = path.with_suffix(".hidden")
        path.rename(hidden)
        self.assertIsNone(self.adapter.read_session(session.session_id))
        hidden.rename(path)
        update = self._records(100, worker=True)[-1]
        update["message"]["output_config"] = {"effort": "medium"}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(update) + "\n")
        refreshed = self.adapter.read_session(session.session_id)
        self.assertEqual(refreshed.total_tokens, 100)
        self.assertEqual(refreshed.assistant_messages, 1)
        self.assertEqual(refreshed.reasoning_effort, "medium")
        self.assertEqual([event.event_id for event in refreshed.events], event_ids)
        spans = {span.session_id: span for span in self.adapter.collect(max_sessions=3)}
        self.assertEqual(spans[session.session_id].tokens, 100)

    def test_collect_also_retains_exact_worker_for_pinning(self):
        self._worker("parent-one", 80)
        worker_id = self.adapter.collect()[0].session_id
        self._worker("parent-two", 120, mtime=200)
        self.adapter.collect(max_sessions=1)
        self.assertEqual(self.adapter.read_session(worker_id).total_tokens, 80)
        fresh_adapter = ClaudeAdapter(self.root)
        self.assertEqual(fresh_adapter.read_session(worker_id).total_tokens, 80)

    def test_unreadable_worker_does_not_drop_siblings_or_forget_pin(self):
        path = self._worker("parent-one", 80)
        self._worker("parent-two", 120)
        worker_id = "agent-same@bucket/parent-one"
        self.adapter.collect_sessions()
        original_open = Path.open

        def fail_worker(candidate, *args, **kwargs):
            if candidate == path:
                raise PermissionError("synthetic unreadable worker")
            return original_open(candidate, *args, **kwargs)

        with patch.object(Path, "open", fail_worker):
            self.assertIsNone(self.adapter.read_session(worker_id))
            self.assertEqual([s.session_id for s in self.adapter.collect_sessions()], ["agent-same@bucket/parent-two"])
            self.assertEqual([s.tokens for s in self.adapter.collect()], [120])
        self.assertEqual(self.adapter.read_session(worker_id).total_tokens, 80)

    def test_disappearing_file_stat_does_not_drop_siblings(self):
        missing = self._worker("parent-one", 80)
        self._worker("parent-two", 120)
        original_stat = Path.stat

        def fail_worker(candidate, *args, **kwargs):
            if candidate == missing:
                raise FileNotFoundError("synthetic disappearing worker")
            return original_stat(candidate, *args, **kwargs)

        with patch.object(Path, "stat", fail_worker):
            self.assertEqual([s.tokens for s in self.adapter.collect()], [120])
            self.assertEqual([s.session_id for s in self.adapter.collect_sessions()], ["agent-same@bucket/parent-two"])

    def test_directory_symlinks_and_non_native_sources_are_not_traversed(self):
        self._worker("parent-one", 80)
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside)
            (external / "external.jsonl").write_text(json.dumps(self._records(120)[-1]), encoding="utf-8")
            (self.root / "projects" / "linked-project").symlink_to(external, target_is_directory=True)
            (self.root / "projects" / "bucket" / "linked-parent").symlink_to(external, target_is_directory=True)
            (self.root / "projects" / "bucket" / "linked.jsonl").symlink_to(external / "external.jsonl")
            self._write("bucket/.hidden.jsonl", self._records(120))
            self._write("bucket/parent-one/subagents/.hidden.jsonl", self._records(120))
            self._write("bucket/parent-one/subagents/unrelated.jsonl", self._records(120))
            self._write("bucket/parent-one/other/agent-wrong.jsonl", self._records(120))
            self._write("bucket/parent-one/subagents/deeper/agent-wrong.jsonl", self._records(120))
            empty = self.root / "projects" / "bucket" / "empty.jsonl"
            empty.touch()
            self.assertEqual([s.tokens for s in self.adapter.collect()], [80])
            self.assertEqual([s.session_id for s in self.adapter.collect_sessions()], ["agent-same@bucket/parent-one"])

    def test_malformed_worker_records_and_truncated_tail_keep_valid_siblings(self):
        valid = self._records(80, worker=True)
        junk = [None, [], 42, {"type": "assistant", "timestamp": "2026-10-03T10:00:01Z", "message": None},
                {"type": "assistant", "timestamp": "2026-10-03T10:00:01Z", "message": {"usage": []}},
                {"type": "assistant", "timestamp": "2026-10-03T10:00:01", "message": {"usage": {"output_tokens": 999999}}}]
        path = self._write("bucket/parent-one/subagents/agent-same.jsonl", [valid[0], *junk, *valid[1:]])
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{"type":"assistant"')
        self._write("bucket/broken/subagents/agent-broken.jsonl", [None, []])
        self._worker("parent-two", 120)
        self.assertEqual(sorted(s.tokens for s in self.adapter.collect()), [80, 120])
        self.assertEqual(sorted(s.total_tokens for s in self.adapter.collect_sessions()), [80, 120])
        self.assertIsNone(self.adapter.read_session("agent-broken@bucket/broken"))


if __name__ == "__main__":
    unittest.main()
