"""Synthetic native OpenCode storage contracts; no installed OpenCode required."""

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from tokenmon.adapters.opencode import OpenCodeAdapter


T = 1_800_000_000_000
FORMATS = ("json", "v1", "v2")


class NativeStore:
    """Write native objects, not an adapter-specific interchange format."""

    def __init__(self, root, format, filename="opencode.db"):
        self.root = Path(root)
        self.format = format
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / filename
        if format == "json":
            return
        with closing(sqlite3.connect(self.db)) as connection, connection:
            if format == "v2":
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS session_v2 (
                        id TEXT PRIMARY KEY, directory TEXT,
                        time_created INTEGER, time_updated INTEGER, model TEXT,
                        fork_session_id TEXT, fork_boundary TEXT, parent_id TEXT,
                        metadata TEXT);
                    CREATE TABLE IF NOT EXISTS session_message (
                        id TEXT PRIMARY KEY, session_id TEXT, type TEXT, seq INTEGER,
                        time_created INTEGER, time_updated INTEGER, data TEXT);
                """)
            else:
                connection.executescript("""
                    CREATE TABLE IF NOT EXISTS session (
                        id TEXT PRIMARY KEY, directory TEXT,
                        time_created INTEGER, time_updated INTEGER, parent_id TEXT);
                    CREATE TABLE IF NOT EXISTS message (
                        id TEXT PRIMARY KEY, session_id TEXT,
                        time_created INTEGER, time_updated INTEGER, data TEXT);
                    CREATE TABLE IF NOT EXISTS part (
                        id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
                        time_created INTEGER, time_updated INTEGER, data TEXT);
                """)

    def _json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def session(self, id="s", created=T, updated=T + 10_000,
                directory="/recorded/project", parent=None, model=None,
                fork=None, boundary=None):
        if self.format == "json":
            return self._json(self.root / "storage/session/project" / (id + ".json"), {
                "id": id, "projectID": "project", "directory": directory,
                "time": {"created": created, "updated": updated},
                **({"parentID": parent} if parent else {}),
            })
        with closing(sqlite3.connect(self.db)) as connection, connection:
            if self.format == "v2":
                connection.execute("INSERT OR REPLACE INTO session_v2 VALUES (?,?,?,?,?,?,?,?,?)",
                                   (id, directory, created, updated,
                                    json.dumps(model) if model else None,
                                    fork, json.dumps(boundary) if boundary else None,
                                    parent, None))
            else:
                connection.execute("INSERT OR REPLACE INTO session VALUES (?,?,?,?,?)",
                                   (id, directory, created, updated, parent))
        return self.db

    def message(self, id, data, session="s", kind=None, seq=1, row_time=T + 1_000_000):
        if self.format == "json":
            payload = {**data, "id": id, "sessionID": session} if isinstance(data, dict) else data
            return self._json(self.root / "storage/message" / session / (id + ".json"), payload)
        with closing(sqlite3.connect(self.db)) as connection, connection:
            if self.format == "v2":
                connection.execute("INSERT OR REPLACE INTO session_message VALUES (?,?,?,?,?,?,?)",
                                   (id, session, kind or data.get("role", "assistant"), seq,
                                    row_time, row_time, json.dumps(data)))
            else:
                connection.execute("INSERT OR REPLACE INTO message VALUES (?,?,?,?,?)",
                                   (id, session, row_time, row_time, json.dumps(data)))
        return self.db

    def part(self, id, message, data, session="s", row_time=T + 1_000_000):
        if self.format == "json":
            return self._json(self.root / "storage/part" / message / (id + ".json"),
                              {**data, "id": id, "messageID": message, "sessionID": session})
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT OR REPLACE INTO part VALUES (?,?,?,?,?,?)",
                               (id, message, session, row_time, row_time, json.dumps(data)))
        return self.db

    def user(self, id="u", session="s", created=T + 1_000, seq=1,
             model=None, variant="arbitrary-selector"):
        data = {"time": {"created": created}, "variant": variant}
        if model:
            data["model"] = model
        if self.format == "v2":
            data["text"] = "synthetic prompt"
        else:
            data["role"] = "user"
        self.message(id, data, session, "user", seq)
        if self.format != "v2":
            self.part(id + "-text", id, {"type": "text", "text": "synthetic prompt"}, session)

    def assistant(self, id="a", session="s", created=T + 2_000,
                  completed=T + 6_000, output=120, seq=2, parent="u",
                  metadata=None, error=False, model="test-model", provider="test-provider",
                  variant="high", content=None):
        data = {"time": {"created": created},
                "tokens": {"input": 1000, "output": output, "reasoning": 80,
                           "total": 9999, "cache": {"read": 2000, "write": 3000}},
                "finish": "stop"}
        if completed is not None:
            data["time"]["completed"] = completed
        if metadata:
            data["metadata"] = metadata
        if self.format == "v2":
            data["model"] = {"id": model, "providerID": provider, "variant": variant}
            data["time"]["streamed"] = T + 5_000
            data["content"] = content if content is not None else [
                {"type": "text", "text": "synthetic answer"},
                {"type": "reasoning", "text": "synthetic reasoning",
                 "time": {"created": T + 2_500, "completed": T + 4_000}},
            ]
            if error:
                data["error"] = {"name": "synthetic-error"}
        else:
            data.update(role="assistant", parentID=parent,
                        providerID=provider, modelID=model, variant=variant)
            if error:
                data["error"] = {"name": "synthetic-error"}
        self.message(id, data, session, "assistant", seq)
        if self.format != "v2":
            self.part(id + "-text", id, {"type": "text", "text": "synthetic answer",
                                       "time": {"start": T + 6_000, "end": T + 6_000}}, session)
            self.part(id + "-reason", id, {"type": "reasoning", "text": "synthetic reasoning",
                                         "time": {"start": T + 2_500, "end": T + 4_000}}, session)
        return data


class TestOpenCodeAdapter(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def store(self, format):
        return NativeStore(self.root / format, format)

    def basic(self, format):
        store = self.store(format)
        store.session()
        store.user()
        store.assistant()
        return store, OpenCodeAdapter(store.root)

    def assert_unmeasured(self, spans):
        for span in spans:
            self.assertEqual(span.agent, "opencode")
            self.assertFalse(span.is_valid)
            self.assertIsNone(span.tps)
            self.assertEqual(span.timing_source, "opencode-unconfirmed")
            self.assertEqual(span.note, "unconfirmed_generation_timing")

    def test_native_generations_count_output_only_and_never_measure_decoding(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                self.assertTrue(adapter.detect())
                spans = adapter.collect()
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].tokens, 120)
                self.assertEqual(spans[0].model, "test-provider/test-model")
                self.assert_unmeasured(spans)
                self.assertIsNone(spans[0].reasoning_effort)
                self.assertIsNone(spans[0].service_tier)
                self.assertIsNone(spans[0].speed)
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.total_tokens, 120)
                self.assertEqual(timeline.assistant_messages, 1)
                self.assertEqual(timeline.user_messages, 1)
                self.assertEqual(timeline.cwd, "/recorded/project")
                self.assertEqual(sum(e.kind == "reasoning" for e in timeline.events), 1)
                self.assertTrue(all(e.tokens is None for e in timeline.events
                                    if e.kind != "assistant_message"))

    def test_v1_mirrored_step_usage_and_distinct_steps_replace_snapshot(self):
        for format in ("json", "v1"):
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                store.part("step-one", "a", {"type": "step-finish", "tokens": {
                    "output": 120, "reasoning": 80, "input": 1000}})
                self.assertEqual(sum(s.tokens for s in adapter.collect()), 120)
                self.assertEqual(adapter.collect_sessions()[0].total_tokens, 120)
                store.part("step-two", "a", {"type": "step-finish", "tokens": {
                    "output": 30, "reasoning": 999}})
                spans = adapter.collect()
                self.assertEqual(sorted(s.tokens for s in spans), [30, 120])
                self.assertEqual(len({s.turn_id for s in spans}), 2)
                self.assert_unmeasured(spans)
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.total_tokens, 150)
                self.assertEqual(timeline.assistant_messages, 1)
                self.assertEqual([e.tokens for e in timeline.events if e.kind == "assistant_message"], [150])
                store.part("step-two", "a", {"type": "step-finish", "tokens": {"output": 40}})
                refreshed = adapter.read_session("s")
                self.assertEqual(refreshed.total_tokens, 160)
                self.assertEqual({s.turn_id for s in spans}, {s.turn_id for s in adapter.collect()})

    def test_v1_bad_step_usage_falls_back_to_snapshot(self):
        for format in ("json", "v1"):
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                for n, output in enumerate((True, -1, 2.5, "90", None)):
                    store.part("bad-step-" + str(n), "a", {
                        "type": "step-finish", "tokens": {"output": output, "reasoning": 999}})
                self.assertEqual(sum(s.tokens for s in adapter.collect()), 120)
                self.assertEqual(adapter.collect_sessions()[0].total_tokens, 120)

    def test_explicit_metadata_not_native_variant_and_no_duplicate_provider(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                store.assistant(metadata={"reasoning_effort": "low", "service_tier": "priority", "speed": "fast"},
                                model="test-provider/test-model")
                span = adapter.collect()[0]
                self.assertEqual(span.model, "test-provider/test-model")
                self.assertEqual((span.reasoning_effort, span.service_tier, span.speed),
                                 ("low", "priority", "fast"))
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.reasoning_effort, "low")
                self.assertEqual(timeline.service_tier, "priority")

    def test_semantic_cutoffs_ignore_migration_rows_and_keep_context(self):
        cutoff = (T + 6_000) / 1000
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                if format == "json":
                    for path in store.root.rglob("*.json"):
                        os.utime(path, (cutoff + 100_000, cutoff + 100_000))
                spans = adapter.collect(min_timestamp=cutoff)
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].ended_at, cutoff)
                self.assertEqual(adapter.collect(min_timestamp=cutoff + .001), [])
                timelines = adapter.collect_sessions(min_timestamp=cutoff)
                self.assertEqual(len(timelines), 1)
                self.assertEqual(timelines[0].user_messages, 1)
                self.assertEqual(timelines[0].updated_at, cutoff)
                self.assertEqual(adapter.collect_sessions(min_timestamp=cutoff + .001), [])

    def test_mutable_usage_and_tools_keep_identity_at_equal_timestamps(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                tool = {"type": "tool", "id": "call-one", "callID": "call-one",
                        "name": "synthetic-tool", "tool": "synthetic-tool",
                        "time": {"created": T + 6_000, "ran": T + 6_000, "completed": T + 6_000},
                        "state": {"status": "completed", "input": {}, "output": "synthetic",
                                  "time": {"start": T + 6_000, "end": T + 6_000}}}
                if format == "v2":
                    store.assistant(output=10, content=[tool])
                else:
                    store.part("tool-one", "a", tool)
                before = adapter.collect_sessions()[0]
                if format == "v2":
                    store.assistant(output=140, content=[tool])
                else:
                    store.assistant(output=140)
                after = adapter.read_session("s")
                self.assertEqual(after.total_tokens, 140)
                self.assertEqual(after.assistant_messages, 1)
                self.assertEqual(after.tool_calls, 1)
                self.assertEqual(sum(e.kind == "tool_output" for e in after.events), 1)
                self.assertEqual({e.event_id for e in before.events}, {e.event_id for e in after.events})
                self.assertEqual(len({e.event_id for e in after.events}), len(after.events))
                store.assistant(id="a-two", seq=3, output=50)
                final = adapter.read_session("s")
                self.assertEqual(final.total_tokens, 190)
                self.assertEqual(final.assistant_messages, 2)

    def test_error_and_missing_completion_are_incomplete_not_measured(self):
        for format in FORMATS:
            for error, completed in ((True, T + 6_000), (False, None)):
                with self.subTest(format=format, error=error):
                    store, adapter = self.basic(format)
                    store.assistant(error=error, completed=completed)
                    spans = adapter.collect()
                    self.assertEqual(len(spans), 1)
                    self.assertEqual(spans[0].tokens, 120)
                    self.assertFalse(spans[0].is_valid)
                    self.assertEqual(spans[0].note, "incomplete_response")
                    self.assertEqual(spans[0].timing_source, "opencode-unconfirmed")
                    self.assertIsNone(spans[0].tps)

    def test_header_and_user_only_sessions_remain_visible(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store = self.store(format)
                store.session("header", directory=None)
                store.session("prompt", directory=None)
                store.user(session="prompt")
                adapter = OpenCodeAdapter(store.root)
                self.assertEqual(adapter.collect(), [])
                timelines = {s.session_id: s for s in adapter.collect_sessions()}
                self.assertEqual(set(timelines), {"header", "prompt"})
                self.assertEqual(timelines["header"].events, [])
                self.assertEqual(timelines["prompt"].user_messages, 1)
                self.assertTrue(all(s.cwd is None for s in timelines.values()))

    def test_v1_cloned_history_is_not_owned_without_parent_marker(self):
        for format in ("json", "v1"):
            with self.subTest(format=format):
                store = self.store(format)
                store.session("child", created=T + 10_000, updated=T + 20_000)
                store.user("copied-u", "child", created=T + 1_000)
                store.assistant("copied-a", "child", created=T + 2_000,
                                completed=T + 11_000, parent="copied-u")
                store.user("own-u", "child", created=T + 12_000)
                store.assistant("own-a", "child", created=T + 13_000,
                                completed=T + 16_000, output=40, parent="own-u")
                adapter = OpenCodeAdapter(store.root)
                self.assertEqual([s.tokens for s in adapter.collect()], [40])
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.total_tokens, 40)
                self.assertEqual(timeline.assistant_messages, 1)
                self.assertEqual(timeline.user_messages, 1)

    def test_v2_fork_boundaries_use_parent_sequence_gaps(self):
        for boundary_type, copy_limit in (("before", 3), ("through", 10)):
            with self.subTest(boundary=boundary_type):
                root = self.root / boundary_type
                store = NativeStore(root, "v2")
                store.session("parent")
                store.user("parent-u", "parent", seq=1)
                store.assistant("parent-a", "parent", seq=3, output=120)
                store.assistant("boundary-a", "parent", seq=10, output=50)
                store.session("child", created=T + 10_000, fork="parent",
                              boundary={"type": boundary_type, "messageID": "boundary-a"})
                store.user("msg_fork_1", "child", seq=1)
                store.assistant("msg_fork_3", "child", seq=3, output=120)
                if copy_limit == 10:
                    store.assistant("msg_fork_10", "child", seq=10, output=50)
                # For 'before', seq 5 is owned even though 5 < boundary seq - 1.
                store.user("child-u", "child", seq=copy_limit + 1, created=T + 12_000)
                store.assistant("child-a", "child", seq=copy_limit + 2,
                                created=T + 13_000, completed=T + 16_000, output=40)
                adapter = OpenCodeAdapter(root)
                child_spans = [s for s in adapter.collect() if s.session_id == "child"]
                self.assertEqual([s.tokens for s in child_spans], [40])
                child = next(s for s in adapter.collect_sessions() if s.session_id == "child")
                self.assertEqual(child.total_tokens, 40)
                self.assertEqual(child.assistant_messages, 1)
                self.assertEqual(child.user_messages, 1)

    def test_sqlite_preferred_over_leftover_json_for_same_id(self):
        for format in ("v1", "v2"):
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                leftover = NativeStore(store.root, "json")
                leftover.session()
                leftover.user()
                leftover.assistant(output=999)
                self.assertEqual(OpenCodeAdapter(store.root).read_session("s").total_tokens, 120)
                self.assertEqual([s.tokens for s in adapter.collect()], [120])
                self.assertEqual([s.total_tokens for s in adapter.collect_sessions()], [120])

    def test_limits_newest_semantic_sessions_and_exact_pinned_refresh(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                self.assertEqual(adapter.collect(max_sessions=0), [])
                self.assertEqual(adapter.collect_sessions(max_sessions=-1), [])
                self.assertEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "s")
                store.session("new", created=T + 20_000, updated=T + 30_000)
                store.user("new-u", "new", created=T + 21_000)
                store.assistant("new-a", "new", created=T + 22_000,
                                completed=T + 26_000, output=60)
                self.assertEqual(adapter.collect_sessions(max_sessions=1)[0].session_id, "new")
                store.assistant(output=130)
                self.assertEqual(adapter.read_session("s").total_tokens, 130)
                source = store.db if format != "json" else store.root / "storage/session/project/s.json"
                hidden = source.with_suffix(source.suffix + ".hidden")
                source.rename(hidden)
                self.assertIsNone(adapter.read_session("s"))
                hidden.rename(source)
                self.assertEqual(adapter.read_session("s").total_tokens, 130)

    def test_uncached_exact_oldest_session_beyond_recent_limit_stays_pinned(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                source = store.db if format != "json" else store.root / "storage/session/project/s.json"
                if format == "json":
                    os.utime(source, (T / 1000, T / 1000))
                for index in range(64):
                    timestamp = T + (index + 1) * 20_000
                    path = store.session("new-" + str(index), created=timestamp, updated=timestamp + 10_000)
                    if format == "json":
                        os.utime(path, (timestamp / 1000, timestamp / 1000))
                timeline = adapter.read_session("s")
                self.assertEqual(timeline.session_id, "s")
                self.assertEqual(timeline.total_tokens, 120)
                store.assistant(output=130)
                self.assertEqual(adapter.read_session("s").total_tokens, 130)
                hidden = source.with_suffix(source.suffix + ".hidden")
                source.rename(hidden)
                self.assertIsNone(adapter.read_session("s"))
                hidden.rename(source)
                self.assertEqual(adapter.read_session("s").total_tokens, 130)

    def test_uncached_sqlite_exact_id_is_parameterized_and_isolated(self):
        for format in ("v1", "v2"):
            with self.subTest(format=format):
                store, _ = self.basic(format)
                session_id = "quoted' OR 1=1 --"
                store.session(session_id)
                store.assistant("quoted-a", session=session_id, output=45)
                (store.root / "opencode-broken.db").write_bytes(b"not a sqlite database")
                adapter = OpenCodeAdapter(store.root)
                self.assertIsNone(adapter.read_session("absent' OR 1=1 --"))
                timeline = adapter.read_session(session_id)
                self.assertEqual(timeline.session_id, session_id)
                self.assertEqual(timeline.total_tokens, 45)

    def test_uncached_json_header_lookup_is_exact_and_rejects_substitution(self):
        store, adapter = self.basic("json")
        source = store.root / "storage/session/project/s.json"
        renamed = source.with_name("different-filename.json")
        source.rename(renamed)
        self.assertEqual(adapter.read_session("s").total_tokens, 120)
        original = renamed.read_text(encoding="utf-8")
        store._json(renamed, {"id": "replacement", "time": {"created": T, "updated": T + 10_000}})
        store.session()
        self.assertIsNone(adapter.read_session("s"))
        renamed.write_text(original, encoding="utf-8")
        self.assertEqual(adapter.read_session("s").total_tokens, 120)

    def test_broken_database_does_not_hide_valid_separate_database(self):
        root = self.root / "databases"
        root.mkdir()
        (root / "opencode.db").write_bytes(b"not a sqlite database")
        store = NativeStore(root, "v2", "opencode-good.db")
        store.session()
        store.user()
        store.assistant()
        adapter = OpenCodeAdapter(root)
        self.assertEqual([s.tokens for s in adapter.collect()], [120])
        self.assertEqual(adapter.collect_sessions()[0].session_id, "s")

    def test_corrupt_json_and_nonobject_records_do_not_discard_good_session(self):
        store, adapter = self.basic("json")
        bad = store.root / "storage/session/project/bad.json"
        for payload in ("{", "null", "[]", '"text"', '{"id":"bad","time":[]}'):
            with self.subTest(payload=payload):
                bad.write_text(payload, encoding="utf-8")
                self.assertEqual([s.tokens for s in adapter.collect()], [120])
                self.assertIn("s", {s.session_id for s in adapter.collect_sessions()})
        (store.root / "storage/part/a/broken.json").write_text("{", encoding="utf-8")
        self.assertEqual(adapter.collect_sessions()[0].total_tokens, 120)

    def test_bad_nested_payloads_and_numeric_counts_are_isolated(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                for index, data in enumerate((None, [], "text", {"time": []},
                                             {"role": "assistant", "time": {"created": True}, "tokens": []},
                                             {"role": "assistant", "time": {"created": float("nan")}, "tokens": None})):
                    store.message("broken-" + str(index), data, kind="assistant", seq=10 + index)
                spans = adapter.collect()
                self.assertEqual(sum(s.tokens for s in spans), 120)
                self.assertEqual(adapter.read_session("s").total_tokens, 120)
                for index, output in enumerate((True, -3, 2.5, "50", None)):
                    store.assistant("invalid-output-" + str(index), output=output, seq=30 + index)
                self.assertEqual(sum(s.tokens for s in adapter.collect()), 120)
                self.assertEqual(adapter.read_session("s").total_tokens, 120)

    def test_sqlite_reading_does_not_change_database_or_create_sidecars(self):
        for format in ("v1", "v2"):
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                before = store.db.read_bytes()
                files = set(store.root.iterdir())
                store.db.chmod(0o444)
                try:
                    self.assertEqual(adapter.collect()[0].tokens, 120)
                    self.assertEqual(adapter.collect_sessions()[0].total_tokens, 120)
                    self.assertEqual(adapter.read_session("s").total_tokens, 120)
                    self.assertEqual(store.db.read_bytes(), before)
                    self.assertEqual(set(store.root.iterdir()), files)
                finally:
                    store.db.chmod(0o644)

    def test_root_xdg_and_database_override_precedence(self):
        fake_home = self.root / "home"
        xdg = self.root / "xdg"
        explicit = self.root / "explicit"
        with patch.dict(os.environ, {"XDG_DATA_HOME": str(xdg),
                                     "OPENCODE_HOME": str(self.root / "wrong"),
                                     "OPENCODE_TEST_HOME": str(self.root / "also-wrong")}, clear=True), \
                patch.object(Path, "home", return_value=fake_home):
            self.assertEqual(OpenCodeAdapter().root, xdg / "opencode")
            self.assertEqual(OpenCodeAdapter(explicit).root, explicit)
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "home", return_value=fake_home):
            self.assertEqual(OpenCodeAdapter().root, fake_home / ".local/share/opencode")
        data = xdg / "opencode"
        store = NativeStore(data, "v2", "selected.db")
        store.session()
        store.user()
        store.assistant()
        for override in ("selected.db", str(store.db)):
            with self.subTest(override=override), patch.dict(os.environ, {
                "XDG_DATA_HOME": str(xdg), "OPENCODE_DB": override}, clear=True):
                self.assertEqual(OpenCodeAdapter().collect()[0].tokens, 120)
                # Explicit data roots must not pick up a database from another root.
                self.assertEqual(OpenCodeAdapter(explicit).collect(), [])

    def test_historical_models_turns_and_parent_lineage_are_recorded_not_projected(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                store.session(parent="lineage-only",
                              model={"id": "current-selector", "providerID": "current-provider"})
                store.user("u-two", created=T + 7_000, seq=3,
                           model={"providerID": "later-provider", "modelID": "later-model",
                                  "id": "later-model"})
                store.assistant("a-two", created=T + 8_000, completed=T + 12_000,
                                seq=4, parent="u-two", model="later-model", provider="later-provider",
                                output=40)
                spans = adapter.collect()
                self.assertEqual({s.model for s in spans},
                                 {"test-provider/test-model", "later-provider/later-model"})
                timeline = adapter.collect_sessions()[0]
                self.assertEqual(timeline.model, "later-provider/later-model")
                self.assertEqual(timeline.total_tokens, 160)
                users = [e for e in timeline.events if e.kind == "user_message"]
                assistants = [e for e in timeline.events if e.kind == "assistant_message"]
                self.assertEqual(len({e.turn_id for e in assistants}), 2)
                self.assertEqual({e.turn_id for e in users}, {e.turn_id for e in assistants})
                self.assertIsNone(timeline.reasoning_effort)

    def test_v2_latest_native_model_selection_does_not_rewrite_history(self):
        store, adapter = self.basic("v2")
        store.message("switch", {"model": {"providerID": "new-provider", "id": "new-model",
                                          "variant": "high"},
                                 "time": {"created": T + 9_000}},
                      kind="model-switched", seq=3)
        self.assertEqual(adapter.collect()[0].model, "test-provider/test-model")
        timeline = adapter.collect_sessions()[0]
        self.assertEqual(timeline.model, "new-provider/new-model")
        self.assertIsNone(timeline.reasoning_effort)
        self.assertEqual(timeline.total_tokens, 120)

    def test_v2_deleted_parent_fallback_recognizes_only_native_copy_ids(self):
        store = self.store("v2")
        store.session("child", created=T + 10_000, fork="deleted-parent",
                      boundary={"type": "through", "messageID": "missing-boundary"})
        stem = "012345abcdefABCDEFGHIJKLMN"
        store.assistant("msg_" + stem + "_3", "child", seq=3,
                        created=T + 12_000, completed=T + 16_000, output=900)
        store.user("own-u", "child", seq=10, created=T + 12_000)
        store.assistant("msg_custom_11", "child", seq=11,
                        created=T + 13_000, completed=T + 16_000, output=40)
        # A native-looking ID with a suffix not matching seq is not a copied row.
        store.assistant("msg_" + stem + "_12", "child", seq=13,
                        created=T + 13_000, completed=T + 16_000, output=20)
        adapter = OpenCodeAdapter(store.root)
        self.assertEqual(sum(s.tokens for s in adapter.collect()), 60)
        child = adapter.collect_sessions()[0]
        self.assertEqual(child.total_tokens, 60)
        self.assertEqual(child.assistant_messages, 2)

    def test_invalid_native_times_do_not_make_migration_time_recent(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                for index, value in enumerate((True, "1800000000000", float("inf"), float("nan"), [])):
                    data = {"role": "assistant", "parentID": "u",
                            "providerID": "test-provider", "modelID": "test-model",
                            "model": {"id": "test-model", "providerID": "test-provider"},
                            "time": {"created": value, "streamed": value, "completed": value},
                            "tokens": {"output": 100}, "content": [], "finish": "stop"}
                    store.message("bad-time-" + str(index), data, kind="assistant",
                                  seq=20 + index, row_time=T + 1_000_000)
                self.assertEqual(adapter.collect(min_timestamp=(T + 100_000) / 1000), [])
                self.assertEqual(adapter.collect_sessions(min_timestamp=(T + 100_000) / 1000), [])

    def test_sqlite_connections_reject_writes_not_just_leave_fixture_unchanged(self):
        for format in ("v1", "v2"):
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                original_connect = sqlite3.connect
                connections = []

                def connect(*args, **kwargs):
                    connection = original_connect(*args, **kwargs)
                    connections.append(connection)
                    with self.assertRaises(sqlite3.OperationalError):
                        connection.execute("CREATE TABLE forbidden_adapter_write (id INTEGER)")
                    return connection

                with patch("tokenmon.adapters.opencode.sqlite3.connect", side_effect=connect):
                    self.assertEqual(adapter.collect()[0].tokens, 120)
                self.assertTrue(connections)

    def test_metadata_spellings_in_native_message_and_tokens_are_explicit(self):
        for format in FORMATS:
            with self.subTest(format=format):
                store, adapter = self.basic(format)
                data = store.assistant()
                data["reasoning_effort"] = "medium"
                data["tokens"].update(service_tier="standard", speed="fast")
                store.message("a", data, kind="assistant", seq=2)
                span = adapter.collect()[0]
                self.assertEqual((span.reasoning_effort, span.service_tier, span.speed),
                                 ("medium", "standard", "fast"))
                self.assertEqual(span.tokens, 120)

    def test_json_malformed_folder_id_is_not_followed_outside_storage(self):
        store, adapter = self.basic("json")
        store._json(store.root / "storage/session/project/traversal.json",
                    {"id": "../outside", "directory": "/not-owned",
                     "time": {"created": T, "updated": T + 100_000}})
        self.assertEqual([s.session_id for s in adapter.collect_sessions()], ["s"])
        self.assertIsNone(adapter.read_session("../outside"))

    def test_exact_database_source_remains_pinned_when_discovery_changes(self):
        for format in ("v1", "v2"):
            with self.subTest(format=format):
                root = self.root / ("pinned-databases-" + format)
                original = NativeStore(root, format, "opencode-original.db")
                original.session()
                original.user()
                original.assistant(output=120)
                adapter = OpenCodeAdapter(root)
                self.assertEqual(adapter.read_session("s").total_tokens, 120)
                newer = NativeStore(root, format, "opencode-new.db")
                newer.session(created=T + 20_000, updated=T + 40_000)
                newer.user(created=T + 21_000)
                newer.assistant(created=T + 22_000, completed=T + 26_000, output=900)
                original.assistant(output=130)
                self.assertEqual(adapter.read_session("s").total_tokens, 130)
                hidden = original.db.with_suffix(".hidden")
                original.db.rename(hidden)
                self.assertIsNone(adapter.read_session("s"))
                hidden.rename(original.db)
                self.assertEqual(adapter.read_session("s").total_tokens, 130)


if __name__ == "__main__":
    unittest.main()
