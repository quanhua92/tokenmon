# Agent guide

## Purpose and constraints

`tokenmon` is a local, read-only CLI for coding-agent generation throughput,
session activity, and timelines. Keep runtime dependencies at zero: Python 3.10+
standard library only. `pyproject.toml` uses Hatchling for packaging; `.python-version`
selects 3.13 for local development. No formatter or Python linter is configured.
`CLAUDE.md` includes this file; keep shared guidance here.
The PyPI package, Python module, and executable are all `tokenmon`.

Never modify agent telemetry or execute commands found in logs. Open JSONL for
reading; open SQLite through `open_ro_db()` (`mode=ro`, `query_only`, short timeout)
and close connections with `contextlib.closing`. Use parameterized SQL values.
Do not add network calls, model requests, telemetry uploads, or real transcripts
to normal operation or tests.

## Code map

| Location | Responsibility |
| --- | --- |
| `src/tokenmon/models.py` | Span validation, summary/event dataclasses, session counts and idle status. |
| `src/tokenmon/analyzer.py` | Time windows, grouping by model, weighted throughput and distribution. |
| `src/tokenmon/adapters/base.py` | `BaseAdapter` contract: `name`, `detect`, `collect`, `collect_sessions`, activity probe. |
| `src/tokenmon/adapters/__init__.py` | Registry, aliases, activity-based auto-detection. |
| `src/tokenmon/adapters/{codex,claude,antigravity}.py` | Source discovery, generation spans, and session timelines. |
| `src/tokenmon/cli.py` | Argument normalization, collection, human formatting, JSON serialization. |
| `src/tokenmon/terminal.py` | `cmd.Cmd` shell, watch loop, completion, three-second timeline cache. |
| `src/tokenmon/live.py` | Shared stats watch loop and append-only following of a pinned session; no `readline` dependency. |
| `src/tokenmon/{__init__,__main__}.py` | Version and module entry point; console script calls `cli.main`. |
| `tests/test_*.py` | Standard-library `unittest`, temporary JSONL/SQLite fixtures, CLI subprocess checks. Model/analyzer tests live in `test_codex_adapter.py`. |
| `.github/workflows/{ci,release}.yml` | PR/main tests and package checks; manual main-only PyPI trusted publishing. Build tools are pinned in `.github/requirements-build.txt`. |
| `.github/RELEASING.md` | Maintainer instructions for publisher setup, releases, and PyPI descriptions. |

## Measurement rules

- Build adapter spans with `create_span()`. It rejects missing/nonfinite timestamps,
  nonpositive duration/tokens, duration below 1s, and rates above 400 TPS. An explicit
  `note` invalidates a span. Preserve `timing_source` and exclusion reasons.
- Measure generation using evidenced boundaries; request/turn elapsed time can
  include latency, tool execution, and waiting. Label fallbacks accurately and
  exclude unconfirmed timing instead of inventing a start or inflating TPS.
- Invalid spans remain available for total/excluded counts; valid spans alone
  contribute tokens, duration, and rates. Weighted TPS is total valid tokens divided
  by total valid duration. Median/min/max use individual valid rates.
- Finite windows use completion time: `0 <= now - span.ended_at <= window_seconds`.
  `all` bypasses time filtering. Supported windows: `30m`, `1d`, `7d`, `30d`, `all`.
- Times are epoch seconds internally; convert source milliseconds/nanoseconds
  explicitly. JSON uses numeric timestamps; human views use local time. ISO parsers
  reject naive timestamps. Keep events chronological, spans ascending by end, and
  session lists descending by update time, including after merging adapters.
- Session duration/activity is separate from generation duration. Status uses
  time since the last event: Active below 120s, Idle below 600s, then Inactive.
  `cwd` is optional recorded metadata; unknown is `None`, never the monitor's cwd.
- Timeline events carry an optional internal `event_id` for live deduplication;
  keep it stable across usage/timestamp updates and out of the public JSON shape.
- Generation spans, sessions, and timeline events expose optional `reasoning_effort`, `service_tier`, and `speed`
  from recorded metadata. Missing values stay `None`/JSON `null`; never infer them
  from model names, TPS, or the monitor's current configuration. Recent human rows
  show effort and speed, mapping recorded priority/fast tiers to `fast` and
  default/standard tiers to `standard`. Settings describe recorded request mode.
  All JSON stream/session/event objects include all three raw fields and normalized
  `speed_mode`, with missing values as `null`. Session fields describe latest
  recorded settings; event fields preserve historical settings. Stats summary/model
  windows include distinct valid-stream `configurations`; singular fields are set
  only when every configuration agrees. Human session/timeline views also show
  available effort and speed.
- Human stats total time uses the shared hours/minutes/seconds formatter; JSON
  durations remain numeric seconds. Explain that overlapping spans sum separately
  and Claude single-record `turn-span` is a latency-inclusive estimate.

## Adapter specifics

Explicit `root`/CLI `--home` overrides the agent environment variable, then its
default home path. Collection defaults to 64 sessions for spans and 32 for timelines;
accept and propagate `max_sessions` and `min_timestamp`. Bound discovery before
expensive parsing, but preserve earlier context needed by records within a window.
`read_session(session_id)` refreshes an exact session for live follow. Built-in
adapters retain discovered source paths or use an exact database ID so the pinned
session continues to be readable outside the recent-session discovery limit.

- **Codex:** `CODEX_HOME` or `~/.codex`. Choose the highest numeric
  `state_*.sqlite`/`logs_*.sqlite` version. Discover unarchived `threads` with optional
  `model`; fall back to recursive `sessions/**/*.jsonl`. Span starts prefer diagnostic
  stream logs, then `item_completed` timing. Correlate item IDs, turn context and
  usage; deduplicate response IDs. Tool-first/placeholder starts are excluded.
  Support diagnostic `timestamp`/`message` and native `ts` + `ts_nanos` /
  `feedback_log_body` schemas. Track cumulative `total_token_usage` snapshots even
  before cutoffs or without pending output, and ignore repeats without clearing
  newer pending items or assigning old tokens to newer timeline events. Effort
  comes from turn context/settings; tier comes from settings or response usage.
  `cwd` comes from the first usable `session_meta`/`turn_context` payload.
- **Claude:** `CLAUDE_HOME` or `~/.claude`; parse `projects/*/*.jsonl`. Group assistant
  chunks by message ID; use final output usage. Multiple chunks use first/last chunk
  timestamps; single records fall back to the preceding user/prompt/tool result.
  Deduplicate assistant messages without losing final usage or repeating tools.
  Preserve earlier prompts and chunks before filtering by completion time. Read
  explicit effort/configuration and final usage tier/speed metadata when present.
  The first nonempty record `cwd` wins, even without a timestamp.
- **Antigravity / agy:** `ANTIGRAVITY_HOME` or `~/.gemini/antigravity-cli`.
  Discover `conversations/*.db`, preferring `conversation_summaries.db` ordering.
  Minimal protobuf-wire decoding stays dependency-free. Steps 14/15/132 are
  user/assistant/tool; metadata fields 1 and 7 (fallback 8) provide timing, field 9
  contains usage (field 3 output tokens). `gen_metadata` supplies model mappings.
  Carry model and explicit effort/tier/speed mappings from their boundary through
  subsequent steps until the next boundary. Preserve decoded invalid spans for
  exclusion counts; skip malformed wire records without losing later valid steps.
  `workspace_uris` supplies the first usable URL-decoded `file://` workspace path.

Treat logs as evolving input: skip bad/unknown records and tolerate missing optional
files, tables, columns, and truncated data. Validate decoded object shapes before
`.get()`. One broken record/session/adapter should not discard later valid data or
other adapters. Diagnostics go to logging/stderr; never contaminate JSON stdout.
Add adapters by implementing `BaseAdapter`, registering names/aliases, and adding
synthetic fixtures for spans and timelines. Deduplicate registry classes, not names.
Auto-detection favors adapters active within 30 days, then the most recent source;
when activity timestamps are unavailable it returns detected sources.

## CLI compatibility

Preserve `stats`/`top` (default), `sessions`/`ps`/`ls`, `timeline`/`log`/`logs`, and
`interactive`/`repl`/`shell` plus `-i`/`--interactive`. Bare agent/option invocations
default to stats. Timeline accepts an ID/prefix or `latest`, with `--agent` selection;
an adapter name in its positional slot selects that adapter's latest session.

Time windows use `--window` explicitly; `-w` now means `--watch` on stats and
does not take a window value. `stats`/`top` and bare stats invocations accept
`-w`/`--watch` and `--interval` (default
2 seconds). CLI and shell watch share stats collection/rendering, including recent
streams and session cards. Recalculate cutoffs each refresh; clear the screen only
when stdout is a terminal. Show recent sessions even without generation spans.

`timeline`/`log`/`logs` accept `-f`/`--follow` and `--interval`. Resolve and pin the
initial session, baseline its existing events without printing them, and append
only new source identities. Do not replay history, redraw the screen, switch to a
newer session, or repeat an event because its metadata changed. Preserve the seen
baseline through temporary read failures. Live modes reject `--json`; follow also
rejects batch `--window` exports. Ctrl+C stops CLI live modes or returns shell watch
to the prompt. Keep live CLI imports independent of the shell and `readline`.

The default collection cutoff is 30 days. `--window` selects a window; `--all` or
`--window all` removes the cutoff, but session limits still apply. Positional agent
`all` selects detected adapter classes; it is separate from history flag `--all`.
Stats recent-session previews currently collect independently of the span cutoff.

Keep `--json` stdout clean and preserve its shapes: stats is an object with `meta`,
`summary`, `models`, `recent_streams`, `recent_sessions`; sessions is an array;
single timeline is an object; window timeline export is an array. Keep `cwd` in
session JSON, missing values as `null`, and rounding at serialization. Human output
supports automatic narrow layouts and `--compact`/`--wide`. Import the shell lazily
from the CLI; it imports CLI formatters and currently requires `readline`.

## Validation and change discipline

Run from the repository root, without requiring installation or network access:

```sh
PYTHONPATH=src python3 -m tokenmon --help
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_claude_adapter.py'
git diff --check
```

`uv run tokenmon` and `uv run python -m unittest discover -s tests` are the
documented installed alternatives. Use focused regression fixtures for changed
behavior, then the full suite. Test boundary timing, duplicate usage, malformed
records, schema fallbacks, cutoffs, root overrides, aliases, and JSON as relevant.
Freeze/inject reference time or use `--all` for unrelated assertions: existing
fixtures contain fixed dates and some Codex item milliseconds are from 2025 while
record timestamps are from 2026. Do not assume passing tests cover these edges.
Update README examples when public behavior changes; synchronize versions in
`pyproject.toml` and `src/tokenmon/__init__.py` when changing the release version.

CI tests Python 3.10–3.14 on Linux and 3.13 on macOS. CLI subprocess tests must use
`sys.executable` so they actually exercise the selected interpreter. Validate workflow
edits with `actionlint`. Pin action references to verified full commit SHAs, keep
checkout credentials disabled, and use hosted runners with read-only PR permissions.
Release input is a version check, not a version bump. Build/test jobs must never have
`id-token: write`; the separate `pypi` environment job only downloads and publishes
the current run's validated distributions. Preserve its main-only/manual gate.

## Regression traps

Keep coverage for these previously reproduced defects:

- The shell's `agents` command must import and enumerate the adapter registry.
- Codex must attach `token_count` usage even when the generic `event_msg` branch runs;
  paired usage formats must not double-count or overwrite another turn's output.
- Claude must update a streamed message with final usage/time while counting its
  assistant response, reasoning blocks, and tool IDs once.
- Session cutoffs must check the final event time after discovery; touching an old
  log must not put it inside a window. Events exactly at the cutoff are included.
- Sort sessions globally after merging adapters so `latest` selects the newest.
- Skip non-object JSON and malformed nested shapes without losing later valid
  records in either span or timeline parsing.
- Watch includes recent streams and session cards, including sessions with no
  generation spans. Follow emits nothing historical, handles equal timestamps and
  streamed metadata updates, and remains pinned when recent-session ordering changes.
- Codex repeated usage snapshots must not create false high-TPS reasoning spans;
  fresh cumulative counts with identical per-response token counts still count.
  Codex/Claude window cutoffs preserve preceding metadata and generation starts.
  Antigravity settings carry across step ranges, bad/zero-duration spans remain
  excluded, and malformed records do not discard later valid steps.
