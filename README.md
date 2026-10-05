# TokenMon

A fast, zero-dependency command-line monitor for local AI coding agents.

It measures how fast models actually generate tokens (Tokens Per Second, TPS) by tracking pure generation time—separating thinking and output from tool runs, file edits, and idle waiting.

[![Recent generation speeds and session activity](https://raw.githubusercontent.com/quanhua92/tokenmon/main/demo-stats-preview.png)](https://raw.githubusercontent.com/quanhua92/tokenmon/main/demo-stats.jpg)

---

## Screenshots

Expand a screenshot below. Click the image to view it at full size.

<details>
<summary><strong>stats</strong> — generation speed and token throughput</summary>

[![Generation statistics](https://raw.githubusercontent.com/quanhua92/tokenmon/main/demo-stats.jpg)](https://raw.githubusercontent.com/quanhua92/tokenmon/main/demo-stats.jpg)

</details>

<details>
<summary><strong>ps</strong> — sessions, activity, and token usage</summary>

[![Session overview](https://raw.githubusercontent.com/quanhua92/tokenmon/main/demo-ps.jpg)](https://raw.githubusercontent.com/quanhua92/tokenmon/main/demo-ps.jpg)

</details>

<details>
<summary><strong>logs</strong> — chronological session timeline</summary>

[![Session timeline](https://raw.githubusercontent.com/quanhua92/tokenmon/main/demo-logs.jpg)](https://raw.githubusercontent.com/quanhua92/tokenmon/main/demo-logs.jpg)

</details>

---

## Features

- **Generation Speed**: Measures TPS from recorded output boundaries and labels timing fallbacks, including Claude prompt-to-response estimates.
- **Zero Extra Dependencies**: Runs on standard Python 3.10+ without installing third-party packages.
- **Strictly Read-Only**: Safely opens local files and SQLite databases in read-only mode (`?mode=ro`). Never locks or changes your logs.
- **Meaningful Averages**: Calculates true weighted speed ($\frac{\text{total tokens}}{\text{total time}}$), median speed, and min/max ranges over rolling time windows (`30m`, `1d`, `7d`, `30d`, `all`).
- **Supports Popular Agents**: Auto-detects **Codex** (`~/.codex`), **Claude Code** (`~/.claude/projects/`), **Antigravity** (`~/.gemini/antigravity-cli`), **Oh My Pi / OMP** (`~/.omp/agent/sessions/`), **Pi** (`~/.pi/agent/sessions/`), and **OpenCode** (`~/.local/share/opencode`; v1 JSON/SQLite and v2 SQLite).
- **Activity Timelines**: Compare prompts, reasoning, model output, tool execution, and waiting across sessions on one time axis.
- **Event Logs**: Inspect the step-by-step history of user prompts, thinking, assistant responses, and tool calls.
- **JSON Ready**: Add `--json` to pipe clean data into `jq` or external dashboards.

---

## Installation

Requires Python 3.10+.

Try it immediately with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uvx tokenmon
uvx tokenmon ps
uvx tokenmon logs
```

For regular use, install it as an isolated tool:

```bash
uv tool install tokenmon
tokenmon
```

If `tokenmon` is not on your PATH, run `uv tool update-shell` and restart your shell.
Upgrade with `uv tool upgrade tokenmon`.

Or install with pip in your Python environment:

```bash
pip install tokenmon
tokenmon
```

---

## Usage

TokenMon uses simple Docker-style subcommands: `stats` (default), `ps` (sessions), `timeline` (visual activity), `logs` (events), and `interactive` (shell).

Running `tokenmon` by itself defaults directly to `stats`.
The examples below use the installed command; with uvx, use `uvx tokenmon` in its place.

### Output Formats

Stats, sessions, visual timelines, and historical logs support terminal, JSON, and
standalone HTML output. `--json` remains a shortcut for `--output json`:

```bash
tokenmon stats --json | jq .
tokenmon ps --json | jq .
tokenmon logs 01a10275 --json | jq .
tokenmon timeline --window 1d --json | jq '.sessions'

tokenmon stats --output html > stats.html
tokenmon ps --output html > sessions.html
tokenmon timeline --output html > timeline.html
tokenmon logs 01a10275 --output html > logs.html
```

HTML reports contain their CSS, SVG charts, and small table-sorting script inline;
they load without network access or extra dependencies. Timeline HTML uses separate
model, reasoning, tool, user, and turn tracks so simultaneous events remain visible.
Live `--watch` and `--follow` modes require terminal output and reject JSON or HTML.

### Generation Speed & Metrics (`stats`, `top`, default)

```bash
tokenmon                                # auto-detect agents
tokenmon stats codex --window 1d         # select agent and window
tokenmon stats -w --interval 5           # live dashboard; Ctrl+C stops
tokenmon omp,pi -w                      # watch selected agents together
tokenmon omp,pi,codex -w                # combine three agents
tokenmon stats --all --wide              # full history and detailed layout
```

Windows: `30m`, `1d`, `7d`, `30d`, `all`; the default history cutoff is 30 days.
`-w` means watch, not a window. Use `--compact` for narrow panes. Watch refreshes
metrics, recent streams, and session cards; redirected output appends snapshots.
`--tasks` limits inspected sessions, including with `--all`.

- **Weighted TPS** = valid tokens ÷ valid generation seconds. Median and range use
  individual valid streams.
- Human tables separate each source agent and model pair; JSON keeps its stable
  model-keyed aggregation.
- Total generation time sums overlapping streams separately; human output uses
  hours/minutes/seconds, while JSON keeps numeric seconds.
- Streams under one second, above 400 TPS, or with unconfirmed timing are excluded.
  Claude's single-record `turn-span` includes latency; OMP's `omp-ttft` measures
  first-output-item to completion, not guaranteed first-text-token decoding.
- **Pi and OpenCode show sessions and token totals, but no confirmed TPS.**
  OpenCode v2's `time.streamed` is a stream **end**, not TTFT.
- Effort and speed come only from recorded metadata, never model names or TPS.
  Missing JSON fields are `null`; session settings are current and event settings
  historical. Stats `configurations` describe valid streams; singular fields are
  `null` when configurations disagree.

### Agent Data

| Agent | Default data root | Native override |
|---|---|---|
| `codex` | `~/.codex` | `CODEX_HOME` |
| `claude` | `~/.claude` | `CLAUDE_HOME` |
| `antigravity` / `agy` | `~/.gemini/antigravity-cli` | `ANTIGRAVITY_HOME` |
| `omp` | `~/.omp/agent` | `PI_CODING_AGENT_DIR`; native profiles/XDG |
| `pi` | `~/.pi/agent` | `PI_CODING_AGENT_DIR` |
| `opencode` | `~/.local/share/opencode` | `XDG_DATA_HOME/opencode`, `OPENCODE_DB` |

`--home` selects the data root and overrides native environment settings.
OMP/Pi roots contain `sessions`, including nested persisted workers. OMP also honors
`PI_CONFIG_DIR`, `OMP_PROFILE`/`PI_PROFILE`, and existing migrated XDG directories.
OpenCode's database override may be absolute or data-relative; in-memory stores
cannot be monitored externally. Existing/copied logs suffice without running an agent.

Select multiple agents with a comma-separated list wherever an agent selector is
accepted: `tokenmon omp,pi -w`, `tokenmon ps omp,pi`, or
`tokenmon logs latest --agent omp,pi,codex`. `tokenmon logs omp,pi` selects the
latest session across those agents; follow still pins just one session.
Names are case-insensitive, surrounding spaces are ignored, and repeated
names/aliases count only once. Unknown names and empty list entries are errors.
Use `all` alone to select detected agents, or omit selection for auto-detection.
Each selected adapter uses its own native root unless `--home` is supplied;
that override applies the same root to every selected adapter.

OpenCode formats were checked at **1.0.0/1.1.65** (split JSON), **1.2.0** (SQLite),
and **2.0.22** (SQLite projections). Detection is schema-based; SQLite takes
precedence over leftover migrated JSON. Assistant/step usage, inherited fork
history, and utility aggregates are not counted twice. Variant names are not
inferred effort; recorded output-token semantics may differ by provider/version.

All adapters include separately persisted worker sessions within their data roots.
Claude worker IDs are `agent-<id>@<project>/<parent>`; Codex uses native thread IDs
and recorded ownership ordinals to exclude copied worker history. Legacy Codex
worker forks without that boundary are excluded rather than double-counted.
Antigravity includes unindexed worker databases; opaque embedded workers and
copied-fork ownership cannot be decoded from the available metadata.

OMP uses recorded request starts to exclude inherited assistant output, recovering
the start from completion and duration when necessary. Parented output without
request-boundary evidence is excluded. Native OMP `session_init` workers remain
distinct from Pi sessions even when both share a data root.

### Active & Recent Sessions (`sessions`, `ps`, `ls`)

List sessions from the last **7 days** by default, with message counts, tool runs,
and idle status. Override with `--window`, or use `--all` for full history:

```bash
# List recent sessions
tokenmon ps

# Filter sessions within a time window
tokenmon ps --window 1d

# Filter to a specific agent
tokenmon ps codex
```

The Status column shows time since the last event using hours, minutes, and
seconds, for example `Idle (2m 05s)` or `Inactive (idle 3h 04m 05s)`.
Durations of 24 hours or more include days, for example
`Inactive (idle 7d 00h 14m 19s)`. Stats total time, session durations, and recent
session cards use the same duration format.

### Visual Activity Timeline (`timeline`)

Compare recent sessions on a shared time axis. Colors identify recorded user,
reasoning, assistant/model-output, tool, and turn-boundary events. Events with
recorded durations become blocks; instantaneous events remain visible as markers.
Blank regions are waiting or unavailable boundaries, never invented activity.

```bash
tokenmon timeline                         # compare the 10 most recent sessions
tokenmon timeline --window 1d --tasks 20 # choose history and lane count
tokenmon timeline --agent codex          # select one or more source agents
tokenmon timeline 01a10275                # visualize one session
tokenmon timeline --json | jq '.sessions'
```

Sessions are displayed longest first. `--compact` and `--wide` control label
density; redirected output uses distinct block characters without ANSI color.
The timeline JSON contains session lanes and ordered recorded-activity intervals.

### Chronological Event Logs (`logs`, `log`)

See the chronological step-by-step history of prompts, model thoughts, responses, and tool calls:

```bash
# View timeline of the latest session
tokenmon logs

# View timeline of a specific session ID or prefix
tokenmon logs 01a10275

# Follow only new events from the session that is latest at startup
tokenmon logs -f

# Follow a specific session ID/prefix, or an agent's latest session
tokenmon logs 01a10275 --follow
tokenmon logs codex -f

# Customize the polling interval (default: 2 seconds)
tokenmon logs -f --interval 1

# Export all session event logs from today in JSON format
tokenmon logs --window 1d --json
```

Follow mode starts at the current end: it prints a short session header, then only
newly observed timeline events. It keeps the initially selected session even when
another session becomes newer, and appends output without clearing the screen or
replaying history. Updates to an existing event's usage or timestamp do not print
that event again. Press Ctrl+C to stop.

`-f` and `--follow` work with `logs` and `log`. Follow selects one
session, so it cannot be combined with the batch export option `--window`; use
`--all` to select a session older than the default 30-day cutoff. The interval must
be a positive, finite number. Polling reads the selected session again rather than
following raw file bytes, which also supports SQLite-backed Antigravity sessions.

### Interactive Shell (`interactive`, `repl`, `shell`, `-i`)

Open an interactive terminal shell with live auto-refresh and tab completion:

```bash
tokenmon interactive
# or
tokenmon -i
```

Available commands inside the shell:
```text
(tokenmon) summary 30m        # view 30m throughput table
(tokenmon) sessions           # list active & inactive sessions
(tokenmon) timeline           # compare session activity lanes
(tokenmon) logs latest        # view step-by-step session events
(tokenmon) recent 15          # view latest generation speeds
(tokenmon) watch 2.0 1d       # metrics, recent streams, and session cards (Ctrl+C to stop)
(tokenmon) help               # list all commands
```

Shell timelines and logs prefer exact session IDs over prefix matches and can open exact
IDs outside the recent-session list. CLI exact-ID lookup also bypasses the
`--tasks` discovery limit; history cutoffs still apply, so use `--all` for older
sessions. Prefix matching searches only the discovered recent sessions.

---

## Adding New Adapters

TokenMon uses a base class in `src/tokenmon/adapters/base.py`:

```python
class BaseAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect(self) -> bool: ...

    @abstractmethod
    def collect(self, max_sessions: int = 64, min_timestamp: float | None = None) -> list[GenerationSpan]: ...

    @abstractmethod
    def collect_sessions(self, max_sessions: int = 32, min_timestamp: float | None = None) -> list[SessionTimeline]: ...
```

To add support for a new agent (e.g. **OpenCode**):
1. Create `src/tokenmon/adapters/opencode.py` subclassing `BaseAdapter`.
2. Implement discovery (`detect()`), stream parsing (`collect()`), and timelines (`collect_sessions()`).
3. Register it in `src/tokenmon/adapters/__init__.py`.

Adapters translate telemetry into shared `GenerationSpan` and
`SessionTimeline`/`TimelineEvent` models; the analyzer and CLI own aggregation,
formatting, and JSON serialization. Build spans with `create_span()` so invalid
measurements remain visible in exclusion counts. Honor collection limits and
completion-time cutoffs without discarding earlier metadata context.

For live follow, override `read_session(session_id)` to reopen a retained exact
source, and give events stable internal `event_id` values that do not change with
usage or timestamps. `OMPAdapter` and `PiAdapter` share
`src/tokenmon/adapters/_pi_jsonl.py` for nested journals and pinned reads; their
separate entry points preserve native roots and timing semantics.

---

## License

MIT

## Copied History and TPS

Workers and forked sessions may inherit earlier assistant outputs. **Ownership**
means counting those outputs only under the session that generated them—not again
under a child that copied them.

For example, 1,000 main-session tokens in 10s plus 200 worker tokens in 4s gives
**1,200 tokens and 85.7 weighted TPS**. Counting the copied main output again,
including its duration, gives **2,200 tokens and 91.7 TPS**: the main response gets
too much weight. Duplicating every response equally leaves weighted TPS unchanged,
but totals are still wrong. TokenMon uses recorded ownership boundaries where
available; see Agent Data above for format limitations.
