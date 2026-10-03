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

- **Real Output Speed**: Measures actual streaming speed (TPS), ignoring tool execution and network idle pauses.
- **Zero Extra Dependencies**: Runs on standard Python 3.10+ without installing third-party packages.
- **Strictly Read-Only**: Safely opens local files and SQLite databases in read-only mode (`?mode=ro`). Never locks or changes your logs.
- **Meaningful Averages**: Calculates true weighted speed ($\frac{\text{total tokens}}{\text{total time}}$), median speed, and min/max ranges over rolling time windows (`30m`, `1d`, `7d`, `30d`, `all`).
- **Supports Popular Agents**: Auto-detects **Codex** (`~/.codex`), **Claude Code** (`~/.claude/projects/`), and **Antigravity** (`~/.gemini/antigravity-cli`).
- **Session Timelines**: Step-by-step history of user prompts, thinking, assistant responses, and tool calls.
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

TokenMon uses simple Docker-style subcommands: `stats` (default), `ps` (sessions), `logs` (timelines), and `interactive` (shell).

Running `tokenmon` by itself defaults directly to `stats`.
The examples below use the installed command; with uvx, use `uvx tokenmon` in its place.

### Generation Speed & Metrics (`stats`, `top`, default)

View token throughput (TPS), rolling averages, and recent generation outputs:

```bash
# Auto-detect local agents and show stats (default)
tokenmon
# or explicitly
tokenmon stats

# Filter to a specific time window (30m, 1d, 7d, 30d, all)
tokenmon stats --window 1d

# Refresh the full dashboard every 2 seconds (Ctrl+C to stop)
tokenmon stats --watch
tokenmon stats --watch --window 1d

# Customize the refresh interval
tokenmon stats --watch --window 1d --interval 5

# Include full history beyond the default 30-day cutoff
tokenmon stats --all

# Target a specific agent or custom folder
tokenmon stats codex
tokenmon stats claude --home ~/.claude

# Compact layout for narrow panes or wide full-detail table
tokenmon stats --compact
tokenmon stats --wide
```

Watch mode refreshes throughput metrics, recent generation streams, and recent
session cards together. It redraws the screen in a terminal; redirected output
appends complete snapshots. Agent selection, `--home`, `--tasks`, `--recent`,
`--compact`, and `--wide` also work with `--watch`.

Time filtering uses the explicit `--window` option across commands. The former
`-w` shortcut has been removed; use `--window 1d` in existing commands and scripts.

### Active & Recent Sessions (`sessions`, `ps`, `ls`)

See all recent sessions, message counts, tool runs, and idle status:

```bash
# List recent sessions
tokenmon ps

# Filter sessions within a time window
tokenmon ps --window 1d

# Filter to a specific agent
tokenmon ps codex
```

### Event Timelines (`timeline`, `logs`, `log`)

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

# Export all session timelines from today in JSON format
tokenmon timeline --window 1d --json
```

Follow mode starts at the current end: it prints a short session header, then only
newly observed timeline events. It keeps the initially selected session even when
another session becomes newer, and appends output without clearing the screen or
replaying history. Updates to an existing event's usage or timestamp do not print
that event again. Press Ctrl+C to stop.

`-f` and `--follow` work with `logs`, `log`, and `timeline`. Follow selects one
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
(tokenmon) timeline latest    # view step-by-step event timeline
(tokenmon) recent 15          # view latest generation speeds
(tokenmon) watch 2.0 1d       # metrics, recent streams, and session cards (Ctrl+C to stop)
(tokenmon) help               # list all commands
```

### Machine-Readable JSON Output

Stats, sessions, and historical timelines support `--json` for easy scripting.
Live `--watch` and `--follow` modes require human output and reject `--json`:

```bash
tokenmon stats --json | jq .
tokenmon ps --json | jq .
tokenmon logs 01a10275 --json | jq .
tokenmon timeline --window 1d --json | jq .
```

---

## Development

Clone the repository to work on TokenMon:

```bash
git clone https://github.com/quanhua92/tokenmon.git
cd tokenmon
uv run tokenmon
uv run python -m unittest discover -s tests
```

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

---

## License

MIT
