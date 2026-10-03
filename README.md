# llm-monitor

A zero-dependency, read-only CLI monitor for local AI coding agent telemetry, streaming generation latency, and token throughput (Tokens Per Second, TPS).

Built from first principles to isolate true model generation speed from tool execution and network idle time.

---

## Features

- **True Generation TPS**: Isolates actual streaming output duration, discarding tool running time, linters, and network queueing latency.
- **Zero Third-Party Dependencies**: Pure Python 3 standard library (`sqlite3`, `json`, `pathlib`, `re`, `argparse`, `statistics`, `dataclasses`). Runs instantly anywhere.
- **Strictly Read-Only**: Connects to SQLite databases with `?mode=ro` and executes `PRAGMA query_only = ON`. Never writes to or locks your agent logs.
- **Mathematically Sound Aggregation**: Calculates true **Weighted Throughput** ($\frac{\sum \text{tokens}}{\sum \text{duration}}$), median speed, and min-max distribution across rolling time windows (`30m`, `1d`, `7d`, `all`).
- **Multi-Agent Adapters**: Native support for **Codex** (`~/.codex`) and **Claude Code** (`~/.claude/projects/`).
- **Session Timelines & Unattended Detection**: Chronological inspection of user prompts, thinking blocks, assistant answers, and tool executions.
- **Machine-Readable**: Includes `--json` mode for easy piping into `jq`, automated alerts, or reporting dashboards.

---

## Installation

`llm-monitor` requires Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/quanhua92/llm-monitor.git
cd llm-monitor

# Run directly via uv (auto-creates venv and installs console script)
uv run llm-monitor

# Or run tests
uv run python -m unittest discover -s tests
```

---

## Usage

`llm-monitor` follows an intuitive **Docker-style** CLI workflow with first-class subcommands and familiar aliases (`stats`/`top`, `sessions`/`ps`, `timeline`/`logs`, `interactive`/`repl`).

Bare `llm-monitor` (with or without flags) directly defaults to `stats`.

### Throughput & Performance (`stats`, `top`, default)
Inspect rolling generation throughput (TPS), median speeds, and recent output streams:

```bash
# Auto-detect local agents and show stats (default)
uv run llm-monitor
# or explicitly
uv run llm-monitor stats

# Filter to a specific rolling window (30m, 1d, 7d, 30d, all)
uv run llm-monitor stats --window 1d

# Include full history without 30-day cutoff
uv run llm-monitor stats --all

# Target a specific agent (codex, claude, antigravity) or custom path
uv run llm-monitor stats codex
uv run llm-monitor stats claude --home ~/.claude

# Concise layout or wide diagnostic table
uv run llm-monitor stats --compact
uv run llm-monitor stats --wide
```

### Active & Recent Sessions (`sessions`, `ps`, `ls`)
List active sessions, user turn counts, assistant responses, tool executions, and idle status:

```bash
# List recent sessions
uv run llm-monitor sessions

# Or using the Docker alias 'ps'
uv run llm-monitor ps

# Filter sessions within a rolling window
uv run llm-monitor ps --window 1d

# Filter to a specific agent
uv run llm-monitor ps codex
```

### Chronological Event Timelines (`timeline`, `log`, `logs`)
Inspect the chronological step-by-step event stream of user prompts, assistant thoughts/answers, tool executions, and turn completions:

```bash
# View timeline of the latest session
uv run llm-monitor timeline
# or using the Docker alias 'logs'
uv run llm-monitor logs

# View timeline of a specific session ID or prefix
uv run llm-monitor timeline 01a10275
uv run llm-monitor logs 01a10275

# Batch export all session timelines within a window (e.g. today's sessions)
uv run llm-monitor timeline --window 1d --json
```

### Interactive Terminal Shell (`interactive`, `repl`, `shell`, `-i`)
Launch an interactive shell with tab completion, query commands, and live auto-refresh dashboard:

```bash
uv run llm-monitor interactive
# or
uv run llm-monitor -i
```

Inside the shell:
```text
(llm-monitor) summary 30m        # view 30m throughput table
(llm-monitor) sessions           # list active & inactive sessions
(llm-monitor) timeline latest    # view step-by-step event timeline
(llm-monitor) recent 15          # inspect recent generation speeds
(llm-monitor) watch 2.0 1d       # live auto-refresh dashboard (Ctrl+C to return)
(llm-monitor) help               # list all commands
```

### Machine-Readable JSON Output
All subcommands support `--json` for easy piping into `jq`, automated alerts, or reporting dashboards:

```bash
uv run llm-monitor stats --json | jq .
uv run llm-monitor ps --json | jq .
uv run llm-monitor logs 01a10275 --json | jq .
uv run llm-monitor timeline --window 1d --json | jq .
```

---

## Example Output

```text
⚡ llm-monitor v0.1.0 [Agents: codex]
📊 Inspected: 48 output streams across up to 64 sessions

🤖 Model: gpt-5
┌─────────────┬───────────────┬────────┬──────────┬──────────────┬────────────┬───────────────┐
│ Time Window │ Valid / Total │ Tokens │ Time (s) │ Weighted TPS │ Median TPS │ Min - Max TPS │
├─────────────┼───────────────┼────────┼──────────┼──────────────┼────────────┼───────────────┤
│ 30m         │ 4 / 4         │ 2,410  │ 32.10    │ 75.1         │ 74.2       │ 68.0 - 82.5   │
│ 1d          │ 18 / 20       │ 12,500 │ 178.40   │ 70.1         │ 71.0       │ 55.4 - 88.0   │
│ 7d          │ 42 / 48       │ 34,200 │ 502.94   │ 68.0         │ 69.5       │ 49.0 - 91.2   │
│ all         │ 42 / 48       │ 34,200 │ 502.94   │ 68.0         │ 69.5       │ 49.0 - 91.2   │
└─────────────┴───────────────┴────────┴──────────┴──────────────┴────────────┴───────────────┘

📋 Recent 3 Generation Streams:
  [2026-10-03 14:10:15] gpt-5              :   82.5 TPS  (  660 tokens in   8.00s) [stream-log]
  [2026-10-03 14:18:22] gpt-5              :   74.2 TPS  (  480 tokens in   6.47s) [item-event]
  [2026-10-03 14:25:01] gpt-5              :   68.0 TPS  (  810 tokens in  11.91s) [stream-log]
```

---

## Architecture & Adding Adapters

`llm-monitor` uses an abstract adapter contract in `src/llm_monitor/adapters/base.py`:

```python
class BaseAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect(self) -> bool: ...

    @abstractmethod
    def collect(self, max_sessions: int = 64) -> list[GenerationSpan]: ...
```

To add support for a new agent (e.g. **Claude Code**):
1. Create `src/llm_monitor/adapters/claude.py` subclassing `BaseAdapter`.
2. Implement log discovery (`detect()`) and stream parsing (`collect()`).
3. Register it in `src/llm_monitor/adapters/__init__.py`.

---

## License

MIT
