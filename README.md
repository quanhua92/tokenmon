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

### Auto-Detect & Monitor
By default, `llm-monitor` automatically detects local agent data (e.g. `~/.codex`):

```bash
uv run llm-monitor
```

### Time Window Filtering
Filter telemetry to specific rolling windows:

```bash
# 30-minute rolling window
uv run llm-monitor --window 30m

# 24-hour rolling window
uv run llm-monitor --window 1d

# 7-day rolling window
uv run llm-monitor --window 7d

# All-time window
uv run llm-monitor --window all
```

### Inspect Recent Streams
Display the latest generation streams with instantaneous speeds:

```bash
uv run llm-monitor --recent 20
```

### Session Activity & Unattended Status
List all recent sessions, user prompts, assistant turns, tool executions, and whether a session is currently active or left unattended:

```bash
uv run llm-monitor --sessions
```

### Chronological Event Timeline
View the full step-by-step chronology of user prompts, assistant answers, thinking, and tool runs for any session:

```bash
# View timeline of the latest session
uv run llm-monitor --timeline

# View timeline of a specific session ID
uv run llm-monitor --timeline 4a8b1c2d
```

### Interactive Terminal Shell (`cmd.Cmd`)
Launch an interactive REPL shell with tab completion and live monitoring:

```bash
uv run llm-monitor -i
# or
uv run llm-monitor interactive
```

Inside the shell:
```text
(llm-monitor) summary 30m        # view 30m throughput table
(llm-monitor) sessions           # list active & unattended sessions
(llm-monitor) timeline latest    # view step-by-step event timeline
(llm-monitor) recent 15          # inspect recent generation speeds
(llm-monitor) watch 2.0 1d       # live auto-refresh dashboard (Ctrl+C to return)
(llm-monitor) help               # list all commands
```

### Target a Specific Agent or Custom Data Path
```bash
# Target Codex specifically
uv run llm-monitor codex
uv run llm-monitor codex --home ~/.codex

# Target Claude Code specifically
uv run llm-monitor claude
uv run llm-monitor claude --home ~/.claude
```

### JSON Output
Pipe structured telemetry into scripts or monitoring dashboards:

```bash
uv run llm-monitor --json | jq .
uv run llm-monitor --sessions --json | jq .
uv run llm-monitor --timeline --json | jq .
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
