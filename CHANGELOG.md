# Changelog

## Unreleased

- Format session Status idle times as hours, minutes, and seconds across session lists, cards, and timelines; include days for durations of 24 hours or more.
- Use the shared duration format for recent session cards, including seconds and days when applicable.
- Add read-only Oh My Pi (`omp`) support for main and nested worker stats, sessions, timelines, watch, and pinned follow.
- Measure OMP's recorded first-output window using duration minus TTFT; exclude unconfirmed timing, tool-first boundaries, and incomplete responses.
- Preserve OMP historical settings and stable live identities, avoid inherited/task usage double-counting, and isolate malformed records and unreadable sources.
- Add native Pi (`pi`) sessions, token totals, historical settings, timelines, watch, and pinned follow using a shared read-only Pi-family journal parser.
- Keep Pi responses in exclusion counts without inventing generation timing or TPS; distinguish native Pi and OMP model-change schemas during discovery.

## [0.1.3] — 2026-10-04

- Format stats duration as hours, minutes, and seconds while keeping JSON durations numeric.
- Show optional recorded reasoning effort and speed mode on streams, sessions, and timeline events; include raw metadata in every JSON view and distinct configurations in stats windows.
- Ignore repeated Codex cumulative usage snapshots instead of assigning previous response tokens to later reasoning.
- Read the native Codex diagnostic log schema and preserve context across Codex and Claude completion-window cutoffs.
- Carry Antigravity model and optional generation settings across step ranges; retain invalid spans for exclusion counts and tolerate malformed protobuf records.
- Clarify that Claude single-record turn-span timing can include latency and waiting.

## [0.1.2] — 2026-10-04

- Add `stats -w` / `--watch` with a configurable `--interval` (default: 2 seconds).
- Share the stats dashboard with interactive watch, including recent generation streams and session cards.
- Show recent sessions even when no generation streams are available.
- Add `logs -f` / `--follow` to append only newly observed events from the session selected at startup.
- Keep follow attached to its selected source and suppress repeated events after streamed metadata updates.
- Repurpose `-w` as the stats watch shortcut; use the explicit `--window` option for time filtering across commands.
- Reject `--json` in live modes and batch `--window` exports with `--follow`.

## [0.1.1] — 2026-10-04

- Document `uvx`, `uv tool install`, and `pip install` as the primary ways to use TokenMon.
- Use the installed `tokenmon` command in usage examples; move cloning instructions to Development.
- Simplify the package description to match the README introduction.
- Move CI, publishing, and PyPI metadata instructions to `.github/RELEASING.md`.
- Add this changelog. CLI behavior is unchanged.

## [0.1.0] — 2026-10-04

Initial public release of TokenMon.

- Monitor Codex, Claude Code, and Antigravity telemetry locally with no runtime dependencies.
- Report generation throughput, weighted averages, and rolling-window statistics.
- Display session activity, token usage, working directories, and event timelines.
- Provide an interactive shell, compact and wide layouts, and JSON output.
- Read agent logs and SQLite databases without modifying them.
- Add CI checks and manual PyPI trusted publishing.

[0.1.3]: https://github.com/quanhua92/tokenmon/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/quanhua92/tokenmon/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/quanhua92/tokenmon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/quanhua92/tokenmon/releases/tag/v0.1.0
