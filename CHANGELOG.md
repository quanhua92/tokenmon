# Changelog

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

[0.1.1]: https://github.com/quanhua92/tokenmon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/quanhua92/tokenmon/releases/tag/v0.1.0
