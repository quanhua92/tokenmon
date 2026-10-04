"""Native Pi adapter: read-only session activity without invented generation timing."""

from __future__ import annotations

import os
from pathlib import Path

from tokenmon.adapters._pi_jsonl import _PiJSONLAdapter


class PiAdapter(_PiJSONLAdapter):
    """Read Pi journals; unconfirmed output timing remains excluded from TPS."""

    @property
    def name(self) -> str:
        return "pi"

    @property
    def root(self) -> Path:
        if self._custom_root is not None:
            return self._custom_root
        agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
        if agent_dir and agent_dir.strip():
            return Path(agent_dir).expanduser().resolve()
        return (Path.home() / ".pi" / "agent").resolve()

    def last_generation_timestamp(self, max_probe_sessions: int = 5) -> float | None:
        """Probe recorded session activity even before any assistant response."""
        sessions = self.collect_sessions(max_sessions=max_probe_sessions)
        return max((session.updated_at for session in sessions), default=None)
