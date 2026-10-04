"""Oh My Pi adapter: read-only parsing of main and nested worker journals."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from tokenmon.adapters._pi_jsonl import _PiJSONLAdapter

logger = logging.getLogger(__name__)

_PROFILE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_RESERVED_PROFILE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\..*)?\Z", re.IGNORECASE)


def _resolve_root(custom_root: Path | None) -> Path:
    if custom_root is not None:
        return custom_root
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if agent_dir and agent_dir.strip():
        return Path(agent_dir).expanduser().resolve()
    profile = os.environ.get("OMP_PROFILE", os.environ.get("PI_PROFILE", "")).strip()
    if profile == "default":
        profile = ""
    if profile and (not _PROFILE.fullmatch(profile) or profile.endswith(".") or _RESERVED_PROFILE.fullmatch(profile)):
        logger.warning("OMPAdapter: ignoring invalid profile name %r", profile)
        profile = ""
    config_name = os.environ.get("PI_CONFIG_DIR") or ".omp"
    config_root = Path.home() / config_name.lstrip("/\\")
    if profile:
        config_root = config_root / "profiles" / profile
    xdg_home = os.environ.get("XDG_DATA_HOME")
    if xdg_home and sys.platform in {"linux", "darwin"}:
        data_root = Path(xdg_home).expanduser() / "omp"
        if profile:
            data_root = data_root / "profiles" / profile
        if data_root.is_dir():
            return data_root.resolve()
    return (config_root / "agent").resolve()


class OMPAdapter(_PiJSONLAdapter):
    """Read recorded OMP output windows and session activity without modifying logs."""

    @property
    def name(self) -> str:
        return "omp"

    @property
    def root(self) -> Path:
        return _resolve_root(self._custom_root)
