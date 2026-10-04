"""Adapters package: registry and discovery for supported agent sources."""

from __future__ import annotations

from typing import Type

from tokenmon.adapters.antigravity import AntigravityAdapter
from tokenmon.adapters.base import BaseAdapter
from tokenmon.adapters.claude import ClaudeAdapter
from tokenmon.adapters.codex import CodexAdapter
from tokenmon.adapters.omp import OMPAdapter
from tokenmon.adapters.pi import PiAdapter
from tokenmon.adapters.opencode import OpenCodeAdapter

import logging
import time

logger = logging.getLogger(__name__)

ADAPTER_REGISTRY: dict[str, Type[BaseAdapter]] = {
    "codex": CodexAdapter,
    "claude": ClaudeAdapter,
    "antigravity": AntigravityAdapter,
    "agy": AntigravityAdapter,
    "omp": OMPAdapter,
    "pi": PiAdapter,
    "opencode": OpenCodeAdapter,
}


def get_adapter(name: str, **kwargs) -> BaseAdapter | None:
    """Instantiate an adapter by name."""
    cls = ADAPTER_REGISTRY.get(name.lower())
    if cls:
        return cls(**kwargs)
    return None


def detect_available_adapters(max_age_days: float = 30.0, **kwargs) -> list[BaseAdapter]:
    """Return instances of all adapters that exist and were active within max_age_days (default 30d).

    If no adapters were active within max_age_days, falls back to the most
    recently active adapter, or all detected adapters if timestamps are unavailable.
    """
    detected_with_time: list[tuple[float, BaseAdapter]] = []
    fallback: list[BaseAdapter] = []
    now = time.time()
    cutoff = now - (max_age_days * 86400.0)

    # Iterate over unique adapter classes (avoiding duplicates from aliases)
    seen_classes = set()
    for cls in ADAPTER_REGISTRY.values():
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        try:
            instance = cls(**kwargs)
            if instance.detect():
                fallback.append(instance)
                try:
                    last_ts = instance.last_generation_timestamp()
                    if last_ts is not None:
                        detected_with_time.append((last_ts, instance))
                except Exception as e:
                    logger.debug("Adapter '%s' error getting last generation timestamp: %s", instance.name, e)
        except Exception as e:
            logger.warning("Adapter error during detection: %s", e)

    if not detected_with_time:
        return fallback

    # Filter to adapters active within max_age_days
    recent = [adapter for ts, adapter in detected_with_time if ts >= cutoff]
    if recent:
        # Sort by most recent activity descending
        recent_tuples = [t for t in detected_with_time if t[0] >= cutoff]
        recent_tuples.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in recent_tuples]

    # If none were active within max_age_days, pick the most recently active one
    detected_with_time.sort(key=lambda item: item[0], reverse=True)
    return [detected_with_time[0][1]]
