"""Adapters package: registry and discovery for supported agent sources."""

from __future__ import annotations

from typing import Type

from llm_monitor.adapters.base import BaseAdapter
from llm_monitor.adapters.claude import ClaudeAdapter
from llm_monitor.adapters.codex import CodexAdapter

ADAPTER_REGISTRY: dict[str, Type[BaseAdapter]] = {
    "codex": CodexAdapter,
    "claude": ClaudeAdapter,
}


def get_adapter(name: str, **kwargs) -> BaseAdapter | None:
    """Instantiate an adapter by name."""
    cls = ADAPTER_REGISTRY.get(name.lower())
    if cls:
        return cls(**kwargs)
    return None


def detect_available_adapters(**kwargs) -> list[BaseAdapter]:
    """Return instances of all adapters whose data directories exist on this system."""
    available = []
    for cls in ADAPTER_REGISTRY.values():
        instance = cls(**kwargs)
        if instance.detect():
            available.append(instance)
    return available
