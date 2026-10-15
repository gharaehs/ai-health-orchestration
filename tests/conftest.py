"""
Offline test harness (D12).

Every test runs with a dummy ANTHROPIC_API_KEY, fresh LLM client singletons
and outbound socket connections blocked, so an accidental live call fails
the test instead of reaching the Anthropic API.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "services" / "api"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _stub_module(name: str, **attrs) -> None:
    """Register a stub only when the real package is not installed."""
    root = name.split(".")[0]
    if root not in sys.modules and importlib.util.find_spec(root) is not None:
        return
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        mod_name = ".".join(parts[:i])
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    for key, value in attrs.items():
        setattr(sys.modules[name], key, value)


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("stubbed dependency is not available in unit tests")


# Heavy runtime dependencies that unit tests never exercise.
_stub_module("mcp.client.session", ClientSession=_Unavailable)
_stub_module("mcp.client.streamable_http", streamablehttp_client=_Unavailable)
_stub_module("chromadb", HttpClient=_Unavailable)
_stub_module("sentence_transformers", SentenceTransformer=_Unavailable)


class NetworkBlockedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-dummy-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    def _blocked(self, *args, **kwargs):
        raise NetworkBlockedError("network access is blocked in unit tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)

    from agents import llm_client
    llm_client.reset_clients()
    yield
    llm_client.reset_clients()
