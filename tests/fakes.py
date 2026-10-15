"""Fake Anthropic SDK clients for offline tests."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any, Callable, Union

Reply = Union[dict, str, Exception, Callable[[dict], Any]]


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def thinking_block(text: str = "hmm") -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=text, signature="sig")


def message(content: Any = "", stop_reason: str = "end_turn",
            input_tokens: int = 11, output_tokens: int = 7) -> SimpleNamespace:
    if isinstance(content, dict) or (
        isinstance(content, list) and not all(hasattr(b, "type") for b in content)
    ):
        content = json.dumps(content)
    blocks = [text_block(content)] if isinstance(content, str) else content
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class _Messages:
    def __init__(self, owner: "FakeClient"):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._next(kwargs)


class _AsyncMessages(_Messages):
    async def create(self, **kwargs):
        return self._owner._next(kwargs)


class _Models:
    def __init__(self, owner: "FakeClient"):
        self._owner = owner

    def retrieve(self, model_id: str, **kwargs):
        self._owner.retrieved.append(model_id)
        if self._owner.retrieve_error is not None:
            raise self._owner.retrieve_error
        return SimpleNamespace(id=model_id, type="model")


class FakeClient:
    """
    Replies are consumed in order. Each reply is a message object, a dict/str
    (wrapped as a text-only message), an Exception (raised), or a callable
    taking the request kwargs.
    """

    def __init__(self, *replies: Any, asynchronous: bool = False):
        self._replies = list(replies)
        self._lock = threading.Lock()
        self.calls: list[dict] = []
        self.retrieved: list[str] = []
        self.retrieve_error: Exception | None = None
        self.messages = _AsyncMessages(self) if asynchronous else _Messages(self)
        self.models = _Models(self)

    def _next(self, kwargs: dict):
        with self._lock:
            self.calls.append(kwargs)
            if not self._replies:
                raise AssertionError("FakeClient ran out of replies")
            reply = self._replies.pop(0)
        if callable(reply) and not isinstance(reply, Exception):
            reply = reply(kwargs)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, (dict, list, str)):
            return message(reply)
        return reply
