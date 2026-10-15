"""
agents/llm_client.py

The single module that talks to the language model (ADR 0001).

Every LLM request in the system goes to Claude through the Anthropic Messages
API via the official `anthropic` SDK:
  - generate()   sync, schema-constrained structured output (agents)
  - complete()   sync, free text (legacy RAG pipeline, smoke test)
  - acomplete()  async, free text + token usage (/api/chat comparator)
  - check_health()  key check + model metadata lookup, no tokens spent

Configuration is read at first use, not at import:
  ANTHROPIC_API_KEY  required; never logged or returned
  ANTHROPIC_MODEL    optional; defaults to DEFAULT_MODEL

Every request pins effort to "medium" and sends no temperature/top_p/top_k or
thinking configuration: the default model rejects non-default sampling and
cannot disable thinking. A reply that stops at max_tokens raises
LLMTruncatedError before anything is parsed.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, TypeVar

import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL   = "claude-opus-5-5"
EFFORT          = "medium"
REQUEST_TIMEOUT = 600.0
API_KEY_ENV     = "ANTHROPIC_API_KEY"
MODEL_ENV       = "ANTHROPIC_MODEL"

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for errors raised by this module."""


class LLMConfigError(LLMError):
    """Required configuration (the API key) is missing."""


class LLMTruncatedError(LLMError):
    """The model stopped at the output limit; the partial reply is discarded."""


class LLMRefusalError(LLMError):
    """The model refused to answer."""


@dataclass(frozen=True)
class Usage:
    input_tokens:  int
    output_tokens: int


# ── Clients (lazy singletons; tests inject fakes via set_clients) ─────────────

_lock = threading.Lock()
_sync_client: Any = None
_async_client: Any = None


def set_clients(sync: Any = None, async_: Any = None) -> None:
    """Test hook: replace the SDK clients with fakes."""
    global _sync_client, _async_client
    with _lock:
        _sync_client = sync
        _async_client = async_


def reset_clients() -> None:
    set_clients(None, None)


def get_api_key() -> str:
    key = (os.environ.get(API_KEY_ENV) or "").strip()
    if not key:
        raise LLMConfigError(f"{API_KEY_ENV} is not set")
    return key


def get_model() -> str:
    return (os.environ.get(MODEL_ENV) or "").strip() or DEFAULT_MODEL


def _client() -> Any:
    global _sync_client
    key = get_api_key()
    with _lock:
        if _sync_client is None:
            _sync_client = anthropic.Anthropic(api_key=key, timeout=REQUEST_TIMEOUT)
        return _sync_client


def _aclient() -> Any:
    global _async_client
    key = get_api_key()
    with _lock:
        if _async_client is None:
            _async_client = anthropic.AsyncAnthropic(api_key=key, timeout=REQUEST_TIMEOUT)
        return _async_client


# ── Request / response helpers ────────────────────────────────────────────────

def _redact(text: str) -> str:
    key = (os.environ.get(API_KEY_ENV) or "").strip()
    return text.replace(key, "***") if key else text


def _request(system: str, user: str, max_tokens: int,
             output_format: Optional[dict] = None) -> dict:
    output_config: dict[str, Any] = {"effort": EFFORT}
    if output_format is not None:
        output_config["format"] = output_format
    return {
        "model":         get_model(),
        "system":        system,
        "messages":      [{"role": "user", "content": user}],
        "max_tokens":    max_tokens,
        "output_config": output_config,
    }


def _text(message: Any) -> str:
    # Thinking blocks can precede text blocks; select by type.
    return "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )


def _check_stop(message: Any, max_tokens: int) -> None:
    reason = getattr(message, "stop_reason", None)
    if reason in ("max_tokens", "model_context_window_exceeded"):
        raise LLMTruncatedError(
            f"LLM reply truncated (stop_reason={reason}, max_tokens={max_tokens})"
        )
    if reason == "refusal":
        raise LLMRefusalError("LLM refused to answer (stop_reason=refusal)")


def _usage(message: Any) -> Usage:
    usage = getattr(message, "usage", None)
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )


def _log_done(kind: str, message: Any, t0: float) -> None:
    u = _usage(message)
    logger.info(
        f"LLM {kind} done in {time.time() - t0:.1f}s "
        f"(stop_reason={getattr(message, 'stop_reason', None)}, "
        f"in={u.input_tokens}, out={u.output_tokens})"
    )


def _log_error(kind: str, err: Exception) -> None:
    logger.error(f"LLM {kind} failed: {type(err).__name__}: {_redact(str(err))}")


# ── Public API ────────────────────────────────────────────────────────────────

def generate(system: str, user: str, *, schema: type[ModelT], max_tokens: int) -> ModelT:
    """
    Schema-constrained call. The JSON schema derived from `schema` is sent as
    structured output; the reply is validated with `schema` afterwards, so a
    pydantic.ValidationError propagates to the caller.
    """
    output_format = {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
    kwargs = _request(system, user, max_tokens, output_format)
    t0 = time.time()
    try:
        message = _client().messages.create(**kwargs)
    except LLMError:
        raise
    except Exception as e:
        _log_error("generate", e)
        raise
    _log_done(f"generate[{schema.__name__}]", message, t0)
    _check_stop(message, max_tokens)
    return schema.model_validate_json(_text(message))


def complete(system: str, user: str, *, max_tokens: int) -> str:
    """Free-text call; returns the concatenated text blocks."""
    kwargs = _request(system, user, max_tokens)
    t0 = time.time()
    try:
        message = _client().messages.create(**kwargs)
    except LLMError:
        raise
    except Exception as e:
        _log_error("complete", e)
        raise
    _log_done("complete", message, t0)
    _check_stop(message, max_tokens)
    return _text(message)


async def acomplete(system: str, user: str, *, max_tokens: int) -> tuple[str, Usage]:
    """Async free-text call; returns the text and token usage."""
    kwargs = _request(system, user, max_tokens)
    t0 = time.time()
    try:
        message = await _aclient().messages.create(**kwargs)
    except LLMError:
        raise
    except Exception as e:
        _log_error("acomplete", e)
        raise
    _log_done("acomplete", message, t0)
    _check_stop(message, max_tokens)
    return _text(message), _usage(message)


def check_health() -> dict:
    """Confirm the key is set and the model is reachable. Spends no tokens."""
    try:
        client = _client()
    except LLMConfigError as e:
        return {"status": "error", "detail": str(e)}
    model = get_model()
    try:
        info = client.models.retrieve(model)
        return {"status": "ok", "model": getattr(info, "id", model)}
    except Exception as e:
        _log_error("health check", e)
        return {"status": "error", "model": model, "detail": _redact(f"{type(e).__name__}: {e}")}
