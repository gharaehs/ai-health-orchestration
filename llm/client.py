"""
llm/client.py

Single shared client for all language-model calls (agents, RAG pipeline,
comparator, health check). Uses the Amazon Bedrock Runtime Converse API.

Authentication uses the standard AWS credential chain (EC2 instance role in
production). Throttling and transient errors are retried by botocore's
adaptive retry mode, which also rate-limits the shared client across threads.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel

from llm.config import CONNECT_TIMEOUT_S, LLMConfig, load_config

logger = logging.getLogger(__name__)

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


class LLMProviderError(RuntimeError):
    """The model provider (Amazon Bedrock) failed the request after retries."""

    def __init__(self, message: str, *, cause: Optional[BaseException] = None):
        super().__init__(f"Bedrock LLM provider error: {message}")
        self.cause = cause


class StructuredOutputError(ValueError):
    """The model did not return a usable structured (tool) response."""


@dataclass
class LLMResponse(Generic[T]):
    content:           T
    prompt_tokens:     int
    completion_tokens: int
    stop_reason:       Optional[str] = None


_lock = threading.Lock()
_config: Optional[LLMConfig] = None
_client: Any = None


def get_config() -> LLMConfig:
    global _config
    if _config is None:
        with _lock:
            if _config is None:
                _config = load_config()
    return _config


def build_client(config: LLMConfig):
    return boto3.client(
        "bedrock-runtime",
        region_name=config.region,
        config=Config(
            retries={"mode": "adaptive", "max_attempts": config.max_retries},
            read_timeout=config.read_timeout_s,
            connect_timeout=CONNECT_TIMEOUT_S,
        ),
    )


def get_client():
    """Return the process-wide bedrock-runtime client (thread-safe)."""
    global _client
    if _client is None:
        config = get_config()
        with _lock:
            if _client is None:
                _client = build_client(config)
    return _client


def set_client(client, config: Optional[LLMConfig] = None) -> None:
    """Override the shared client (and optionally config). Intended for tests."""
    global _client, _config
    with _lock:
        _client = client
        if config is not None:
            _config = config


def reset() -> None:
    """Drop the cached client and config so they are rebuilt from the env."""
    global _client, _config
    with _lock:
        _client = None
        _config = None


# ── Public API ────────────────────────────────────────────────────────────────

def generate_text(
    system: str,
    user: str,
    max_tokens: Optional[int] = None,
) -> LLMResponse[str]:
    """Generate free text. Returns the concatenated text blocks plus usage."""
    response = _converse(system, user, max_tokens)
    blocks = response["output"]["message"]["content"]
    text = "".join(b["text"] for b in blocks if "text" in b)
    return _wrap(response, text)


def generate_structured(
    system: str,
    user: str,
    output_model: type[M],
    max_tokens: Optional[int] = None,
) -> LLMResponse[M]:
    """
    Generate output conforming to `output_model` via a forced tool call.

    Raises:
        pydantic.ValidationError: tool input does not match the schema.
        StructuredOutputError:    no tool call returned (e.g. truncated output).
        LLMProviderError:         Bedrock failed after retries.
    """
    tool_name = tool_name_for(output_model)
    tool_config = {
        "tools": [{
            "toolSpec": {
                "name": tool_name,
                "description": (
                    f"Submit the final {output_model.__name__} result. "
                    "Call this tool exactly once with the complete result."
                ),
                "inputSchema": {"json": inline_schema(output_model.model_json_schema())},
            }
        }],
        "toolChoice": {"tool": {"name": tool_name}},
    }
    response = _converse(system, user, max_tokens, tool_config=tool_config)

    blocks = response["output"]["message"]["content"]
    tool_inputs = [
        b["toolUse"]["input"] for b in blocks
        if "toolUse" in b and b["toolUse"].get("name") == tool_name
    ]
    if not tool_inputs:
        raise StructuredOutputError(
            f"Model returned no '{tool_name}' tool call "
            f"(stopReason={response.get('stopReason')})"
        )

    output = output_model.model_validate(tool_inputs[0])
    return _wrap(response, output)


def tool_name_for(output_model: type[BaseModel]) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", output_model.__name__).lower()
    return f"submit_{snake}"[:64]


def inline_schema(schema: dict) -> dict:
    """Resolve local `#/$defs/...` references and drop `$defs`."""
    defs = schema.get("$defs", {})

    def resolve(node, stack: tuple[str, ...]):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.split("/")[-1]
                if name in stack:
                    raise ValueError(f"Recursive schema reference '{name}' cannot be inlined")
                target = copy.deepcopy(defs[name])
                extras = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**target, **extras}, stack + (name,))
            return {k: resolve(v, stack) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, stack) for v in node]
        return node

    return resolve(schema, ())


# ── Internals ─────────────────────────────────────────────────────────────────

def _converse(
    system: str,
    user: str,
    max_tokens: Optional[int],
    tool_config: Optional[dict] = None,
) -> dict:
    config = get_config()
    kwargs: dict[str, Any] = {
        "modelId": config.model_id,
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens or config.max_tokens},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    if tool_config:
        kwargs["toolConfig"] = tool_config

    t0 = time.time()
    try:
        response = get_client().converse(**kwargs)
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Bedrock converse failed for model {config.model_id}: {e}")
        raise LLMProviderError(str(e), cause=e) from e

    usage = response.get("usage", {})
    logger.info(
        f"Bedrock converse ok in {time.time() - t0:.1f}s "
        f"(in={usage.get('inputTokens')}, out={usage.get('outputTokens')}, "
        f"stop={response.get('stopReason')})"
    )
    return response


def _wrap(response: dict, content: T) -> LLMResponse[T]:
    usage = response.get("usage", {})
    return LLMResponse(
        content=content,
        prompt_tokens=int(usage.get("inputTokens", 0)),
        completion_tokens=int(usage.get("outputTokens", 0)),
        stop_reason=response.get("stopReason"),
    )
