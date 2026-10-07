"""
llm/config.py

Environment-based configuration for the Bedrock LLM client.

    AWS_REGION          Bedrock region                       (default: us-east-1)
    BEDROCK_MODEL_ID    Model or inference profile ID        (default: Claude Opus 5.5)
    LLM_MAX_TOKENS      Default output-token cap per call    (default: 4096)
    LLM_READ_TIMEOUT_S  Read timeout per request in seconds  (default: 300)
    LLM_MAX_RETRIES     Retries after the first attempt      (default: 6)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Copy the exact inference profile ID from the Bedrock console once
# Opus 5.5 access is enabled, and set BEDROCK_MODEL_ID if it differs.
DEFAULT_MODEL_ID       = "global.anthropic.claude-opus-5-5-v1:0"
DEFAULT_REGION         = "us-east-1"
DEFAULT_MAX_TOKENS     = 4096
DEFAULT_READ_TIMEOUT_S = 300
DEFAULT_MAX_RETRIES    = 6
CONNECT_TIMEOUT_S      = 10


@dataclass(frozen=True)
class LLMConfig:
    region:         str
    model_id:       str
    max_tokens:     int
    read_timeout_s: int
    max_retries:    int


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def load_config() -> LLMConfig:
    """Read the LLM configuration from the environment."""
    return LLMConfig(
        region=_env_str("AWS_REGION", DEFAULT_REGION),
        model_id=_env_str("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID),
        max_tokens=_env_int("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        read_timeout_s=_env_int("LLM_READ_TIMEOUT_S", DEFAULT_READ_TIMEOUT_S),
        max_retries=_env_int("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES),
    )
