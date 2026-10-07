"""Shared Amazon Bedrock LLM client used by agents, RAG pipeline and API."""

from llm.client import (
    LLMProviderError,
    LLMResponse,
    StructuredOutputError,
    generate_structured,
    generate_text,
    get_client,
    get_config,
    reset,
    set_client,
)
from llm.config import LLMConfig, load_config
from llm.health import check_llm_health

__all__ = [
    "LLMConfig",
    "LLMProviderError",
    "LLMResponse",
    "StructuredOutputError",
    "check_llm_health",
    "generate_structured",
    "generate_text",
    "get_client",
    "get_config",
    "load_config",
    "reset",
    "set_client",
]
