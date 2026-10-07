import pytest

from llm import client as llm_client
from llm.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_READ_TIMEOUT_S,
    DEFAULT_REGION,
    load_config,
)

ENV_VARS = ["AWS_REGION", "BEDROCK_MODEL_ID", "LLM_MAX_TOKENS", "LLM_READ_TIMEOUT_S", "LLM_MAX_RETRIES"]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_when_env_unset(clean_env):
    cfg = load_config()
    assert cfg.region == DEFAULT_REGION
    assert cfg.model_id == DEFAULT_MODEL_ID
    assert "opus-5-5" in cfg.model_id
    assert cfg.max_tokens == DEFAULT_MAX_TOKENS
    assert cfg.read_timeout_s == DEFAULT_READ_TIMEOUT_S
    assert cfg.max_retries == DEFAULT_MAX_RETRIES


def test_env_overrides_honored(clean_env):
    clean_env.setenv("AWS_REGION", "eu-central-1")
    clean_env.setenv("BEDROCK_MODEL_ID", "eu.anthropic.some-other-model")
    clean_env.setenv("LLM_MAX_TOKENS", "1234")
    clean_env.setenv("LLM_READ_TIMEOUT_S", "99")
    clean_env.setenv("LLM_MAX_RETRIES", "2")
    cfg = load_config()
    assert cfg.region == "eu-central-1"
    assert cfg.model_id == "eu.anthropic.some-other-model"
    assert cfg.max_tokens == 1234
    assert cfg.read_timeout_s == 99
    assert cfg.max_retries == 2


def test_shared_client_uses_adaptive_retries_and_timeouts(clean_env):
    clean_env.setenv("AWS_REGION", "eu-west-1")
    clean_env.setenv("LLM_MAX_RETRIES", "4")
    clean_env.setenv("LLM_READ_TIMEOUT_S", "77")
    client = llm_client.get_client()
    assert client is llm_client.get_client()
    assert client.meta.region_name == "eu-west-1"
    assert client.meta.service_model.service_name == "bedrock-runtime"
    cfg = client.meta.config
    assert cfg.retries["mode"] == "adaptive"
    # botocore: max_attempts = retries after the first call → 5 total attempts
    assert cfg.retries["total_max_attempts"] == 5
    assert cfg.read_timeout == 77
    assert cfg.connect_timeout == 10
