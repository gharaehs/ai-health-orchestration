import asyncio
import logging

import pytest
from pydantic import BaseModel, ValidationError

from agents import llm_client
from agents.llm_client import (
    DEFAULT_MODEL,
    LLMConfigError,
    LLMRefusalError,
    LLMTruncatedError,
)
from fakes import FakeClient, message, text_block, thinking_block


class Answer(BaseModel):
    value: int
    label: str


def _use(fake):
    llm_client.set_clients(sync=fake)
    return fake


def test_default_model():
    fake = _use(FakeClient("hi"))
    llm_client.complete("sys", "user", max_tokens=10)
    assert DEFAULT_MODEL == "claude-opus-5-5"
    assert fake.calls[0]["model"] == "claude-opus-5-5"


def test_model_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-other-1")
    fake = _use(FakeClient("hi"))
    llm_client.complete("sys", "user", max_tokens=10)
    assert fake.calls[0]["model"] == "claude-other-1"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_key_names_variable(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    fake = _use(FakeClient("hi"))
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        llm_client.complete("s", "u", max_tokens=10)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        llm_client.generate("s", "u", schema=Answer, max_tokens=10)
    assert fake.calls == []


def test_missing_key_without_fake_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        llm_client.complete("s", "u", max_tokens=10)


def _assert_settings(call, max_tokens):
    assert call["output_config"]["effort"] == "medium"
    for forbidden in ("temperature", "top_p", "top_k", "thinking"):
        assert forbidden not in call
    assert call["max_tokens"] == max_tokens
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "user"}]


def test_request_settings_complete_and_generate():
    fake = _use(FakeClient("hi", {"value": 1, "label": "a"}))
    llm_client.complete("sys", "user", max_tokens=123)
    llm_client.generate("sys", "user", schema=Answer, max_tokens=456)
    _assert_settings(fake.calls[0], 123)
    assert "format" not in fake.calls[0]["output_config"]
    _assert_settings(fake.calls[1], 456)
    fmt = fake.calls[1]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["properties"]) == {"value", "label"}


def test_request_settings_acomplete():
    fake = FakeClient("hi", asynchronous=True)
    llm_client.set_clients(async_=fake)
    text, usage = asyncio.run(llm_client.acomplete("sys", "user", max_tokens=99))
    assert text == "hi"
    assert (usage.input_tokens, usage.output_tokens) == (11, 7)
    _assert_settings(fake.calls[0], 99)


def test_thinking_blocks_are_ignored():
    reply = message([thinking_block("secret reasoning"), text_block('{"value": 2,'),
                     text_block(' "label": "b"}')])
    _use(FakeClient(reply, message([thinking_block(), text_block("plain")])))
    assert llm_client.generate("s", "u", schema=Answer, max_tokens=10) == Answer(value=2, label="b")
    assert llm_client.complete("s", "u", max_tokens=10) == "plain"


def test_generate_returns_parsed_model():
    _use(FakeClient({"value": 7, "label": "seven"}))
    result = llm_client.generate("s", "u", schema=Answer, max_tokens=10)
    assert isinstance(result, Answer)
    assert result.value == 7


def test_generate_validation_error_propagates():
    _use(FakeClient({"value": "not-a-number", "label": "x"}))
    with pytest.raises(ValidationError):
        llm_client.generate("s", "u", schema=Answer, max_tokens=10)


@pytest.mark.parametrize("fn", ["generate", "complete"])
def test_truncation_raises_before_parsing(fn):
    _use(FakeClient(message('{"value": 1, "lab', stop_reason="max_tokens")))
    with pytest.raises(LLMTruncatedError, match="truncated"):
        if fn == "generate":
            llm_client.generate("s", "u", schema=Answer, max_tokens=10)
        else:
            llm_client.complete("s", "u", max_tokens=10)


def test_refusal_raises():
    _use(FakeClient(message("", stop_reason="refusal")))
    with pytest.raises(LLMRefusalError):
        llm_client.generate("s", "u", schema=Answer, max_tokens=10)


def test_check_health_ok():
    fake = _use(FakeClient())
    result = llm_client.check_health()
    assert result == {"status": "ok", "model": "claude-opus-5-5"}
    assert fake.retrieved == ["claude-opus-5-5"]
    assert fake.calls == []


def test_check_health_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = llm_client.check_health()
    assert result["status"] == "error"
    assert "ANTHROPIC_API_KEY" in result["detail"]


def test_check_health_error_never_contains_key(monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    fake = _use(FakeClient())
    fake.retrieve_error = RuntimeError("model not found for key sk-ant-supersecret")
    with caplog.at_level(logging.DEBUG):
        result = llm_client.check_health()
    assert result["status"] == "error"
    assert "sk-ant-supersecret" not in str(result)
    assert "sk-ant-supersecret" not in caplog.text
    assert fake.calls == []


def test_upstream_error_is_logged_without_key(monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")
    _use(FakeClient(RuntimeError("auth failed for sk-ant-supersecret")))
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        llm_client.complete("s", "u", max_tokens=10)
    assert "sk-ant-supersecret" not in caplog.text
    assert "auth failed" in caplog.text
