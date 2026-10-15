import pytest
from pydantic import BaseModel, ValidationError

from agents import base, llm_client
from agents.base import BaseAgent
from agents.llm_client import LLMConfigError, LLMTruncatedError
from fakes import FakeClient, message
import samples


class Out(BaseModel):
    value: int


class StubAgent(BaseAgent):
    AGENT_NAME = "StubAgent"
    OUTPUT_SCHEMA = Out
    MAX_TOKENS = 1234

    def _build_query(self, profile, goals, context):
        return "q"

    def _build_prompt(self, profile, goals, context, rag_context):
        return "the prompt"


def _run(fake):
    llm_client.set_clients(sync=fake)
    return StubAgent().run(samples.profile(), samples.goals())


def test_forwards_schema_and_max_tokens():
    fake = FakeClient({"value": 3})
    assert _run(fake) == Out(value=3)
    call = fake.calls[0]
    assert call["max_tokens"] == 1234
    assert set(call["output_config"]["format"]["schema"]["properties"]) == {"value"}
    assert call["messages"][0]["content"] == "the prompt"
    assert "JSON only" not in call["system"]


def test_validation_failure_is_retried_then_succeeds():
    fake = FakeClient({"value": "bad"}, {"value": 5})
    assert _run(fake) == Out(value=5)
    assert len(fake.calls) == 2


def test_validation_failure_fails_after_retries():
    fake = FakeClient({"value": "bad"}, {"value": "worse"})
    with pytest.raises(ValidationError):
        _run(fake)
    assert len(fake.calls) == base.MAX_RETRIES


def test_truncation_fails_without_retry():
    fake = FakeClient(message('{"val', stop_reason="max_tokens"), {"value": 1})
    with pytest.raises(LLMTruncatedError):
        _run(fake)
    assert len(fake.calls) == 1


def test_upstream_error_fails_without_retry():
    fake = FakeClient(RuntimeError("overloaded"), {"value": 1})
    with pytest.raises(RuntimeError, match="overloaded"):
        _run(fake)
    assert len(fake.calls) == 1


def test_missing_key_fails_run_naming_variable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        StubAgent().run(samples.profile(), samples.goals())
