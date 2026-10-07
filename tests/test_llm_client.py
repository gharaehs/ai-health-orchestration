import json
import time
from typing import Optional

import pytest
from botocore.awsrequest import AWSResponse
from botocore.stub import ANY, Stubber
from pydantic import BaseModel, ValidationError

import llm
from llm import client as llm_client
from llm.client import inline_schema, tool_name_for
from llm.config import LLMConfig

from conftest import converse_response, tool_response

CFG = LLMConfig(region="us-east-1", model_id="test-model", max_tokens=500,
                read_timeout_s=30, max_retries=3)


@pytest.fixture
def stubbed():
    client = llm_client.build_client(CFG)
    llm.set_client(client, CFG)
    with Stubber(client) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()


class Inner(BaseModel):
    label: str
    score: Optional[float] = None


class Outer(BaseModel):
    title: str
    items: list[Inner]
    by_key: dict[str, list[Inner]] = {}


# ── generate_text ─────────────────────────────────────────────────────────────

def test_generate_text_returns_content_and_usage(stubbed):
    stubbed.add_response(
        "converse",
        converse_response([{"text": "Hello "}, {"text": "world"}], inp=12, out=3),
        expected_params={
            "modelId": "test-model",
            "system": [{"text": "sys"}],
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 500},
        },
    )
    resp = llm.generate_text("sys", "hi")
    assert resp.content == "Hello world"
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 3


def test_generate_text_max_tokens_override(stubbed):
    stubbed.add_response(
        "converse", converse_response([{"text": "x"}]),
        expected_params={"modelId": "test-model", "system": ANY, "messages": ANY,
                         "inferenceConfig": {"maxTokens": 42}},
    )
    llm.generate_text("sys", "hi", max_tokens=42)


def test_non_retryable_error_raises_provider_error(stubbed):
    stubbed.add_client_error("converse", service_error_code="AccessDeniedException",
                             http_status_code=403, service_message="no access")
    with pytest.raises(llm.LLMProviderError, match="Bedrock"):
        llm.generate_text("sys", "hi")


# ── Retries (exercise botocore's real adaptive retry handler) ─────────────────
# Stubber bypasses the retry handler, so these tests inject raw HTTP responses
# at the `before-send` stage instead.

def _http(status: int, body: dict, headers: Optional[dict] = None):
    class _Raw:
        def stream(self, **_):
            yield json.dumps(body).encode()
    return AWSResponse("https://bedrock", status, headers or {}, _Raw())


def _install_http(responses: list):
    client = llm_client.build_client(CFG)
    sent = []

    def before_send(request, **_):
        sent.append(request)
        return responses.pop(0)

    client.meta.events.register("before-send.bedrock-runtime.Converse", before_send)
    llm.set_client(client, CFG)
    return sent


THROTTLE = ({"message": "Too many requests"}, {"x-amzn-ErrorType": "ThrottlingException"})


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)


def test_throttled_request_is_retried_then_succeeds(no_sleep):
    sent = _install_http([
        _http(429, THROTTLE[0], THROTTLE[1]),
        _http(200, converse_response([{"text": "recovered"}])),
    ])
    resp = llm.generate_text("sys", "hi")
    assert resp.content == "recovered"
    assert len(sent) == 2


def test_exhausted_retries_raise_provider_error(no_sleep):
    total_attempts = CFG.max_retries + 1
    sent = _install_http([_http(429, THROTTLE[0], THROTTLE[1]) for _ in range(total_attempts)])
    with pytest.raises(llm.LLMProviderError) as exc:
        llm.generate_text("sys", "hi")
    assert "Bedrock" in str(exc.value)
    assert "ThrottlingException" in str(exc.value)
    assert len(sent) == total_attempts


# ── generate_structured ───────────────────────────────────────────────────────

def test_inline_schema_removes_refs():
    schema = inline_schema(Outer.model_json_schema())
    dumped = json.dumps(schema)
    assert "$ref" not in dumped and "$defs" not in dumped
    assert schema["properties"]["items"]["items"]["properties"]["label"]["type"] == "string"
    assert schema["properties"]["by_key"]["additionalProperties"]["items"]["required"] == ["label"]


def test_generate_structured_valid_tool_input(stubbed):
    name = tool_name_for(Outer)
    assert name == "submit_outer"
    payload = {"title": "t", "items": [{"label": "a", "score": 1.5}]}
    stubbed.add_response(
        "converse", tool_response(name, payload, inp=7, out=9),
        expected_params={
            "modelId": "test-model", "system": [{"text": "sys"}], "messages": ANY,
            "inferenceConfig": {"maxTokens": 500},
            "toolConfig": {
                "tools": [{"toolSpec": {"name": name, "description": ANY,
                                        "inputSchema": {"json": inline_schema(Outer.model_json_schema())}}}],
                "toolChoice": {"tool": {"name": name}},
            },
        },
    )
    resp = llm.generate_structured("sys", "make it", Outer)
    assert isinstance(resp.content, Outer)
    assert resp.content.items[0].label == "a"
    assert (resp.prompt_tokens, resp.completion_tokens) == (7, 9)


def test_generate_structured_schema_violation_raises(stubbed):
    stubbed.add_response("converse", tool_response("submit_outer", {"title": "t", "items": [{"score": "x"}]}))
    with pytest.raises(ValidationError):
        llm.generate_structured("sys", "make it", Outer)


def test_generate_structured_without_tool_call_raises(stubbed):
    stubbed.add_response("converse", converse_response([{"text": "partial"}], stop_reason="max_tokens"))
    with pytest.raises(llm.StructuredOutputError, match="max_tokens"):
        llm.generate_structured("sys", "make it", Outer)
