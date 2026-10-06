import asyncio
import json
import time

import boto3
import httpx
import pytest
from botocore.awsrequest import AWSResponse
from botocore.stub import Stubber

from agents import llm_client
from agents.llm_client import (
    BedrockBackend,
    LLMAccessDeniedError,
    LLMConfig,
    LLMConfigError,
    LLMThrottledError,
    LLMValidationError,
    VLLMBackend,
)

MODEL_ID = "us.anthropic.claude-opus-test-v1:0"
REGION = "us-east-1"


@pytest.fixture(autouse=True)
def aws_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for var in ("AWS_PROFILE", "AWS_SESSION_TOKEN", "LLM_PROVIDER", "BEDROCK_MODEL_ID",
                "VLLM_URL", "VLLM_MODEL", "LLM_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    llm_client.reset_backend()
    yield
    llm_client.reset_backend()


def bedrock_config(**kw):
    return LLMConfig(provider="bedrock", bedrock_model_id=MODEL_ID, aws_region=REGION, **kw)


def runtime_backend():
    client = boto3.client("bedrock-runtime", region_name=REGION)
    return BedrockBackend(bedrock_config(), client=client), client


def converse_response(text="hello", input_tokens=12, output_tokens=3):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens,
                  "totalTokens": input_tokens + output_tokens},
        "metrics": {"latencyMs": 10},
    }


# ── Bedrock ───────────────────────────────────────────────────────────────────

def test_bedrock_normal_reply():
    backend, client = runtime_backend()
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            converse_response(text='  {"ok": true}  ', input_tokens=100, output_tokens=7),
            expected_params={
                "modelId": MODEL_ID,
                "system": [{"text": "sys"}],
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "inferenceConfig": {"maxTokens": 2048, "temperature": 0.2},
            },
        )
        result = backend.complete("sys", "hi", max_tokens=2048, temperature=0.2)
        stub.assert_no_pending_responses()

    assert (result.text, result.input_tokens, result.output_tokens) == ('{"ok": true}', 100, 7)
    assert result.latency_s >= 0


def test_bedrock_omits_empty_system():
    backend, client = runtime_backend()
    with Stubber(client) as stub:
        stub.add_response(
            "converse", converse_response(),
            expected_params={
                "modelId": MODEL_ID,
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "inferenceConfig": {"maxTokens": 50, "temperature": 0.3},
            },
        )
        assert backend.complete("", "hi", 50, 0.3).text == "hello"


class _Raw:
    def __init__(self, body: bytes):
        self._body = body

    def stream(self, **_):
        yield self._body


def test_bedrock_throttling_is_retried_by_botocore(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    backend = BedrockBackend(bedrock_config(max_attempts=3))
    calls = []

    def fake_send(request, **_):
        calls.append(request.url)
        if len(calls) == 1:
            body = json.dumps({"message": "Too many requests"}).encode()
            return AWSResponse(request.url, 429,
                               {"x-amzn-ErrorType": "ThrottlingException"}, _Raw(body))
        body = json.dumps(converse_response(text="after retry")).encode()
        return AWSResponse(request.url, 200, {}, _Raw(body))

    backend._client.meta.events.register("before-send.bedrock-runtime.Converse", fake_send)
    result = backend.complete("s", "u", 10, 0.2)

    assert len(calls) == 2
    assert result.text == "after retry"


def test_bedrock_throttling_exhausted_maps_error():
    backend, client = runtime_backend()
    with Stubber(client) as stub:
        stub.add_client_error("converse", "ThrottlingException", "slow down", 429)
        with pytest.raises(LLMThrottledError, match="ThrottlingException"):
            backend.complete("s", "u", 10, 0.2)


@pytest.mark.parametrize("code,status,exc", [
    ("AccessDeniedException", 403, LLMAccessDeniedError),
    ("UnrecognizedClientException", 403, LLMAccessDeniedError),
    ("ValidationException", 400, LLMValidationError),
    ("ResourceNotFoundException", 404, LLMValidationError),
])
def test_bedrock_error_mapping(code, status, exc):
    backend, client = runtime_backend()
    with Stubber(client) as stub:
        stub.add_client_error("converse", code, "nope", status)
        with pytest.raises(exc, match=code):
            backend.complete("s", "u", 10, 0.2)


def test_bedrock_requires_model_id():
    with pytest.raises(LLMConfigError):
        BedrockBackend(LLMConfig(provider="bedrock", aws_region=REGION))


def test_bedrock_health_inference_profile():
    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    control = boto3.client("bedrock", region_name=REGION)
    backend = BedrockBackend(bedrock_config(), client=runtime, control_client=control)
    with Stubber(control) as stub:
        stub.add_response(
            "get_inference_profile",
            {"inferenceProfileName": "opus", "inferenceProfileArn": "arn:aws:bedrock:x",
             "inferenceProfileId": MODEL_ID, "status": "ACTIVE", "type": "SYSTEM_DEFINED",
             "models": [{"modelArn": "arn:aws:bedrock:y"}]},
            {"inferenceProfileIdentifier": MODEL_ID},
        )
        assert backend.health() == {"provider": "bedrock", "model": MODEL_ID, "status": "ok"}


def test_bedrock_health_foundation_model_access_denied():
    model = "anthropic.claude-opus-test-v1:0"
    runtime = boto3.client("bedrock-runtime", region_name=REGION)
    control = boto3.client("bedrock", region_name=REGION)
    cfg = LLMConfig(provider="bedrock", bedrock_model_id=model, aws_region=REGION)
    backend = BedrockBackend(cfg, client=runtime, control_client=control)
    with Stubber(control) as stub:
        stub.add_client_error("get_foundation_model", "AccessDeniedException", "denied", 403,
                              expected_params={"modelIdentifier": model})
        info = backend.health()
    assert info["status"] == "error"
    assert "AccessDeniedException" in info["detail"]


# ── vLLM ──────────────────────────────────────────────────────────────────────

def vllm_backend(handler):
    cfg = LLMConfig(provider="vllm", vllm_url="http://llm:8000/", vllm_model="/models/llama")
    return VLLMBackend(cfg, transport=httpx.MockTransport(handler))


def test_vllm_normal_reply():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": " hi there "}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4},
        })

    result = vllm_backend(handler).complete("sys", "user", 1024, 0.3)

    assert seen["url"] == "http://llm:8000/v1/chat/completions"
    assert seen["body"] == {
        "model": "/models/llama",
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "user"}],
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    assert (result.text, result.input_tokens, result.output_tokens) == ("hi there", 20, 4)


@pytest.mark.parametrize("status,exc", [(429, LLMThrottledError), (400, LLMValidationError)])
def test_vllm_error_mapping(status, exc):
    backend = vllm_backend(lambda r: httpx.Response(status, text="bad"))
    with pytest.raises(exc):
        backend.complete("s", "u", 10, 0.2)


def test_vllm_health():
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "/models/llama"}]})

    assert vllm_backend(handler).health() == {
        "provider": "vllm", "model": "/models/llama", "status": "ok"}


# ── Config / module helpers ───────────────────────────────────────────────────

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "VLLM")
    monkeypatch.setenv("VLLM_URL", "http://x:1")
    monkeypatch.setenv("LLM_TIMEOUT", "30")
    cfg = LLMConfig.from_env(vllm_model="m", vllm_url=None)
    assert (cfg.provider, cfg.vllm_url, cfg.vllm_model, cfg.timeout) == ("vllm", "http://x:1", "m", 30.0)
    assert cfg.model == "m"


def test_unknown_provider_and_health_never_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(LLMConfigError):
        llm_client.create_backend()
    info = llm_client.health()
    assert info["status"] == "error" and info["provider"] == "openai"


class _FakeBackend:
    provider = "fake"

    def __init__(self):
        self.calls = []

    def complete(self, system, user, max_tokens, temperature):
        self.calls.append(dict(system=system, user=user, max_tokens=max_tokens,
                               temperature=temperature))
        return llm_client.LLMResult('{"a": 1}', 1, 2, 0.1)


def test_acomplete_uses_default_backend(monkeypatch):
    fake = _FakeBackend()
    monkeypatch.setattr(llm_client, "_default_backend", fake)
    result = asyncio.run(llm_client.acomplete("s", "u", 1024, 0.3))
    assert result.text == '{"a": 1}'
    assert fake.calls == [dict(system="s", user="u", max_tokens=1024, temperature=0.3)]


def test_base_agent_call_llm_delegates(monkeypatch):
    from agents import base

    class Agent(base.BaseAgent):
        MAX_TOKENS_OVERRIDE = 1500

        def _build_query(self, *a): return ""
        def _build_prompt(self, *a): return ""
        def _parse_output(self, raw): return raw

    fake = _FakeBackend()
    monkeypatch.setattr(llm_client, "_default_backend", fake)
    assert Agent()._call_llm("prompt") == '{"a": 1}'
    assert fake.calls == [dict(system=base.SYSTEM_PROMPT, user="prompt",
                               max_tokens=1500, temperature=0.2)]
