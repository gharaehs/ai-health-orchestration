import asyncio

import httpx
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

import main
from core import comparator

from conftest import TEST_CONFIG, converse_response


# ── 4.2 Comparator ────────────────────────────────────────────────────────────

def test_comparator_returns_both_responses_with_usage(fake_bedrock, monkeypatch):
    async def fake_retrieve(query, collections, top_k=3):
        return [{"collection": collections[0], "excerpt": "protein intake guideline",
                 "score": 0.9, "metadata": {}}]

    monkeypatch.setattr(comparator, "retrieve_chunks", fake_retrieve)

    def responder(kw):
        user = kw["messages"][0]["content"][0]["text"]
        if "[CONTEXT:" in user:
            return converse_response([{"text": "rag answer protein"}], inp=300, out=40)
        return converse_response([{"text": "base answer"}], inp=100, out=20)

    fake = fake_bedrock(responder)
    result = asyncio.run(comparator.run_comparison("How much protein?", {"age": 30}))

    base, rag = result["base_response"], result["rag_response"]
    assert (base["content"], base["prompt_tokens"], base["completion_tokens"]) == ("base answer", 100, 20)
    assert (rag["content"], rag["prompt_tokens"], rag["completion_tokens"]) == ("rag answer protein", 300, 40)
    assert base["error"] is None and rag["error"] is None
    assert len(fake.calls) == 2
    assert all(c["inferenceConfig"]["maxTokens"] == comparator.COMPARISON_MAX_TOKENS for c in fake.calls)
    assert result["retrieval"]["chunks_retrieved"] == 1


def test_comparator_reports_provider_error(fake_bedrock, monkeypatch):
    async def fake_retrieve(*a, **k):
        return []

    monkeypatch.setattr(comparator, "retrieve_chunks", fake_retrieve)
    fake_bedrock(lambda kw: ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow"}}, "Converse"))
    result = asyncio.run(comparator.run_comparison("q", {}))
    assert "Bedrock" in result["base_response"]["error"]
    assert result["base_response"]["prompt_tokens"] == 0


# ── 4.3 Health endpoint ───────────────────────────────────────────────────────

@pytest.fixture
def chroma_ok(monkeypatch):
    def handler(request: httpx.Request):
        if request.url.path.endswith("/heartbeat"):
            return httpx.Response(200, json={"nanosecond heartbeat": 1})
        return httpx.Response(200, json=[{"name": "a"}, {"name": "b"}])

    real = httpx.AsyncClient
    monkeypatch.setattr(main.httpx, "AsyncClient",
                        lambda *a, **k: real(transport=httpx.MockTransport(handler)))


def test_health_llm_ok(fake_bedrock, chroma_ok):
    fake = fake_bedrock(lambda kw: converse_response([{"text": "p"}], stop_reason="max_tokens", out=1))
    with TestClient(main.app) as c:
        body = c.get("/api/health").json()
        assert body["services"]["llm"] == {"status": "ok", "model": TEST_CONFIG.model_id}
        assert body["services"]["chromadb"] == {"status": "ok", "collections": 2}
        assert body["status"] == "ok"
        assert "vllm" not in body["services"]
        assert fake.calls[0]["inferenceConfig"]["maxTokens"] == 1

        c.get("/api/health")
        assert len(fake.calls) == 1  # cached for 60s


def test_health_llm_access_denied(fake_bedrock, chroma_ok):
    fake_bedrock(lambda kw: ClientError(
        {"Error": {"Code": "AccessDeniedException",
                   "Message": "You don't have access to the model with the specified model ID."}},
        "Converse",
    ))
    with TestClient(main.app) as c:
        body = c.get("/api/health").json()
    llm_status = body["services"]["llm"]
    assert llm_status["status"] == "error"
    assert llm_status["model"] == TEST_CONFIG.model_id
    assert "AccessDeniedException" in llm_status["detail"]
    assert body["status"] == "degraded"
