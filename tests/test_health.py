import httpx
import pytest
from fastapi.testclient import TestClient

from agents import llm_client
from fakes import FakeClient
import main


def _chroma_ok(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/heartbeat"):
        return httpx.Response(200, json={"nanosecond heartbeat": 1})
    return httpx.Response(200, json=[{"name": "a"}, {"name": "b"}])


@pytest.fixture
def client(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(main.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(_chroma_ok), **kw))
    return TestClient(main.app)


def test_health_ok(client):
    fake = FakeClient()
    llm_client.set_clients(sync=fake)
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["services"]["llm"] == {"status": "ok", "model": "claude-opus-5-5"}
    assert set(body["services"]) == {"llm", "chromadb"}
    assert fake.calls == []


def test_health_missing_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["services"]["llm"]["status"] == "error"
    assert "ANTHROPIC_API_KEY" in body["services"]["llm"]["detail"]
    assert set(body["services"]) == {"llm", "chromadb"}


def test_health_invalid_model(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-does-not-exist")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    fake = FakeClient()
    fake.retrieve_error = RuntimeError("not_found_error: model claude-does-not-exist")
    llm_client.set_clients(sync=fake)
    resp = client.get("/api/health")
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["services"]["llm"]["status"] == "error"
    assert set(body["services"]) == {"llm", "chromadb"}
    assert "sk-ant-secret-value" not in resp.text
    assert fake.calls == []
