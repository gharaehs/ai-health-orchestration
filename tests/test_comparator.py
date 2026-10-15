import asyncio

from agents import llm_client
from core import comparator
from fakes import FakeClient, message


def _run(fake, monkeypatch):
    async def no_chunks(query, collections, top_k=3):
        return [{"collection": "nutrition_guidelines", "excerpt": "eat fibre daily",
                 "score": 0.9, "metadata": {}}]
    monkeypatch.setattr(comparator, "retrieve_chunks", no_chunks)
    llm_client.set_clients(async_=fake)
    return asyncio.run(comparator.run_comparison("what to eat?", {"age": 30}, "general"))


def test_comparator_makes_two_calls(monkeypatch):
    fake = FakeClient(message("base answer", input_tokens=20, output_tokens=5),
                      message("rag answer eat fibre", input_tokens=40, output_tokens=8),
                      asynchronous=True)
    result = _run(fake, monkeypatch)
    assert len(fake.calls) == 2
    base, rag = result["base_response"], result["rag_response"]
    assert base["content"] == "base answer" and base["error"] is None
    assert (base["prompt_tokens"], base["completion_tokens"]) == (20, 5)
    assert rag["content"] == "rag answer eat fibre" and rag["error"] is None
    assert (rag["prompt_tokens"], rag["completion_tokens"]) == (40, 8)
    assert set(base) == {"content", "latency_s", "prompt_tokens", "completion_tokens", "error"}
    assert "eat fibre daily" in fake.calls[1]["messages"][0]["content"]
    for call in fake.calls:
        assert call["max_tokens"] == 2048
        assert call["output_config"] == {"effort": "medium"}
    assert "grounding_score" in result["metrics"]


def test_comparator_truncation_sets_error(monkeypatch):
    fake = FakeClient(message("base answer"),
                      message("partial", stop_reason="max_tokens"),
                      asynchronous=True)
    rag = _run(fake, monkeypatch)["rag_response"]
    assert rag["content"] == ""
    assert "truncated" in rag["error"]


def test_comparator_upstream_error_sets_error(monkeypatch):
    fake = FakeClient(RuntimeError("authentication_error"), message("rag"), asynchronous=True)
    base = _run(fake, monkeypatch)["base_response"]
    assert base["content"] == ""
    assert "authentication_error" in base["error"]
