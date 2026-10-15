from agents import llm_client
from fakes import FakeClient
from rag import pipeline as pipeline_module
from rag.pipeline import HealthRAGPipeline


class _DummyRetriever:
    def __init__(self, *args, **kwargs):
        pass

    def retrieve(self, **kwargs):
        return []


def test_pipeline_routes_through_shared_client(monkeypatch):
    monkeypatch.setattr(pipeline_module, "HealthRetriever", _DummyRetriever)
    fake = FakeClient('{"plan": "ok"}')
    llm_client.set_clients(sync=fake)
    p = HealthRAGPipeline()
    result = p.query({"age": 30}, "fat_loss", output_type="meal_plan", rag_enabled=False)
    assert result.response_json == {"plan": "ok"}
    call = fake.calls[0]
    assert call["max_tokens"] == 4096
    assert call["output_config"] == {"effort": "medium"}
    assert "temperature" not in call
