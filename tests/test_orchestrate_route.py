import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.schemas import OrchestrationResult
from routes import orchestrate
from samples import SAMPLE_BODY


class BlockingOrchestrator:
    def __init__(self):
        self.release = {}
        self.started = []

    def run(self, profile, goals):
        n = len(self.started)
        self.started.append(n)
        self.release.setdefault(n, threading.Event()).wait(10)
        return OrchestrationResult()


def _wait_for(predicate, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(orchestrate.router, prefix="/api")
    monkeypatch.setattr(orchestrate, "_jobs", {})
    with TestClient(app) as c:
        yield c


def _status(client, job_id):
    return client.get(f"/api/orchestrate/status/{job_id}").json()


def test_executor_is_single_worker():
    assert orchestrate._executor._max_workers == 1


def test_second_job_waits_until_first_finishes(client, monkeypatch):
    stub = BlockingOrchestrator()
    for i in (0, 1):
        stub.release[i] = threading.Event()
    monkeypatch.setattr(orchestrate, "get_orchestrator", lambda: stub)

    first = client.post("/api/orchestrate", json=SAMPLE_BODY).json()["job_id"]
    assert _wait_for(lambda: _status(client, first)["status"] == "running")
    second = client.post("/api/orchestrate", json=SAMPLE_BODY).json()["job_id"]

    time.sleep(0.2)
    assert _status(client, second)["status"] == "pending"

    stub.release[0].set()
    assert _wait_for(lambda: _status(client, first)["status"] == "complete")
    assert _wait_for(lambda: _status(client, second)["status"] == "running")

    stub.release[1].set()
    assert _wait_for(lambda: _status(client, second)["status"] == "complete")


def test_job_fails_naming_key_when_unset(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(orchestrate, "get_orchestrator", lambda: called.append(1))
    job = client.post("/api/orchestrate", json=SAMPLE_BODY).json()["job_id"]
    assert _wait_for(lambda: _status(client, job)["status"] == "failed")
    assert "ANTHROPIC_API_KEY" in _status(client, job)["error"]
    assert called == []
