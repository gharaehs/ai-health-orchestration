import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.schemas import OrchestrationResult
from routes import orchestrate

TIMEOUT = 5


class BlockingOrchestrator:
    """Each run() blocks until released, so tests control job completion."""

    def __init__(self):
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.running = 0

    def run(self, profile, goals):
        with self.lock:
            self.running += 1
        assert self.release.wait(TIMEOUT)
        return OrchestrationResult()


@pytest.fixture
def client(monkeypatch):
    fake = BlockingOrchestrator()
    executor = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(orchestrate, "get_orchestrator", lambda: fake)
    monkeypatch.setattr(orchestrate, "_executor", executor)
    monkeypatch.setattr(orchestrate, "_jobs", {})
    app = FastAPI()
    app.include_router(orchestrate.router, prefix="/api")
    with TestClient(app) as c:
        yield c, fake
    fake.release.set()
    executor.shutdown(wait=True)


def submit(c) -> str:
    body = c.get("/api/orchestrate/sample-request").json()
    r = c.post("/api/orchestrate", json=body)
    assert r.status_code == 200
    assert r.json()["status"] == "pending"
    return r.json()["job_id"]


def status(c, job_id) -> str:
    return c.get(f"/api/orchestrate/status/{job_id}").json()["status"]


def wait_for(predicate, timeout=TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_jobs_run_concurrently_and_excess_queue(client):
    c, fake = client
    a, b, extra = submit(c), submit(c), submit(c)

    assert wait_for(lambda: status(c, a) == "running" and status(c, b) == "running")
    assert wait_for(lambda: fake.running == 2)
    time.sleep(0.1)
    assert status(c, extra) == "pending"

    fake.release.set()
    assert wait_for(lambda: all(status(c, j) == "complete" for j in (a, b, extra)))
    assert c.get(f"/api/orchestrate/result/{a}").status_code == 200


def test_unknown_job_returns_404(client):
    c, _ = client
    assert c.get("/api/orchestrate/status/nope").status_code == 404
    assert c.get("/api/orchestrate/result/nope").status_code == 404


def test_max_concurrent_jobs_from_env(monkeypatch):
    import core.config as config

    monkeypatch.delenv("MAX_CONCURRENT_JOBS", raising=False)
    assert importlib.reload(config).MAX_CONCURRENT_JOBS == 3
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "5")
    assert importlib.reload(config).MAX_CONCURRENT_JOBS == 5
    monkeypatch.delenv("MAX_CONCURRENT_JOBS")
    importlib.reload(config)
    assert orchestrate._executor._max_workers == orchestrate.MAX_CONCURRENT_JOBS
