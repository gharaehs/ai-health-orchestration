import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "services" / "api"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Fake credentials so botocore never consults a real profile in tests.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.pop("AWS_PROFILE", None)

import llm  # noqa: E402
from llm import health as llm_health  # noqa: E402
from llm.config import LLMConfig  # noqa: E402

TEST_CONFIG = LLMConfig(
    region="us-east-1",
    model_id="test-model-id",
    max_tokens=1000,
    read_timeout_s=30,
    max_retries=3,
)


@pytest.fixture(autouse=True)
def _reset_llm():
    llm.reset()
    llm_health.clear_cache()
    yield
    llm.reset()
    llm_health.clear_cache()


def converse_response(content: list[dict], stop_reason="end_turn", inp=10, out=5) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": inp, "outputTokens": out, "totalTokens": inp + out},
        "metrics": {"latencyMs": 1},
    }


def tool_response(name: str, payload: dict, **kw) -> dict:
    return converse_response(
        [{"toolUse": {"toolUseId": "t1", "name": name, "input": payload}}],
        stop_reason="tool_use",
        **kw,
    )


class FakeBedrock:
    """Minimal stand-in for a bedrock-runtime client: returns queued responses."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responder(kwargs)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def fake_bedrock():
    """Install a FakeBedrock client. Usage: fake = fake_bedrock(responder)."""
    def install(responder):
        fake = FakeBedrock(responder)
        llm.set_client(fake, TEST_CONFIG)
        return fake
    return install
