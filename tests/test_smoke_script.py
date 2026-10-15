import importlib.util
from pathlib import Path

from agents import llm_client
from fakes import FakeClient

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "test_llm.py"


def _load_script(monkeypatch):
    spec = importlib.util.spec_from_file_location("test_llm_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_dotenv", lambda path: None)
    return module


def test_smoke_test_success_prints_reply(monkeypatch, capsys):
    script = _load_script(monkeypatch)
    fake = FakeClient("Hello there.")
    llm_client.set_clients(sync=fake)
    assert script.main() == 0
    out = capsys.readouterr().out
    assert "Reply: Hello there." in out
    assert "Model: claude-opus-5-5" in out
    assert len(fake.calls) == 1
    assert fake.calls[0]["max_tokens"] == 256


def test_smoke_test_without_key_fails_naming_variable(monkeypatch, capsys):
    script = _load_script(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert script.main() == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
