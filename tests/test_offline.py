import socket

import pytest

from conftest import NetworkBlockedError


def test_socket_connect_is_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlockedError):
            s.connect(("api.anthropic.com", 443))
    finally:
        s.close()


def test_dummy_key_is_set(monkeypatch):
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "test-dummy-key"
