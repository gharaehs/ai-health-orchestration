"""
llm/health.py

Minimal Converse probe for /api/health. Proves that region, model ID, model
access and IAM permissions all work. Results are cached for 60 seconds.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from llm.client import LLMProviderError, generate_text, get_config

CACHE_TTL_S = 60

_cache_lock = threading.Lock()
_cached: Optional[tuple[float, dict]] = None


def check_llm_health(force: bool = False) -> dict:
    """Return {"status": "ok"|"error", "model": <id>, ["detail": <msg>]}."""
    global _cached
    with _cache_lock:
        if not force and _cached and time.monotonic() - _cached[0] < CACHE_TTL_S:
            return dict(_cached[1])

    model_id = get_config().model_id
    try:
        generate_text("", "ping", max_tokens=1)
        result = {"status": "ok", "model": model_id}
    except LLMProviderError as e:
        result = {"status": "error", "model": model_id, "detail": str(e)}
    except Exception as e:
        result = {"status": "error", "model": model_id, "detail": f"{type(e).__name__}: {e}"}

    with _cache_lock:
        _cached = (time.monotonic(), result)
    return dict(result)


def clear_cache() -> None:
    global _cached
    with _cache_lock:
        _cached = None
