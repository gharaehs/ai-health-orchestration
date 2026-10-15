#!/usr/bin/env python3
"""
scripts/test_llm.py

LLM smoke test (`make test-llm`): one tiny real request to the configured
Anthropic model. Prints the reply and exits 0, or prints the error and exits 1.
Reads ANTHROPIC_API_KEY / ANTHROPIC_MODEL from the environment, falling back to
the repository's .env file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE parser; existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    load_dotenv(ROOT / ".env")
    from agents import llm_client

    try:
        reply = llm_client.complete(
            "You are a terse assistant.",
            "Say hello in one short sentence.",
            max_tokens=256,
        )
    except Exception as e:
        print(f"LLM smoke test failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"Model: {llm_client.get_model()}")
    print(f"Reply: {reply.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
