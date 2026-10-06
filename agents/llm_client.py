"""
agents/llm_client.py

Provider-neutral LLM client shared by agents/, rag/ and services/api.

    from agents.llm_client import complete
    result = complete(system="...", user="...", max_tokens=2048, temperature=0.2)
    result.text, result.input_tokens, result.output_tokens, result.latency_s

Providers (selected with LLM_PROVIDER):
  bedrock (default) — Amazon Bedrock Converse API, IAM credentials from the
                      default boto3 chain (instance role on EC2).
  vllm              — OpenAI-compatible /v1/chat/completions (local vLLM).

Environment:
  LLM_PROVIDER       bedrock | vllm                (default: bedrock)
  BEDROCK_MODEL_ID   model or inference-profile ID (required for bedrock)
  AWS_REGION         Bedrock region                (default: boto3 chain)
  LLM_MAX_ATTEMPTS   botocore attempts incl. first (default: 5)
  VLLM_URL           vLLM base URL                 (default: http://localhost:8000)
  VLLM_MODEL         vLLM served model name        (default: /models/llama)
  LLM_TIMEOUT        request timeout in seconds    (default: 600)

One-shot test:  python -m agents.llm_client "Say hello!"
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

PROVIDERS = ("bedrock", "vllm")


# ── Results & errors ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMResult:
    text:          str
    input_tokens:  int
    output_tokens: int
    latency_s:     float


class LLMError(RuntimeError):
    """Base class for all LLM client errors."""


class LLMConfigError(LLMError):
    """Missing or invalid configuration (e.g. BEDROCK_MODEL_ID unset)."""


class LLMThrottledError(LLMError):
    """Provider throttled the request and retries were exhausted."""


class LLMAccessDeniedError(LLMError):
    """Credentials missing/invalid or IAM / model access not granted."""


class LLMValidationError(LLMError):
    """Provider rejected the request (bad model ID, params, prompt too long)."""


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMConfig:
    provider:         str = "bedrock"
    bedrock_model_id: Optional[str] = None
    aws_region:       Optional[str] = None
    max_attempts:     int = 5
    vllm_url:         str = "http://localhost:8000"
    vllm_model:       str = "/models/llama"
    timeout:          float = 600.0

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMConfig":
        env = os.environ
        cfg = cls(
            provider=env.get("LLM_PROVIDER", "bedrock").strip().lower(),
            bedrock_model_id=env.get("BEDROCK_MODEL_ID") or None,
            aws_region=env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or None,
            max_attempts=int(env.get("LLM_MAX_ATTEMPTS", "5")),
            vllm_url=env.get("VLLM_URL", "http://localhost:8000"),
            vllm_model=env.get("VLLM_MODEL", "/models/llama"),
            timeout=float(env.get("LLM_TIMEOUT", "600")),
        )
        overrides = {k: v for k, v in overrides.items() if v is not None}
        return replace(cfg, **overrides) if overrides else cfg

    @property
    def model(self) -> Optional[str]:
        return self.bedrock_model_id if self.provider == "bedrock" else self.vllm_model


# ── Bedrock backend ───────────────────────────────────────────────────────────

_THROTTLING_CODES = {
    "ThrottlingException", "TooManyRequestsException",
    "ServiceQuotaExceededException", "ServiceUnavailableException",
}
_ACCESS_CODES = {
    "AccessDeniedException", "UnrecognizedClientException",
    "ExpiredTokenException", "InvalidSignatureException",
}
_VALIDATION_CODES = {"ValidationException", "ResourceNotFoundException"}


def _is_inference_profile(model_id: str) -> bool:
    if model_id.startswith("arn:"):
        return "inference-profile/" in model_id
    prefix = model_id.split(".", 1)[0]
    return prefix in {"us", "eu", "apac", "jp", "au", "ca", "us-gov", "global"}


class BedrockBackend:
    provider = "bedrock"

    def __init__(self, config: LLMConfig, client: Any = None, control_client: Any = None):
        if not config.bedrock_model_id:
            raise LLMConfigError(
                "BEDROCK_MODEL_ID is not set (required when LLM_PROVIDER=bedrock)"
            )
        self.config = config
        self.model_id = config.bedrock_model_id
        self._client = client or self._make_client("bedrock-runtime")
        self._control_client = control_client

    def _make_client(self, service: str) -> Any:
        import boto3
        from botocore.config import Config

        return boto3.client(
            service,
            region_name=self.config.aws_region,
            config=Config(
                retries={"max_attempts": self.config.max_attempts, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=self.config.timeout,
            ),
        )

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> LLMResult:
        from botocore.exceptions import BotoCoreError, ClientError

        request: dict[str, Any] = {
            "modelId":  self.model_id,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            request["system"] = [{"text": system}]

        t0 = time.perf_counter()
        try:
            response = self._client.converse(**request)
        except ClientError as e:
            raise _map_client_error(e) from e
        except BotoCoreError as e:
            raise LLMError(f"Bedrock request failed: {e}") from e
        latency = time.perf_counter() - t0

        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content).strip()
        usage = response.get("usage", {})
        if response.get("stopReason") == "max_tokens":
            logger.warning(f"Bedrock response truncated at maxTokens={max_tokens}")

        return LLMResult(
            text=text,
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            latency_s=round(latency, 2),
        )

    def health(self) -> dict:
        from botocore.exceptions import BotoCoreError, ClientError

        info = {"provider": self.provider, "model": self.model_id}
        try:
            if self._control_client is None:
                self._control_client = self._make_client("bedrock")
            if _is_inference_profile(self.model_id):
                r = self._control_client.get_inference_profile(
                    inferenceProfileIdentifier=self.model_id
                )
                state = r.get("status", "ACTIVE")
            else:
                r = self._control_client.get_foundation_model(
                    modelIdentifier=self.model_id
                )
                state = r.get("modelDetails", {}).get(
                    "modelLifecycle", {}
                ).get("status", "ACTIVE")
            info["status"] = "ok" if state == "ACTIVE" else "error"
            if state != "ACTIVE":
                info["detail"] = f"model status: {state}"
        except ClientError as e:
            info.update(status="error", detail=str(_map_client_error(e)))
        except (BotoCoreError, LLMError) as e:
            info.update(status="error", detail=str(e))
        return info


def _map_client_error(e: Any) -> LLMError:
    err = e.response.get("Error", {})
    code = err.get("Code", "")
    msg = f"Bedrock {code}: {err.get('Message', '')}".strip()
    if code in _THROTTLING_CODES:
        return LLMThrottledError(msg)
    if code in _ACCESS_CODES:
        return LLMAccessDeniedError(msg)
    if code in _VALIDATION_CODES:
        return LLMValidationError(msg)
    return LLMError(msg)


# ── vLLM backend ──────────────────────────────────────────────────────────────

class VLLMBackend:
    provider = "vllm"

    def __init__(self, config: LLMConfig, transport: Optional[httpx.BaseTransport] = None):
        self.config = config
        self.base_url = config.vllm_url.rstrip("/")
        self.model_id = config.vllm_model
        self._transport = transport

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(timeout=timeout, transport=self._transport)

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> LLMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        t0 = time.perf_counter()
        try:
            with self._client(self.config.timeout) as client:
                r = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            msg = f"vLLM HTTP {status}: {e.response.text[:300]}"
            if status == 429:
                raise LLMThrottledError(msg) from e
            if status in (400, 404, 422):
                raise LLMValidationError(msg) from e
            raise LLMError(msg) from e
        except httpx.HTTPError as e:
            raise LLMError(f"vLLM request failed: {e}") from e
        latency = time.perf_counter() - t0

        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=round(latency, 2),
        )

    def health(self) -> dict:
        info = {"provider": self.provider, "model": self.model_id}
        try:
            with self._client(5) as client:
                r = client.get(f"{self.base_url}/v1/models")
                r.raise_for_status()
                models = r.json().get("data") or []
            if models:
                info["model"] = models[0].get("id", self.model_id)
            info["status"] = "ok"
        except Exception as e:
            info.update(status="error", detail=str(e))
        return info


# ── Factory & module-level helpers ────────────────────────────────────────────

def create_backend(config: Optional[LLMConfig] = None):
    config = config or LLMConfig.from_env()
    if config.provider == "bedrock":
        return BedrockBackend(config)
    if config.provider == "vllm":
        return VLLMBackend(config)
    raise LLMConfigError(
        f"Unknown LLM_PROVIDER '{config.provider}' (expected one of {PROVIDERS})"
    )


_default_backend = None
_default_lock = threading.Lock()


def get_backend():
    """Process-wide backend built from the environment (lazy, thread-safe)."""
    global _default_backend
    if _default_backend is None:
        with _default_lock:
            if _default_backend is None:
                _default_backend = create_backend()
    return _default_backend


def reset_backend() -> None:
    global _default_backend
    with _default_lock:
        _default_backend = None


def complete(system: str, user: str, max_tokens: int = 2048,
             temperature: float = 0.2) -> LLMResult:
    return get_backend().complete(system, user, max_tokens, temperature)


async def acomplete(system: str, user: str, max_tokens: int = 2048,
                    temperature: float = 0.2) -> LLMResult:
    return await asyncio.to_thread(complete, system, user, max_tokens, temperature)


def health() -> dict:
    """Never raises; returns {provider, model, status[, detail]}."""
    try:
        return get_backend().health()
    except Exception as e:
        cfg = LLMConfig.from_env()
        return {"provider": cfg.provider, "model": cfg.model,
                "status": "error", "detail": str(e)}


def _main(argv: list[str]) -> int:
    prompt = " ".join(argv) or "Say hello!"
    cfg = LLMConfig.from_env()
    print(f"provider={cfg.provider} model={cfg.model}")
    try:
        r = complete(system="", user=prompt, max_tokens=50, temperature=0.2)
    except LLMError as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(r.text)
    print(f"[{r.latency_s}s, in={r.input_tokens} out={r.output_tokens} tokens]")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
