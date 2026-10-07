# Contract: Shared LLM Client (`llm/client.py`)

The single entry point for **all** model calls (user requirement: "one shared client").
Nothing outside `llm/` may import `boto3`'s `bedrock-runtime` client or call a model endpoint.

## Public interface

```python
from llm import get_llm_client, LLMResponse, LLMError, LLMConfigError, LLMTruncatedError, LLMUnavailableError

client = get_llm_client()          # process-wide singleton; settings from env (see data-model.md)

resp: LLMResponse = client.complete(
    user="...",                    # required
    system="...",                  # optional
    max_tokens=None,               # optional, default LLM_MAX_TOKENS
    temperature=None,              # optional
    caller="LabAnalysisAgent",     # for logs
)

resp = await client.acomplete(...) # same args; runs complete() in a worker thread

client.check(deep: bool = False) -> dict
# {"status": "ok"|"error", "provider": "bedrock", "model": <id>, "region": <region>, "detail"?: str}
# deep=False: settings + credential resolution only. deep=True: also a 1-token converse call.
```

## Behavior guarantees

1. Calls Bedrock Runtime `Converse` with `modelId=BEDROCK_MODEL_ID`,
   `system=[{"text": system}]` (if given), `messages=[{"role": "user", "content": [{"text": user}]}]`
   and `inferenceConfig={"maxTokens", "temperature"?}`. It never sends `topP` and never adds an
   assistant prefill.
2. Credentials come only from the default AWS chain (the instance role). The client never reads
   access keys from project config.
3. Retries: botocore adaptive mode, `max_attempts=LLM_MAX_ATTEMPTS`. When retries are exhausted
   or access is denied, it raises `LLMUnavailableError`.
4. `stop_reason == "max_tokens"` raises `LLMTruncatedError`, and the partial text is attached
   to the exception for logging.
5. Each call logs one INFO line: caller, model, latency, input/output tokens and stop reason. It
   never logs prompt or response bodies above DEBUG, since they contain health data.
6. The client is thread-safe. Orchestrator worker threads and the async API share it.

## Call-site mapping

| Call site | Before | After |
|-----------|--------|-------|
| `agents/base.py` `BaseAgent._call_llm(prompt, attempt)` | `httpx` → vLLM | `get_llm_client().complete(system=<existing system prompt>, user=prompt, max_tokens=getattr(self, "MAX_TOKENS_OVERRIDE", None), temperature=0.2, caller=self.AGENT_NAME).text`. The signature is unchanged, so subclasses are untouched |
| `BaseAgent.run()` retry loop | catches `JSONDecodeError`/`ValueError` | also catches `LLMTruncatedError` as a retryable attempt |
| `NutritionAgent.run()` / `GroceryAgent` direct `_call_llm` calls | no retry on truncation | keep their current behavior. A raised `LLMTruncatedError` fails the agent with a clear message |
| `services/api/core/comparator.py` `call_vllm()` | async `httpx` → vLLM | renamed `call_llm()`; uses `acomplete(...)`. The returned dict keys (`content`, `latency_s`, `prompt_tokens`, `completion_tokens`, `error`) are **unchanged** and filled from `LLMResponse` |
| `rag/pipeline.py` `_call_vllm()` | `httpx` → vLLM | `_call_llm()` → `complete(...)`; `vllm_url` constructor arg removed |
| `services/api/routes/orchestrate.py` | patches `agents.base.VLLM_URL` | patch removed |
| `services/api/main.py` `/api/health` | `GET vllm/v1/models` | `client.check(deep=...)` (see `api.md`) |
