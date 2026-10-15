# Design

## Context

ADR 0001 and proposal.md (Why) explain the motivation. This section covers only the current code that shapes the approach.

The code makes vLLM `/v1/chat/completions` requests in three separate places. None of them share code:

| Call site | Transport | Params | Notes |
|---|---|---|---|
| `agents/base.py` `BaseAgent._call_llm` | sync `httpx` | `max_tokens` 2048 or `MAX_TOKENS_OVERRIDE`, `temperature` 0.2, timeout 600 s | Used by all four agents. `NutritionAgent` calls it twice (Mon–Thu, Fri–Sun) and `GroceryAgent` calls it directly. The module-level `VLLM_URL` is monkey-patched by `services/api/routes/orchestrate.py`. |
| `rag/pipeline.py` `HealthRAGPipeline._call_vllm` | sync `requests` | `max_tokens` 2048, `temperature` 0.3 | Constructor takes `vllm_url` and `model_name`. `scripts/test_rag.py` passes `--vllm-url`. |
| `services/api/core/comparator.py` `call_vllm` | async `httpx` | `max_tokens` 1024, `temperature` 0.3 | Backs `/api/chat`. Returns `{content, latency_s, prompt_tokens, completion_tokens, error}` and never raises. |

Other relevant facts:

- `/api/health` calls vLLM `GET /v1/models`.
- Orchestration jobs run in a `ThreadPoolExecutor(max_workers=1)`.
- The `api` image copies `agents/` and `rag/` into `/app`, so `agents.*` is importable from all three call sites.
- AGENTS.md already describes `agents/llm_client.py`, `make test`, `requirements-dev.txt` and `docs/aws-runbook.md`. None of them exist yet.
- AGENTS.md also describes Bedrock and `LLM_PROVIDER`, which ADR 0001 rejects.
- `tests/` contains no tests.

## Goals / Non-Goals

**Goals:**
- One module owns all Anthropic access: client construction, configuration, request shape, error translation and the health probe.
- Each call site keeps its current contract with its own callers: agents get a `str`, the pipeline gets a `str`, and the comparator gets its result dict.
- Remove vLLM-specific code, configuration and UI text completely, with no dead flags left behind.

**Non-Goals:**
- Prompt rewrites or prompt-caching optimisation. The only exception is the factual "local, privacy-preserving" wording; see Decisions.
- Streaming responses, tool use, or the native structured-output and JSON features of the API. Agents keep their own JSON extraction.
- Changing Orchestration concurrency, the job store or the polling UX.

## Decisions

### D1. Shared client lives at `agents/llm_client.py`
It exposes three functions:
- `complete(system, user, *, max_tokens, temperature=None) -> str`
- `acomplete(...)` with the same arguments, for async callers
- `check_health() -> dict`

Lazily created singletons of `anthropic.Anthropic` and `anthropic.AsyncAnthropic` are cached at module level.
- **Why here**: AGENTS.md already names this path. `agents/` is copied into the api image, and `rag/` and `services/api/core/` can both import it.
- **Alternatives**: A top-level `llm/` package was rejected because it would mean new Dockerfile `COPY` lines, and the stray `llm/` folder holds only `__pycache__`. A class-based provider interface was rejected because ADR 0001 rules out a provider switch, so the abstraction would have no second implementation.

### D2. Use the official `anthropic` SDK instead of raw HTTP
The SDK handles auth headers, API versioning, typed errors, and retries with backoff for 429, 529 and 5xx. Raw `httpx` would mean rebuilding all of that. The SDK is added to `services/api/requirements.txt` and to `requirements-dev.txt`.

### D3. Configuration is read when the client is first used, not at import
- `ANTHROPIC_API_KEY` is required.
- `ANTHROPIC_MODEL` defaults to a single `DEFAULT_MODEL` constant holding the Claude Opus 5.5 ID.

If the key is missing or blank, the client raises `LLMConfigError("ANTHROPIC_API_KEY is not set")`. This lets modules import in tests and lets `/api/health` report the problem instead of crashing at startup. The key is passed to the SDK explicitly and is never placed in log messages.

### D4. Request shape
Each request is `messages.create(model, system=<system>, messages=[{"role":"user","content":<user>}], max_tokens=<cap>)`. The response text is all `text` content blocks joined together and stripped.
- **System prompts**: The current per-site system prompts are passed through unchanged.
- **Temperature**: It is passed only when the caller gives one. Current values are kept: 0.2 for agents and 0.3 for the pipeline and comparator.
- **Truncation**: If `stop_reason == "max_tokens"`, a warning is logged. Truncated JSON then fails parsing, and the existing agent retry handles it.
- **Timeout**: 600 s, the same as today. SDK `max_retries` stays at the default of 2.

### D5. Errors
`complete`/`acomplete` let `anthropic.APIError` subclasses and `LLMConfigError` propagate.
- **Agents**: The orchestrator already catches agent exceptions and marks them failed, so the existing failure path covers these errors.
- **Comparator**: It keeps its try/except and puts `str(e)` into `error`. SDK error strings do not contain the key.
- **Pipeline**: It logs the error and re-raises, as today.

### D6. Health probe uses `models.retrieve(model)`
This confirms both that the key is valid and that the model is available, without generating tokens, so it costs nothing per poll. It returns `{"status":"ok","model":id}` or `{"status":"error","detail":...}`. In `main.py`, the `services.vllm` entry is replaced by `services.llm`.
- **Alternative**: A one-token `messages.create` was rejected because every health poll would be billed.

### D7. Call-site adaptation
- **`BaseAgent._call_llm`**: delegates to `llm_client.complete`. It keeps its signature, its logging and the `MAX_TOKENS_OVERRIDE` lookup. Remove `VLLM_URL`/`VLLM_MODEL` from `agents/base.py`, and remove the monkey-patch in `routes/orchestrate.py`.
- **`HealthRAGPipeline`**: rename `_call_vllm` to `_call_llm`. Remove the `vllm_url` and `model_name` constructor arguments; the model comes from configuration (D3). `RAGResult` fields that report a model name use the configured ID.
- **`comparator.call_vllm`**: rename to `call_llm` and have it use `acomplete`. Token counts come from `response.usage.input_tokens/output_tokens`, so `acomplete` needs a variant that returns usage. Implement this as `acomplete_with_usage` returning `(text, usage)` instead of widening `acomplete`'s return type.
- **`services/api/core/config.py`**: remove `VLLM_URL` and `VLLM_MODEL`; keep `CHROMA_URL` and `REQUEST_TIMEOUT`.
- **`scripts/test_rag.py`**: remove `--vllm-url`.

### D8. Compose and secrets
- Delete the `llm` service and its GPU `deploy` block, and remove `llm` from `api.depends_on`.
- The `api` service gets `ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}` and `ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-}`. Compose interpolates both from the git-ignored `.env`.
- Add an `.env.example` file listing both variables, with no values.
- Use `:-` rather than `:?` so the stack still starts without a key. `/api/health` then reports the missing key and jobs fail with a clear error, which is the behavior the spec defines. A Compose-level hard stop would hide that path.

### D9. `make test-llm` runs `python3 scripts/test_llm.py` on the host
The script loads `.env` if present (a minimal parser, no new dependency), calls `llm_client.complete` with a short prompt, and prints the reply. On any error it exits with code 1. A `make test` target runs `pytest tests/`. `make download-model` is removed.

### D10. Unit tests in `tests/test_llm_client.py` with the SDK mocked
No network access is needed. The tests cover:
- the default and overridden model
- a missing key raising `LLMConfigError` that names the variable
- text-block joining
- temperature being omitted when `None`
- the `max_tokens` warning
- `check_health` ok and error shapes, including that the error detail never contains the key
- that `BaseAgent._call_llm` forwards `MAX_TOKENS_OVERRIDE`

`requirements-dev.txt` contains `pytest`, `anthropic`, `pydantic` and `httpx`.

### D11. Removals and docs
- **Delete**: `services/llm/`, `scripts/download_model.sh` and `training/`. Remove the `training/evaluation_results.json` line from `.gitignore`. Keep the `models/*` ignore, because the `models/` directory stays harmlessly ignored.
- **README**: rewrite the architecture, services, prerequisites, setup, commands and troubleshooting sections for Anthropic. Replace the vLLM/LoRA/QLoRA sections with a short note pointing to ADR 0001 and git history.
- **AGENTS.md**: drop Bedrock, `LLM_PROVIDER`, `make start-vllm` and the vLLM service line, so that it matches ADR 0001.
- **`docs/aws-runbook.md`** (new, already referenced by AGENTS.md): covers a CPU-only EC2 instance, the security group restricted to the owner's IP on ports 3000 and 8002, setting `.env`, and the Anthropic Console spend limit.

### D12. Wording that is no longer true
- The system prompt in `rag/prompt_builder.py` currently says "local, privacy-preserving". Change it to a neutral "health orchestration system", because Health Profiles are now sent to Anthropic. This is the only prompt edit.
- Update the frontend status label, model badge, System info rows and "~7 min on T4" hints. `StatusDot` labels stay static.
- Update the Llama-specific comments in `rag/`.

## Risks / Trade-offs

- **Health data leaves the host.** Health Profiles and Blood Markers are now sent to Anthropic. → ADR 0001 accepts this. The README and runbook must state it plainly, and the misleading "privacy-preserving" prompt wording is removed (D12).
- **Per-token cost exposed to anyone who can reach the app.** → Restrict the security group to the owner's IP and set a Console spend limit (runbook). The health probe is free (D6).
- **The model may reject `temperature`, or the exact Opus 5.5 ID string may differ from the assumed one.** → Temperature is optional in the client (D4). The model ID lives in one constant and can be overridden with an environment variable (D3). `make test-llm` and `/api/health` surface either problem immediately.
- **Longer and more verbose outputs could hit `max_tokens` caps that were sized for Llama.** → Truncation is logged (D4) and the parse retry already exists. Raise an agent's `MAX_TOKENS_OVERRIDE` only if the logs show truncation during verification.
- **Rate limits (429) or overload (529) during an Orchestration, which makes about 6 calls.** → SDK retries with backoff, then the agent fails visibly (D5).
- **Breaking API change: `/api/health` returns `services.vllm` → `services.llm`.** → The only in-repo consumer is the frontend, which does not read that key. External scripts are not known to exist.

## Migration Plan

1. Merge the change. Then, on the server, create `.env` from `.env.example` with a real `ANTHROPIC_API_KEY`.
2. Set a spend limit in the Anthropic Console, and restrict the security group to the owner's IP.
3. Run `make stop && make start`. Then run `make test-llm`, `curl /api/health` and `python scripts/test_orchestration.py`.
4. Move to a CPU-only instance type and stop or terminate the g4dn.xlarge.
5. Rollback: check out the previous commit on a GPU instance with `models/llama` present, and run `make download-model` if needed. Nothing persistent changes, because ChromaDB data and contracts are untouched.

## Open Questions

- Exact Anthropic model ID string for Claude Opus 5.5. Confirm it with `client.models.list()` during implementation; it only changes the `DEFAULT_MODEL` constant.
- Which CPU instance size the runbook recommends. The api and mcp-rag images both load torch and sentence-transformers, so roughly 8 GB of RAM is expected. Confirm with `docker stats` after the switch.
