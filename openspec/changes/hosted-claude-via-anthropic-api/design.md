# Design

## Context

The motivation is in proposal.md (Why), ADR 0001 and `docs/decisions.md` (referred to below as "decision N"). This section covers only the current code that shapes the approach.

The code calls vLLM `/v1/chat/completions` in three separate places that share no code:

| Call site | Transport | Params | Notes |
|---|---|---|---|
| `agents/base.py` `BaseAgent._call_llm` | sync `httpx` | `max_tokens` 2048 (no agent overrides it), `temperature` 0.2 | All agents use it. `run()` retries `MAX_RETRIES=2` times on JSON/validation errors. |
| `rag/pipeline.py` `HealthRAGPipeline._call_vllm` | sync `requests` | 2048, 0.3 | Constructor takes `vllm_url` and `model_name`. `scripts/test_rag.py --vllm-url`. |
| `services/api/core/comparator.py` `call_vllm` | async `httpx` | 1024, 0.3 | Backs `/api/chat` (base and RAG calls). Never raises; returns `{content, latency_s, prompt_tokens, completion_tokens, error}`. |

How each agent turns the reply into output:
- **Lab Analysis**: `_extract_json` on the reply, then two fix-ups: a guard that clears `analysed_markers` when no Blood Markers were given, and flattening of list fields into strings.
- **Training**: `_extract_json` on the reply.
- **Nutrition**: overrides `run()` and makes two calls, one for Mon–Thu and one for Fri–Sun. It merges the two `{"days": [...]}` replies and computes the averages.
- **Grocery**: overrides `run()` and makes one call per ingredient category, regex-matching a JSON array from each reply; on failure it falls back to the raw ingredients. It makes one more call for shopping tips, also regex-matched, with fallback default tips.

The prompts currently embed "JSON only" instructions and JSON schema examples.

`agents/orchestrator.py` runs Lab, Nutrition, Training and Grocery one after another. A comment says Nutrition and Training are "parallel conceptually … sequential to avoid parallel LLM calls on the T4". Each agent runs inside `_run_agent`, which catches exceptions. Jobs run in `routes/orchestrate.py` on `ThreadPoolExecutor(max_workers=1)`.

What the Anthropic docs say about `claude-opus-5-5` (checked 2026-10-15):
- **Sampling**: any non-default `temperature`, `top_p` or `top_k` returns a 400 error, and SDK v1.0+ does not accept them.
- **Thinking**: adaptive thinking is always on, `thinking: disabled` returns a 400 error, and `effort` (default `medium`) controls thinking depth.
- **Tool choice**: forced tool use (`tool_choice` `any`/`tool`) returns a 400 error.
- **Structured outputs** (`output_config.format` with a JSON schema) are generally available for this model.
- **Thinking blocks** can come before text blocks, so blocks must be selected by `type`.
- **`max_tokens`** must leave room for thinking.

Other facts:
- The `api` image copies `agents/` and `rag/`, so `agents.*` is importable everywhere it is needed.
- AGENTS.md references `agents/llm_client.py`, `make test` and `requirements-dev.txt`, none of which exist yet. It also references Bedrock, `LLM_PROVIDER` and `docs/aws-runbook.md`, which decision 1 drops.
- `tests/` and the top-level `llm/` folder contain only `__pycache__`.

## Goals / Non-Goals

**Goals:**
- One module owns all Anthropic access: configuration, request settings, structured output, truncation and error handling, and the health probe.
- Agents receive validated Pydantic objects directly; text-level JSON extraction is removed.
- Nutrition and Training overlap in time; nothing else about agent ordering or failure isolation changes.
- vLLM code, configuration, docs and UI text are removed completely.

**Non-Goals:**
- Prompt caching, streaming, batch API, fast mode, task budgets.
- Changing the `OrchestrationResult` contract or the polling UX.
- Structured output for the comparator or the legacy pipeline. Neither is an agent: the comparator returns free text, and the pipeline keeps its existing prompt-and-parse flow (decision 10 asks only that it move to the shared client).

## Decisions

### D1. Shared client at `agents/llm_client.py` (decisions 2, 3)
The module exposes:
- `generate(system, user, *, schema: type[BaseModel], max_tokens) -> BaseModel` (sync; used by agents)
- `complete(system, user, *, max_tokens) -> str` (sync; used by the pipeline and the smoke test)
- `acomplete(system, user, *, max_tokens) -> tuple[str, Usage]` (async; used by the comparator)
- `check_health() -> dict`
- exceptions `LLMConfigError` and `LLMTruncatedError`

Lazily created `anthropic.Anthropic`/`AsyncAnthropic` singletons are cached at module level. The sync client is thread-safe, which matters for D7.
- **Why this location:** AGENTS.md already names it, and both `rag/` and `services/api/core/` can import it.
- **Alternatives:** a top-level `llm/` package was rejected because it needs extra Dockerfile `COPY` lines and decision 1 deletes the stale `llm/` folder. A provider interface was rejected because ADR 0001 rules out a provider switch.

### D2. Official `anthropic` SDK (decision 2)
The SDK gives typed errors, a timeout, retries with backoff on 429/529/5xx (default `max_retries=2`), and `transform_schema` for structured outputs. Pin it to a version that supports `output_config.effort` and structured outputs, and add it to `services/api/requirements.txt` and `requirements-dev.txt`. Raw `httpx` was rejected because it would mean rebuilding all of that.

### D3. Configuration is read at first use, not at import (decision 3)
- `ANTHROPIC_API_KEY` is required. If it is missing or blank, the client raises `LLMConfigError("ANTHROPIC_API_KEY is not set")`.
- `ANTHROPIC_MODEL` defaults to `DEFAULT_MODEL = "claude-opus-5-5"`.

Reading at first use keeps imports and tests independent of the environment, and lets `/api/health` report the problem. The key is passed explicitly to the SDK and never logged.

### D4. Request settings (decision 5, amended)
Every request sends:
- `model`
- `system`
- one user message
- `max_tokens`
- `output_config={"effort": "medium"}`, plus `format` when D5 applies

Requests never send `temperature`, `top_p`, `top_k` or `thinking`.

Decision 5 asked for "thinking off, temperature 0.2" and for the docs to be checked. The docs show that `claude-opus-5-5` rejects both settings. The user chose to omit them and pin `effort` to `medium` (the model's default, made explicit so a future default change cannot shift latency or cost silently). `docs/decisions.md` is amended to match.

The reply text is the concatenation of the `text` blocks only, selected by `type`; thinking blocks are ignored. The timeout is 600 s.
- **Alternatives:** `effort=low` gives less reasoning for the Lab Analysis and Meal Plan. A model that accepts `temperature` would contradict decision 3.

### D5. Structured output with `messages.create` + JSON schema, validated afterwards (decision 4)
`generate()` sends `anthropic.transform_schema(Model)` as `output_config.format = {"type": "json_schema", "schema": ...}` through `messages.create`. It checks `stop_reason` first, then validates the reply text with `Model.model_validate_json`.
- **Why not `messages.parse`:** the SDK validates inside `parse`, so a reply cut off at `max_tokens` surfaces as a `ValidationError` before `stop_reason` can be checked, and would be retried instead of failing as truncation (D6). Amended during implementation.
- **Why not tool use:** forced `tool_choice` returns a 400 error on Opus 5.5, and `auto` does not guarantee a call.
- **Validation:** Pydantic validates the reply, so constraints the schema cannot express (such as `min_length`) are still enforced.
- **Retries:** a `pydantic.ValidationError` triggers the existing `MAX_RETRIES=2` loop in `BaseAgent.run()`.
- **`stop_reason`:** `refusal` raises an error. `max_tokens` (and `model_context_window_exceeded`) is handled by D6.

LLM-facing models:

| Agent / call | Schema |
|---|---|
| Lab Analysis | `LabAnalysisOutput` |
| Training | `TrainingOutput` |
| Nutrition, each half-week | new `HalfWeekPlan{days: list[DayPlan]}` |
| Grocery consolidation, per category | new `ConsolidatedItems{items: list[ConsolidatedItem]}` (name, total_quantity, estimated_cost_eur) |
| Grocery shopping tips | new `ShoppingTips{tips: list[str]}` |

The new wrapper models go in `agents/schemas.py`; public contracts are unchanged.

Agent changes:
- **`BaseAgent`:** `_call_llm(prompt, schema, max_tokens)` returns the parsed model.
- **`_extract_json` and every regex or brace extraction are deleted.**
- **`_parse_output(raw)` becomes `_postprocess(model)`:**
  - Lab Analysis keeps the guard that clears `analysed_markers` when no Blood Markers were given.
  - Lab Analysis's list flattening is dropped, because the schema now types those fields as strings.
  - Nutrition still merges the two halves and computes the averages.
- **Prompts** lose their "JSON only" instructions and schema examples (decision 4). Their content otherwise stays the same.
- **Grocery** keeps its per-category fallback to raw items and its default tips.

### D6. Per-agent `max_tokens`; truncation is a hard error (decision 6)
Each agent declares `MAX_TOKENS`, replacing `MAX_TOKENS_OVERRIDE`. Thinking counts against `max_tokens`, so the caps are larger than the old Llama ones.

| Call | `max_tokens` |
|---|---|
| Nutrition (each half-week call) | 8192 |
| Lab Analysis | 4096 |
| Training | 4096 |
| Grocery consolidation, per category | 2048 |
| Grocery shopping tips | 2048 |
| Comparator | 2048 |
| Legacy pipeline | 4096 |
| `make test-llm` | 256 |

When `stop_reason == "max_tokens"`, the client raises `LLMTruncatedError` before any parsing. `BaseAgent.run()` does not retry it, so the agent fails with the error in its `AgentResult`. Grocery's per-category fallback still applies, because Grocery treats a truncated category like any other failed category call. The comparator puts the error into `error`.
- **Assumption:** the Meal Plan keeps its two half-week calls, each capped at 8192. One 7-day call with thinking would sit close to the cap, and keeping the split changes nothing observable. The caps are tuned once in task 9.1 if live runs show truncation.

### D7. Nutrition and Training in parallel (decision 7)
In `Orchestrator.run`:
1. Run Lab Analysis as today.
2. Submit Nutrition and Training to a local `ThreadPoolExecutor(max_workers=2)`.
   - Each receives its own shallow copy of `context`, so neither sees the other's writes.
   - Each still goes through `_run_agent`, so its failures stay isolated.
3. Wait for both, then append their `AgentResult`s in a fixed order: Nutrition, then Training.
4. Run Grocery, or mark it skipped, as today.

Threads are used rather than asyncio because agents and the SDK client are synchronous. Remove the T4 comment.
- **Shared retriever:** both agents use the shared retriever (ChromaDB HTTP client and SentenceTransformer). Concurrent read-only queries to both are expected to be safe; the risk is listed below.

### D8. One Orchestration at a time (decision 8)
Keep `_executor = ThreadPoolExecutor(max_workers=1)` in `routes/orchestrate.py`, which queues later jobs as `pending`. Only the comment changes, from "on T4" to "cost guard: one Orchestration at a time".

### D9. Call-site adaptation (decisions 9, 10)
- **Agents:** remove `VLLM_URL`, `VLLM_MODEL`, `TEMPERATURE` and `httpx` from `agents/base.py`, and remove the `VLLM_URL` monkey-patch in `routes/orchestrate.py`.
- **Comparator:**
  - `call_vllm` becomes `call_llm` and uses `acomplete`. The token counts come from `usage`.
  - The comparator keeps its try/except error shape and still makes two calls per chat.
- **Legacy pipeline:**
  - Rename `_call_vllm` to `_call_llm` and route it through `complete`.
  - Drop the `vllm_url`, `model_name` and `temperature` constructor arguments.
  - Keep `_parse_json`, since the pipeline is not an agent.
  - Keep the file and `scripts/test_rag.py`, and remove `--vllm-url`.
- **`services/api/core/config.py`:** remove `VLLM_URL` and `VLLM_MODEL`.

### D10. Health probe and smoke test (decision 11)
**Health probe:** `check_health()` does the following:
- Check that the key is set; if not, return an error naming it.
- Call `models.retrieve(model)`, which fetches metadata without generating tokens.
- Return `{"status": "ok", "model": id}`, or `{"status": "error", "detail": ...}` with the key stripped.

In `main.py`, `services.vllm` becomes `services.llm`.

**Smoke test:** `make test-llm` runs `python3 scripts/test_llm.py`. The script:
- loads `.env` with a minimal parser (no new dependency)
- calls `complete()` with a one-line prompt and `max_tokens` 256
- prints the reply, or prints the error and exits with code 1

### D11. Compose and secrets
- Delete the `llm` service and its GPU `deploy` block, and remove `llm` from `api.depends_on`.
- The `api` service gets `ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}` and `ANTHROPIC_MODEL=${ANTHROPIC_MODEL:-}` from the git-ignored `.env`. `:-` rather than `:?` keeps the missing-key path observable through `/api/health` and failed jobs, as the spec requires.
- Add an `.env.example` with both keys and no values.

### D12. Offline tests (decision 13)
Add `tests/conftest.py` with an autouse fixture that:
- sets a dummy `ANTHROPIC_API_KEY`
- resets the client singletons
- makes `socket.socket.connect` raise, so any accidental live call fails the test

Tests inject a fake SDK client through a module-level hook in `llm_client`. `make test` runs `pytest tests/`. `requirements-dev.txt` contains `pytest`, `anthropic`, `pydantic`, `httpx` and `fastapi` (for the API route and health tests). Live calls happen only in `make test-llm` and `scripts/test_orchestration.py`.

### D13. Removals and documentation (decisions 1, 12)
- **Delete:** `services/llm/`, `scripts/download_model.sh`, `training/`, the top-level `llm/` folder and every `__pycache__` directory outside `node_modules`. `__pycache__` is already in `.gitignore`. Also drop `make download-model` and the `training/evaluation_results.json` line from `.gitignore`.
- **AGENTS.md:** rewrite to match ADR 0001. Remove Bedrock, `LLM_PROVIDER`, `BEDROCK_MODEL_ID`, `AWS_REGION`, `make start-vllm`, the vLLM service line and `docs/aws-runbook.md`, and list `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`.
- **No runbook is created.**
- **README:** rewrite the architecture, services, prerequisites, setup, commands and troubleshooting sections for the Anthropic API. Add a short "Deployment (personal/demo)" section covering:
  - the security group restricted to the owner's IP
  - the Console spend limit
  - the health-data note

  Replace the vLLM/LoRA/QLoRA sections with a pointer to ADR 0001 and git history.
- **ADR 0001:** add a consequence. Blood Markers, medical history and medications are sent to Anthropic unredacted, with no zero-data-retention agreement; this is accepted for a personal/demo deployment.
- **`docs/decisions.md`:** amend decision 5 with the verified API constraint and the resolution from D4.
- **`rag/prompt_builder.py`:** "local, privacy-preserving" becomes neutral wording, and the Llama comments are updated.
- **Frontend:** update the status label, the model badge ("Claude Opus 5.5"), the System info rows and the T4 timing hints.

## Risks / Trade-offs

- **Health data leaves the host** (decision 12). → Accepted for personal/demo and recorded in ADR 0001 and the README. No redaction and no zero-data retention.
- **Per-token cost exposed to anyone who can reach the app.** → The security group is restricted to the owner's IP, a Console spend limit is set, Orchestrations run one at a time (D8), and the health probe is free (D10).
- **Thinking eats into `max_tokens`, so truncation is now fatal.** → Caps are sized with headroom (D6), and task 9.1 runs a live Orchestration and tunes them once.
- **Structured outputs do not support some JSON Schema features.** → The SDK strips unsupported constraints and validates them with Pydantic afterwards. If a contract model is still rejected, add a dedicated LLM-facing model for that call, as D5 already does for the wrapper models.
- **Parallel agents share the retriever and SentenceTransformer.** → Queries are read-only. If task 9.1 shows contention or errors, put a lock around retrieval only; the LLM calls would still overlap.
- **429/529 errors are more likely with two concurrent calls.** → SDK retries with backoff, then the agent fails visibly.
- **`/api/health` changes from `services.vllm` to `services.llm` (breaking).** → The only in-repo consumer is the frontend, which does not read that key.
- **An SDK version without `effort` or structured outputs.** → Pin a minimum version in both requirements files.

## Migration Plan

1. Merge. On the server, create `.env` from `.env.example` with a real key.
2. Set the Console spend limit and restrict the security group to the owner's IP.
3. Run `make stop && make start`, then `make test-llm`, `curl /api/health` and `python scripts/test_orchestration.py`.
4. Move to a CPU-only instance and stop the g4dn.xlarge.
5. **Rollback:** check out the previous commit on a GPU instance with `models/llama` present. ChromaDB data and the API contract (apart from `/api/health`) are unchanged.

## Open Questions

- CPU instance size for the demo host. The api and mcp-rag images load torch and sentence-transformers, so about 8 GB of RAM is expected. Confirm with `docker stats` after the switch; this only affects the README note.
