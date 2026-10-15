# Proposal

## Why

Every LLM call today goes to a self-hosted Llama 3.1 8B on vLLM. That needs a GPU g4dn.xlarge (T4) and gives a much weaker model.

ADR 0001 settles the move to Claude Opus 5.5 through the direct Anthropic API, and `docs/decisions.md` (13 decisions from grilling) fixes the details. This change carries out both, so the stack can run on a small CPU-only instance and produce a better Lab Analysis, Meal Plan, Training Program and Grocery List.

The scope is the model switch for a personal/demo deployment only.

## What Changes

- **Shared client.** Add one LLM client that talks only to the Anthropic Messages API through the official `anthropic` Python SDK (decision 2). It authenticates with `ANTHROPIC_API_KEY`. The model comes from `ANTHROPIC_MODEL`, default `claude-opus-5-5` (decision 3). There is no provider switch.
- **Fixed generation settings** (decision 5, amended after checking the API docs). Opus 5.5 rejects any non-default `temperature` and cannot disable thinking. Requests therefore send no `temperature` and no `thinking` field, and pin `effort` to `medium` so latency and cost stay predictable.
- **Schema-constrained agent output** (decision 4). Each agent call sends a JSON schema generated from a Pydantic model through the API's structured outputs, and the result is still validated with Pydantic. "JSON only" prompting, schema examples in prompts, and regex/brace JSON extraction are removed. Forced tool use is not an option, because Opus 5.5 rejects it.
- **Per-agent `max_tokens`** (decision 6). The Meal Plan gets about 8k and the other agents get smaller caps. A truncated reply is a hard error and is never parsed or retried.
- **Parallel Nutrition and Training.** Nutrition and Training run in parallel within one Orchestration once the Lab Analysis is done (decision 7). Grocery still waits for Nutrition.
- **One Orchestration at a time.** Only one Orchestration runs at a time, kept as a cost guard (decision 8). Later jobs wait as `pending`.
- **Route the remaining call sites through the shared client:**
  - the `/api/chat` base-vs-RAG comparator, kept with two Opus calls per chat (decision 9)
  - the legacy `rag/pipeline.py`, which is moved to the client, not deleted (decision 10)
- **BREAKING:** `/api/health` reports an `llm` entry in place of `vllm`. The check confirms that the key is set and fetches the model's metadata, spending no tokens. `make test-llm` becomes a tiny real call (decision 11).
- **BREAKING:** Remove the vLLM service and GPU reservation from `docker-compose.yml`. Delete `services/llm/`, `scripts/download_model.sh`, `make download-model`, `training/`, the leftover top-level `llm/` folder and all `__pycache__` directories (decision 1).
- **BREAKING:** Replace `VLLM_URL`/`VLLM_MODEL` and the `--vllm-url` flag with `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`.
- **Make ADR 0001 the source of truth in the docs** (decision 1). Rewrite AGENTS.md to drop Bedrock, `LLM_PROVIDER` and `docs/aws-runbook.md`; no runbook is created. Update README and frontend text that names the old stack.
- **Record the health-data consequence in ADR 0001** (decision 12). Blood Markers, medical history and medications are sent to Anthropic unredacted, with no zero-data-retention agreement. This is accepted for a personal/demo deployment.
- **Offline unit tests** (decision 13). `make test` mocks the Anthropic client and never touches the network. Live calls happen only in `make test-llm` and `scripts/test_orchestration.py`.

## Capabilities

### New Capabilities
- `llm-access`: How the system gets completions from its language model. It covers the single hosted provider, configuration and secrets, fixed generation settings, schema-constrained agent output, output limits and truncation, error behavior, the health check, the smoke test, and offline tests.
- `orchestration`: How one Orchestration schedules its agents and how Orchestrations are admitted. It covers the dependency order, the Nutrition/Training parallelism, the Grocery dependency, and the one-at-a-time limit.

### Modified Capabilities
<!-- None: openspec/specs/ is empty, so no capability exists to modify. -->

## Impact

- **Code:**
  - `agents/base.py`, all four agents (`lab_analysis_agent.py`, `nutrition_agent.py`, `training_agent.py`, `grocery_agent.py`), `agents/orchestrator.py` and `agents/schemas.py` (LLM-facing wrapper models only; the `OrchestrationResult` contract is unchanged)
  - `rag/pipeline.py` and `rag/prompt_builder.py`
  - `services/api/main.py`, `core/config.py`, `core/comparator.py` and `routes/orchestrate.py`
  - `scripts/test_rag.py` and `services/frontend/src/App.tsx`
- **New code:**
  - the shared client module
  - `tests/` (the unit tests and a network-blocking fixture)
  - `requirements-dev.txt`, which is referenced but does not exist yet
  - `scripts/test_llm.py` and `.env.example`
- **Dependencies:** Add `anthropic` (a version with structured outputs and `effort` support). Remove the vLLM image and the GPU/NVIDIA runtime.
- **Infrastructure:** Compose no longer needs a GPU host. The server needs `ANTHROPIC_API_KEY` in the git-ignored `.env`. Inference is billed per token, and each Orchestration makes about 6 or more Opus calls.
- **Docs:** README, AGENTS.md, ADR 0001 (new consequence) and `docs/decisions.md` (the decision 5 amendment).
- **Out of scope:**
  - moving to the company AWS account
  - runbooks or production hardening beyond the ADR 0001 safeguards (security group restricted to the owner's IP, Console spend limit)
  - redaction or a zero-data-retention agreement
  - RAG retrieval and embeddings, and the MCP server
  - raising the one-at-a-time Orchestration limit
  - rewriting `REFLECTION.md`, which is a historical record
