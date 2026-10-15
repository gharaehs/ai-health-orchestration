# Proposal

## Why

Every LLM call today goes to a self-hosted Llama 3.1 8B on vLLM. That requires a GPU g4dn.xlarge (T4) and gives a much weaker model. ADR 0001 settles the move to Claude Opus 5.5 through the direct Anthropic API. This change carries out that decision so the stack can run on a small CPU-only instance and produce a better Lab Analysis, Meal Plan, Training Program and Grocery List.

## What Changes

- Add one shared LLM client that talks only to the Anthropic Messages API (official `anthropic` Python SDK). It authenticates with an `ANTHROPIC_API_KEY` secret.
- Route every current LLM call site through that client:
  - `agents/base.py` `BaseAgent._call_llm`, which all four agents and the Orchestration use
  - `rag/pipeline.py` `HealthRAGPipeline._call_vllm`
  - `services/api/core/comparator.py` `call_vllm`, which backs `/api/chat`
- **BREAKING** Change `/api/health` to report an `llm` service entry (Anthropic reachable, key valid, model available) in place of the `vllm` entry.
- **BREAKING** Remove the vLLM service and the GPU reservation from `docker-compose.yml`. Delete `services/llm/`, `scripts/download_model.sh`, `make download-model` and `training/`. Git history keeps them.
- **BREAKING** Replace the `VLLM_URL` / `VLLM_MODEL` settings and the `--vllm-url` CLI flag with Anthropic settings (`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`). Do not add a provider switch.
- Make `make test-llm` send its smoke prompt to Anthropic instead of `localhost:8000`.
- Update the user-facing text that names the old stack: frontend status labels, model badges, System info panel and "~7 min on T4" hints; README; AGENTS.md; and LLM-specific comments in `rag/`.
- Add pytest unit tests for the shared client, with the Anthropic SDK mocked.
- Document the operational safeguards that ADR 0001 requires: lock the security group to the owner's IP, and set a spend limit in the Anthropic Console.

## Capabilities

### New Capabilities
- `llm-access`: How the system gets text completions from its language model. It covers the single hosted provider, configuration and secrets, request parameters, error behavior, and the health check of the model dependency.

### Modified Capabilities
<!-- None: openspec/specs/ is empty, so no capability exists to modify. -->

## Impact

- **Code**:
  - `agents/base.py`
  - `agents/nutrition_agent.py` and `agents/grocery_agent.py` (indirectly, through `_call_llm`)
  - `rag/pipeline.py` and `rag/prompt_builder.py` (comments and system-prompt wording)
  - `services/api/main.py`, `services/api/core/config.py`, `services/api/core/comparator.py` and `services/api/routes/orchestrate.py`
  - `scripts/test_rag.py`
  - `services/frontend/src/App.tsx`
- **New code**: a shared client module and `tests/` unit tests. A dev requirements file will also be needed, since `requirements-dev.txt` is referenced but does not exist.
- **Dependencies**: add `anthropic` to `services/api/requirements.txt`. Remove the vLLM image and the GPU/NVIDIA runtime requirement.
- **Infrastructure**: Docker Compose no longer needs a GPU host, and the `api` service no longer depends on `llm`. The server needs `ANTHROPIC_API_KEY` in `.env`, which is already git-ignored. Inference is billed per token.
- **Docs**: README, AGENTS.md (it currently mentions Bedrock, `LLM_PROVIDER`, `make start-vllm` and `docs/aws-runbook.md`, which conflicts with ADR 0001), and a new deployment runbook section.
- **Out of scope**:
  - Changing agent prompts, Pydantic contracts, or RAG retrieval and embeddings
  - Changing the MCP server
  - Raising Orchestration concurrency (`ThreadPoolExecutor(max_workers=1)`)
  - Rewriting `REFLECTION.md`, which is a historical record
