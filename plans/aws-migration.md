# Plan: AWS migration — CPU EC2 + Claude Opus 5.5 via Bedrock

Source: `proposals/aws-migration.md`, `research/2026-10-06-codebase-overview.md`.

## Problem
The stack runs in a personal AWS account on a GPU host serving Llama 3.1 8B via vLLM. It must move to the
company AWS account, on a CPU-only EC2 instance, with generation provided by Claude Opus 5.5 via Bedrock Converse
(IAM role auth). vLLM URL/model are hard-coded in 4 places; there is no IaC and no test suite.

## Decisions (confirmed)
- **Provider:** switchable — `LLM_PROVIDER=bedrock|vllm` (default `bedrock`). vLLM path retained.
- **Infra:** manual runbook (docs), no Terraform/CDK.
- **Data:** copy `data/chroma` via S3 (no re-ingest; `scripts/ingest.py` untouched).
- **Agent simplification** (Nutrition/Grocery 2,048-token workarounds): out of scope; follow-up.
- **training/ and scripts/download_model.sh:** remove. Consequence: `vllm` mode requires weights placed in
  `models/llama` manually; the `llm` Compose service moves behind an opt-in profile.
- **Model ID / region:** unknown → env placeholders (`BEDROCK_MODEL_ID`, `AWS_REGION`), prerequisite to confirm.

## Approach

### 1. Shared LLM client (`llm_client` package, importable by agents/, rag/, services/api)
- New module (e.g. `agents/llm_client.py`, copied into api image already via `COPY agents/`) exposing
  `complete(system, user, max_tokens, temperature) -> LLMResult(text, input_tokens, output_tokens, latency_s)`.
- Backends: `BedrockBackend` (boto3 `bedrock-runtime.converse`, adaptive retries via botocore config,
  error mapping for throttling/access/validation) and `VLLMBackend` (current OpenAI-compatible HTTP call).
- Config from env: `LLM_PROVIDER`, `BEDROCK_MODEL_ID`, `AWS_REGION`, `VLLM_URL`, `VLLM_MODEL`, `LLM_TIMEOUT`.
- Async wrapper for the comparator (`asyncio.to_thread`).

### 2. Wire call sites
- `agents/base.py:24-29, 113-145` — `_call_llm` delegates to client; keep system prompt, `MAX_TOKENS_OVERRIDE`,
  temperature 0.2, retry/parse loop (`:147-159`) unchanged. Nutrition/Grocery callers unchanged.
- `services/api/core/comparator.py:85-111` — `call_vllm` → provider-neutral `call_llm` using client; keep
  1024 tokens / 0.3, real token usage from Converse.
- `rag/pipeline.py:200-209, 373-398` — `_call_vllm` → client; keep constructor args for vLLM mode.
- `services/api/routes/orchestrate.py:66-67` — delete import-time URL patch.
- `services/api/core/config.py` — env-driven settings.
- `services/api/main.py:28-54` — `/api/health` reports `llm` (provider, model, status) via
  `bedrock:GetFoundationModel`/inference-profile lookup (or `/v1/models` for vLLM). Frontend keys updated.
- `scripts/test_rag.py:238-253` — `--vllm-url` honored only in vLLM mode; add `--provider`.

### 3. Compose & images
- `docker-compose.yml`: `llm` service behind `profiles: ["vllm"]`; drop from api `depends_on`; add
  `LLM_PROVIDER`, `AWS_REGION`, `BEDROCK_MODEL_ID` env to api.
- `services/api/Dockerfile`: CPU-only torch like `services/mcp-rag/Dockerfile:7-12`; drop `--reload`? (keep as-is
  unless trivial — note only).
- `services/api/requirements.txt`: add `boto3`.

### 4. Unit tests (new, minimal)
- `tests/test_llm_client.py` with pytest + botocore `Stubber`: normal reply, throttling→retry, access-denied
  mapping, vLLM backend via `httpx` mock transport. Add `pytest` to a dev requirements file.

### 5. Clean-up & docs
- Remove `training/`, `scripts/download_model.sh`, `make download-model`; fix Makefile help (`:5`, `:11-12`).
- `make test-llm` → `python -m` one-shot call through the client (works for either provider).
- Frontend: `App.tsx:235` status dot and System-info panel (`:731-755`) show provider/model from `/api/health`.
- `README.md`, `AGENTS.md`: Bedrock setup, env vars, vLLM profile, removal of training.
- New `docs/aws-runbook.md`: discovery of current deployment, company prerequisites, provisioning
  (EC2 CPU + Docker/Compose, IAM role scoped to Opus model/inference profile + S3 bucket read, Bedrock model
  access, SG exposing 3000 only or ALB/TLS, EBS for `data/chroma`), Chroma copy via S3, deploy, rollback,
  CloudWatch metrics (invocations, latency, throttles, tokens), cost estimate (~13–15 calls/run, 2/chat).

### 6. Validation & rollout (executed by operator per runbook)
1. Baseline on current system: `scripts/test_orchestration.py --output baseline.json` and `--minimal`.
2. Local test against Bedrock with company creds; unit tests pass; `npm run lint && npm run build`.
3. Provision; copy Chroma; `make start`.
4. `/api/health` ok; orchestration full+minimal exit 0 with 4/4; compare to baseline (schemas, 7-day plan,
   session count); `test_rag.py --health-check` counts = 13,349; `--evaluate` completes; UI smoke test.
5. Cut over; keep personal stack stopped until sign-off; decommission.

## Notes / risks
- Single host, in-memory jobs, Vite dev server, CORS `*` — unchanged (out of scope, noted in runbook).
- Opus 5.5 availability/quota in region; may need cross-region inference profile ID (env-configurable).
- Data classification/logging requirements must be confirmed before go-live.
- Infra/ops steps cannot be verified from this repo; code changes verified via unit tests + local Bedrock run.
