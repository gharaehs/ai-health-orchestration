# Research: Migrate to Company AWS Account with Claude Opus 5.5

**Feature**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Date**: 2026-10-07

## Current-state findings (codebase audit)

The self-hosted Llama model is called from **four separate places**, each with its own HTTP code:

| Call site | Used by | Today |
|-----------|---------|-------|
| `agents/base.py` → `BaseAgent._call_llm()` | All 4 agents (Nutrition makes 2 calls, Grocery 1 per category) | Sync `httpx` POST to vLLM `/v1/chat/completions` |
| `services/api/core/comparator.py` → `call_vllm()` | `POST /api/chat` (Chat & Compare, base + RAG) | Async `httpx` POST |
| `rag/pipeline.py` → `HealthRAGPipeline._call_vllm()` | `scripts/test_rag.py`, `rag/evaluate.py` | Sync `httpx` POST |
| `services/api/main.py` → `/api/health` | Operators | Checks vLLM `GET /v1/models` |

Other findings:

- `services/api/routes/orchestrate.py` monkey-patches `agents.base.VLLM_URL` at import time.
  The shared client replaces this.
- The frontend never calls `/api/health`. It only shows **hard-coded** labels such as
  `vLLM :8000`, `Llama 3.1 8B`, `LoRA: health-v1`, and model rows in the Config view
  (`services/frontend/src/App.tsx`). No frontend data contract depends on the LLM.
- The agents' MCP retrieval overrides (`lab_analysis_agent.py`, `nutrition_agent.py`,
  `training_agent.py`) catch every exception and return `""`. Generation then continues
  **without grounding**. This conflicts with the spec edge case "MUST NOT silently continue
  without grounding".
- `orchestrate.py` uses `ThreadPoolExecutor(max_workers=1)`, sized for one T4 GPU. It can stay
  as is. Bedrock removes the GPU limit, but the spec requires no new throughput.
- `services/api/requirements.txt` installs the default `torch` wheel (~2.5 GB of CUDA libraries),
  which is only needed for CPU embeddings.

## Decisions

### R1. Model access API: Bedrock Runtime **Converse** API via `boto3`

- **Decision**: Use `bedrock-runtime` `converse()` from `boto3`, with `system`, `messages`
  and `inferenceConfig` (`maxTokens`, `temperature`).
- **Rationale**: Converse is Bedrock's model-agnostic chat API. It maps one-to-one onto the
  current `system + user → text` call shape and returns `usage` (input/output tokens) and
  `stopReason`. Those values replace the vLLM `usage` fields that the comparator reports.
  `boto3` reads the instance IAM role automatically.
- **Alternatives considered**:
  - `invoke_model` with the Anthropic Messages body: model-specific payloads and more code.
  - `anthropic[bedrock]` SDK (`AnthropicBedrock`): it works, but it adds a second SDK on top of
    boto3 with no benefit for plain text completions.
  - Anthropic API directly: rejected in the 2026-10-07 clarification, because data must stay in
    the company's AWS account.

### R2. Authentication: EC2 instance profile (IAM role), reached from containers

- **Decision**: Attach an IAM role to the company EC2 instance. Containers get credentials
  through the default boto3 chain from the Instance Metadata Service (IMDSv2). Set the
  instance's **IMDSv2 hop limit to 2** so Docker bridge-network containers can reach IMDS.
  No access keys go in `.env` or in the compose file.
- **Rationale**: This satisfies FR-009 and the constitution's no-secrets rule. With the default
  hop limit of 1, IMDSv2 token requests from bridge-networked containers fail, and boto3
  then fails with `NoCredentialsError`.
- **Minimum IAM policy**: `bedrock:InvokeModel` (the permission Converse needs) on the Claude Opus
  5.5 inference-profile ARN and its underlying foundation-model ARNs. The deep health check (R7)
  needs nothing extra.
- **Alternatives considered**: Long-lived access keys in `.env` were rejected (secret handling,
  rotation). Host networking for containers was rejected (it changes the service-name DNS that
  Principle III relies on).

### R3. Model identifier and region: configuration, not code

- **Decision**: Set the model and region with environment variables `BEDROCK_MODEL_ID` and
  `AWS_REGION`, with no hard-coded default for the model ID. The value is the Claude Opus 5.5
  **inference profile ID or ARN** available in the company account. The operator finds it at
  deploy time with `aws bedrock list-inference-profiles` (see quickstart). Startup fails fast
  if `BEDROCK_MODEL_ID` is unset.
- **Rationale**: Recent Claude models on Bedrock are usually invoked through cross-region
  inference profiles, not bare model IDs. The exact ID and the regions where it's available
  depend on the account and can't be confirmed from this repo. A config value avoids
  hard-coding a guess.
- **Alternatives considered**: Hard-coding an ID was rejected because it is unverified and
  differs by region or geography prefix.

### R4. Structured output: keep existing prompt + parse + Pydantic validation

- **Decision**: Keep each agent's existing JSON prompt, `_extract_json`/`_parse_output`, and its
  retry loop. The shared client returns raw text plus `stop_reason`. If `stop_reason ==
  "max_tokens"`, the client raises `LLMTruncatedError`. Agents treat that like a parse failure
  (retry, then fail), so a cut-off plan is never parsed into partial output. Do not use assistant
  prefill.
- **Rationale**: Principle V and FR-003 require the same schemas. The current parsers already
  validate against them, so this is the smallest change. Some recent Claude models reject
  assistant-message prefill, so prefill is not used.
- **Alternatives considered**: Forced tool use with JSON schemas from Pydantic
  (`model_json_schema()`) gives stronger guarantees. It is deferred because it would require
  rewriting every agent's parsing path, which goes beyond "reuse existing agent classes".
  It can be a follow-up feature.

### R5. Token limits and sampling

- **Decision**: The shared client defaults to `max_tokens = 4096` (agents) and keeps per-call
  overrides (`MAX_TOKENS_OVERRIDE`; the comparator keeps 1024). Temperature carries over from
  each call site (agents 0.2, comparator 0.3, RAG pipeline 0.3). Only `temperature` is sent,
  never `top_p` too.
- **Rationale**: Claude tends to write more than Llama 3.1 8B. The current 2048 limit risks
  truncating a half-week meal plan, and the higher limit lowers that risk. The truncation
  detection in R4 is the safety net. Some newer Claude models reject `temperature` and `top_p`
  together.
- **Alternatives considered**: Raising the limit to the model maximum was rejected (cost, and
  failure takes longer to surface).

### R6. Timeouts, retries, throttling (FR-010)

- **Decision**: Configure the boto3 client with `Config(retries={"mode": "adaptive",
  "max_attempts": 4}, connect_timeout=10, read_timeout=300)`. Throttling, service-unavailable
  and timeout errors are retried by botocore. After retries run out, the shared client raises
  `LLMUnavailableError`. The agent run then fails, and the job status becomes `failed` with
  that message through the existing `Job.error` path.
- **Rationale**: Adaptive retry handles Bedrock `ThrottlingException`. A 300 s read timeout
  bounds the worst case per call. With R5 limits, a single generation is far shorter than the
  old roughly 7-minute T4 run.
- **Alternatives considered**: A custom retry loop was rejected because it duplicates botocore.

### R7. `/api/health` LLM probe

- **Decision**: Replace the `vllm` entry in `services` with an `llm` entry
  (`provider`, `model`, `region`, `status`). By default it checks configuration and credential
  resolution only (no model call, no cost). `GET /api/health?deep=true` also runs a 1-token
  Converse call.
- **Rationale**: Nothing in the frontend reads this endpoint, so renaming the key does not
  break the dashboard (FR-007 is about client-facing formats; the top-level
  `{status, services}` envelope is kept). Running a model call on every health poll would cost
  money.
- **Alternatives considered**: Keeping the key name `vllm` was rejected as misleading. Using
  `bedrock:GetInferenceProfile` was rejected because it needs extra IAM permission.

### R8. Shared client location: new top-level `llm/` package

- **Decision**: Create `llm/client.py` with the single `LLMClient` (sync `complete()` and async
  `acomplete()` via `asyncio.to_thread`). Every call site from the audit imports from `llm`. The
  API Dockerfile adds `COPY llm/ ./llm/`.
- **Rationale**: Agents, `rag/`, and `services/api/core/` all need the client. A neutral package
  avoids `rag` importing from `agents`, or the reverse.
- **Alternatives considered**: Putting it in `agents/llm_client.py` was rejected because it
  would make `rag/` and the comparator depend on the agents package.

### R9. Knowledge base and corpus migration

- **Decision**: Copy `data/chroma/` and `data/corpus/` from the personal instance to the company
  instance as tarballs. Use `rsync`/`scp` over SSH, or go through a temporary S3 bucket in the
  company account if the two accounts can't reach each other over SSH. Verify by collection
  document counts (13,349 total) and a 10-query top-5 comparison (SC-005, SC-006). Re-running
  ingestion from the corpus files is the fallback.
- **Rationale**: A byte copy keeps embeddings and IDs identical, which makes SC-006 pass by
  construction. Re-ingesting could change chunk order or IDs.
- **Alternatives considered**: Re-ingesting only was rejected as the primary path because it is
  slower and SC-006 equivalence is not guaranteed.

### R10. Compute sizing in the company account

- **Decision**: Use a CPU-only EC2 instance (no GPU; FR-011), x86_64, at least 4 vCPU and 16 GB
  RAM (e.g. `m7i.xlarge`), with at least 40 GB gp3, Ubuntu 22.04 LTS, Docker and the Compose
  plugin, and an instance profile per R2. Inbound rules stay public on ports 3000, 8002, 8004
  and 8001, matching today (clarification: no access control). Keep 8000 closed, since the
  `llm` service is removed.
- **Rationale**: The remaining load is CPU embeddings (MiniLM), ChromaDB and the web services.
  16 GB covers torch, Chroma and Node.
- **Alternatives considered**: Moving to ECS or Fargate was rejected because the user's
  constraint is to keep Docker Compose as is. Keeping the g4dn GPU instance was rejected under
  FR-011 (cost).

### R11. Grounding failures must be visible

- **Decision**: In the three agent `_retrieve()` overrides, an MCP exception raises
  `RetrievalUnavailableError` instead of returning `""`. The job fails with a clear error. An
  empty but successful retrieval still logs a warning and proceeds, as it does today.
- **Rationale**: This implements the spec edge case. Without it, a broken MCP server in the new
  account would quietly produce ungrounded plans, which SC-006 would not catch.
- **Alternatives considered**: Adding a "grounded: false" flag to outputs was rejected because it
  would change the schemas (FR-003).

### R12. Removed or retained artifacts (FR-013)

- **Decision**: Remove the `llm` service from `docker-compose.yml` and delete `services/llm/`.
  Remove the `download-model` and `test-llm` Makefile targets, replacing `test-llm` with a
  Bedrock smoke test. Keep `training/` and `scripts/download_model.sh` in the repository as
  reference only, with a README note that they are not part of the runtime. Switch the API's
  torch install to the CPU wheel index.
- **Rationale**: Nothing at runtime may depend on Llama or LoRA, and keeping Module 2 work for
  reference is allowed.
