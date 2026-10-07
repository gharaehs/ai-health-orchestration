# Design

## Context

See proposal.md (Why) for the motivation. The relevant current state:

- Four call sites post OpenAI-style `/v1/chat/completions` requests to vLLM. Each one builds its own payload with the model hardcoded as `/models/llama`:
  - `agents/base.py::_call_llm`: sync `httpx`, used by every agent
  - `rag/pipeline.py::_call_vllm`: sync `requests`
  - `services/api/core/comparator.py::call_vllm`: async `httpx`
  - `services/api/main.py`: health check against `/v1/models`
- `routes/orchestrate.py` patches the module global `agents.base.VLLM_URL` at import time.
- Agent outputs are parsed from free text with `_extract_json` (regex, fence stripping, brace slicing) plus a parse-retry loop. Every agent already has a Pydantic output model in `agents/schemas.py`.
- Workarounds for the T4:
  - Nutrition generates the week in two halves to fit the 2048-token output limit.
  - The orchestrator runs agents sequentially.
  - The job executor uses `max_workers=1`.
- The API image is built from the repo root and copies `services/api/`, `agents/` and `rag/`.
- The company account is provisioned by hand. Access to Opus 5.5 on Bedrock is not yet enabled.

## Goals / Non-Goals

**Goals:**
- One LLM client module that owns provider, configuration, retries and response normalization.
- Structured agent outputs that are valid against the schema by construction, not recovered with regex.
- Remove every T4-motivated serialization and splitting workaround.
- A deployment that runs on a CPU-only EC2 instance with no AWS keys anywhere.

**Non-Goals:**
- Infrastructure as code. Provisioning stays manual and is documented in the README.
- Changes to retrieval, ChromaDB, the MCP RAG server, embeddings or corpus content.
- Rewriting prompts beyond what single-call nutrition and tool use require.
- Streaming responses, prompt caching, extended thinking or cost dashboards.
- Supporting more than one provider at runtime. Model ID is configurable, but provider abstraction is not.
- Persistent job storage. The in-memory job store is unchanged.

## Decisions

### D1. Bedrock Converse API via `boto3`
Call `bedrock-runtime.converse()` from `boto3`.
- **Why**: Converse supports system prompts, tool use (`toolConfig`) and token usage in one model-agnostic request shape. `boto3` picks up the EC2 instance role automatically.
- **Alternatives**:
  - `InvokeModel` with Anthropic's native messages body: works, but ties request shapes to the Anthropic body format and duplicates what Converse already normalizes.
  - The `anthropic` SDK's Bedrock client: a nice API, but it adds a second SDK on top of boto3 credentials for no real gain here.

### D2. A new top-level `llm/` package as the single client
Create `llm/client.py`, exposing roughly `generate_text(system, user, max_tokens?)` and `generate_structured(system, user, output_model, max_tokens?)`, both returning content plus usage. Add `COPY llm/ ./llm/` to the API Dockerfile.
- **Why**: `agents/`, `rag/` and `services/api/` all need the client. A neutral package avoids `rag` importing from `agents` or the other way round, and it replaces the `VLLM_URL` monkey-patch with configuration read once from the environment.
- **Alternative**: put it in `agents/base.py`. Rejected because rag and the comparator would then depend on the agents package.

### D3. Configuration via environment variables
`AWS_REGION`, `BEDROCK_MODEL_ID`, `LLM_MAX_TOKENS` (default output cap), `LLM_READ_TIMEOUT_S`, `LLM_MAX_RETRIES`, `MAX_CONCURRENT_JOBS`. These are set in `docker-compose.yml` for the `api` service.
- `BEDROCK_MODEL_ID` defaults to the Opus 5.5 identifier documented in the README. In practice it will likely be a cross-region **inference profile** ID (for example a `us.`, `eu.` or `global.` prefixed ID) rather than a bare model ID. The README tells the operator to copy the exact ID from the Bedrock console after enabling access.
- Agents keep a per-agent `MAX_TOKENS_OVERRIDE`. Nutrition gets a larger cap for the full week.

### D4. Structured output through a forced tool call
`generate_structured` defines one tool, `submit_<output>`, whose `inputSchema.json` is `OutputModel.model_json_schema()`. It forces `toolChoice = {"tool": {"name": ...}}`, reads `toolUse.input` (already a dict) and runs `OutputModel.model_validate(...)`.
- Pydantic emits `$defs`/`$ref` for nested models. The client inlines references before sending (a small helper) so that schema support differences can't cause failures.
- Agents whose `_parse_output` currently post-processes the data (for example Lab Analysis calorie calculation, or Nutrition averages) keep that logic, but run it on the validated dict instead of the regex-extracted one.
- The `run()` retry loop stays, but now retries on `ValidationError` (spec: schema violation → retry → failure).
- Grocery's per-category consolidation and shopping-notes calls use small dedicated schemas in the same way.
- `_extract_json` is deleted once no caller remains. `rag/pipeline.py` keeps text generation and its existing JSON parse. Its output schema lives in the prompt and is outside the agents' contract.
- **Alternative**: prompt for JSON and keep the regex. Rejected because scope B explicitly wants schema enforcement.

### D5. Retries delegated to botocore
Create the client with `botocore.config.Config(retries={"mode": "adaptive", "max_attempts": LLM_MAX_RETRIES}, read_timeout=LLM_READ_TIMEOUT_S, connect_timeout=10)`. Adaptive mode backs off on `ThrottlingException`, service-unavailable and transient errors, and rate-limits the client across threads.
- When retries run out, the client raises an `LLMProviderError` that wraps the botocore error, so failures name the provider.
- **Alternative**: a hand-rolled backoff loop. That's more code and doesn't share a rate limiter across threads.

### D6. Concurrency model stays thread-based
The agents and boto3 are synchronous, so we keep threads rather than converting to async.
- **Jobs**: `ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)`, default 3. Status changes to `running` only when the worker starts, so queued jobs naturally stay `pending`.
- **Within a pipeline**: once Lab finishes, submit Nutrition and Training to a 2-worker executor. Each gets a **shallow copy** of `context`, so concurrent writes can't race. Nutrition's result is merged back before Grocery runs.
- **Grocery**: category consolidation calls are fanned out over a small pool (around 4).
- One boto3 client is shared. boto3 clients are thread-safe, and the `Orchestrator` singleton holds no per-run state apart from `context`, which is per-run.
- **Comparator** (async FastAPI): wraps calls in `asyncio.to_thread` and keeps its parallel base and RAG calls.

### D7. LLM health check
`/api/health` sends a minimal Converse request (`maxTokens=1`, trivial prompt) and caches the result for 60 seconds. This proves that region, model ID, model access and IAM permission all work. Key `llm`, with fields `status`, `model` and, on failure, `detail`.
- **Alternative**: the control-plane `GetFoundationModel` call. Rejected because it doesn't prove invoke permission or model access, which is exactly the failure expected while Opus access isn't enabled yet.

### D8. Usage mapping
Converse `usage.inputTokens` and `usage.outputTokens` map to the existing `prompt_tokens` and `completion_tokens` fields, so the comparator UI is unchanged.

### D9. Removals
Delete:
- `services/llm/`
- `scripts/download_model.sh`
- the `llm` service, GPU reservation and `depends_on: llm` in compose
- the Makefile `download-model` and `test-llm` targets (replaced by a `test-llm` target that hits `/api/health`)
- `training/`
- `--vllm-url` in `scripts/test_rag.py`
- frontend references to vLLM, T4 and LoRA

Also remove `models/` usage. Keep `models/.gitkeep` only if something else still uses `models/`. Otherwise delete it.

## Risks / Trade-offs

- **[Containers can't reach the instance role]** With IMDSv2 and the default hop limit of 1, Docker containers can't get instance-role credentials. → Set the instance metadata **hop limit to 2** during manual setup (documented in the README). The health check will report `llm: error` if this is missed.
- **[Opus access not enabled / wrong region]** → The health check (D7) surfaces it explicitly. The README lists the steps: enable model access, copy the inference profile ID, and use an IAM policy that covers the inference profile **and** the underlying foundation-model ARNs in every region the profile routes to.
- **[Tool-schema incompatibility]** A Pydantic feature such as `$ref`, `anyOf` for `Optional` or enums might be rejected or ignored. → Inline `$ref`s (D4), and validate all four schemas against Bedrock early in the implementation, before rewriting the agents.
- **[Bedrock quotas with new concurrency]** Concurrent jobs, parallel agents and grocery fan-out multiply the request rate. → Adaptive retries (D5), small default pool sizes and configurable `MAX_CONCURRENT_JOBS`. Check the account's Opus quotas during setup.
- **[Cost]** Opus is billed per token, and RAG context makes prompts large. → Explicit output caps per agent. Accept the cost as a trade-off for no longer running a GPU instance full time.
- **[Large single nutrition response]** A full week in one tool call is a big output. → Raise the Nutrition `MAX_TOKENS_OVERRIDE` (for example 16k) and the read timeout. The schema-violation retry covers truncation.
- **[Behavioral drift]** Outputs will differ from Llama's, and there's no golden test set. → Run `scripts/test_orchestration.py` and `scripts/test_rag.py` against Bedrock and check the outputs by hand. Accept that outputs will differ, since that's the point of the change.
- **[Loss of offline/local operation and the Module 1/2 learning artifacts]** → Accepted. The history stays in git.

## Migration Plan

1. **Code** (this change): implement the decisions above. For local development, use `AWS_PROFILE` or env-var credentials mounted into the `api` container. Never commit them.
2. **Company AWS, done by hand:**
   1. In the target region, enable Bedrock model access for Claude Opus 5.5 and note the inference profile ID.
   2. Create an IAM role for EC2 with `bedrock:InvokeModel` (which also covers Converse) on that inference profile and its underlying model ARNs.
   3. Launch a CPU-only Ubuntu instance (enough RAM for torch and sentence-transformers plus Chroma, for example 4 vCPU / 16 GB). Attach the role and set the IMDS hop limit to 2. Use a security group that allows ports 3000 and 8002 (and 8001/8004 only if needed) from trusted IPs. Optionally attach an Elastic IP.
   4. Install Docker and Docker Compose. No NVIDIA toolkit is needed.
3. **Data**: `scp` `data/chroma/` and `data/corpus/` from the personal instance, or re-run `scripts/ingest.py`.
4. **Configure and start**: set `AWS_REGION` and `BEDROCK_MODEL_ID`, run `docker compose up -d`, and confirm `/api/health` reports `llm: ok` and the expected Chroma collection count.
5. **Verify**: run the orchestration and RAG test scripts, plus one end-to-end run from the UI.
6. **Cut over and decommission**: once verified, stop the personal-account instance. Keep its EBS snapshot until you're confident.

**Rollback**: The personal-account deployment stays untouched until step 6. To roll back, keep using it, or redeploy the pre-change git revision there.

## Open Questions

- The exact Bedrock inference profile ID and region for Opus 5.5 in the company account. This becomes known once access is enabled. It's configuration only.
- The final values for default output caps and `MAX_CONCURRENT_JOBS`. These get tuned during verification against real quotas and latency.
