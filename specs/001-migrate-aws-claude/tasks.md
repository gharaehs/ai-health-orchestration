---

description: "Task list for migrating to the company AWS account with Claude Opus 5.5 via Bedrock"
---

# Tasks: Migrate to Company AWS Account with Claude Opus 5.5

**Input**: Design documents from `specs/001-migrate-aws-claude/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/llm-client.md, contracts/api.md, quickstart.md

**Tests**: Included. The plan defines `tests/test_llm_client.py` and `tests/test_agents_contract.py`
(constitution Principle V and its quality gates).

**Organization**: Tasks are grouped by user story (spec.md) so each story can be implemented and
verified on its own.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1 = full plan on the new platform, US2 = grounded retrieval, US3 = dashboard
  views, US4 = decommission the personal account
- Paths are relative to the repository root

## Key facts for implementers (read before starting)

- The current model calls are `agents/base.py` `BaseAgent._call_llm()` (httpx → vLLM),
  `services/api/core/comparator.py` `call_vllm()`, `rag/pipeline.py`
  `HealthRAGPipeline._call_vllm()` (uses `requests`) and the vLLM probe in
  `services/api/main.py` `/api/health`.
- `services/api/routes/orchestrate.py` line ~66 monkey-patches `agents.base.VLLM_URL`.
- `agents/orchestrator.py` `_run_agent()` catches every agent exception and records it as
  `AgentResult(status=FAILED, error=str(e))`. The job still ends `complete`, so agent failures
  appear in `result.agent_results[].error`. Error messages MUST therefore start with the error
  class name (e.g. `"LLMUnavailableError: ..."`) so they are identifiable there.
- `agents/schemas.py` MUST NOT be modified (FR-003, Principle I).
- Credentials come only from the default boto3 chain (instance IAM role). Never add access keys
  anywhere (FR-009).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Dependencies and skeletons for the shared client and tests

- [ ] T001 Add `boto3>=1.35` to `services/api/requirements.txt`. Install torch from the CPU wheel
  index by adding the line `--extra-index-url https://download.pytorch.org/whl/cpu` above `torch`
  and pinning `torch==2.5.1+cpu`, so no CUDA libraries are pulled (FR-011, research R12)
- [ ] T002 [P] Create the package skeleton `llm/__init__.py`, `llm/errors.py`, `llm/settings.py`
  and `llm/client.py` (empty modules with docstrings) per plan.md "Source Code"
- [ ] T003 [P] Create `tests/__init__.py`, `tests/conftest.py` (adds the repo root to `sys.path`;
  sets env `AWS_REGION=us-east-1`, `BEDROCK_MODEL_ID=test-model`,
  `AWS_ACCESS_KEY_ID=testing`, `AWS_SECRET_ACCESS_KEY=testing` via `monkeypatch` in an
  autouse fixture), and `requirements-dev.txt` containing `pytest`, `boto3`, `pydantic==2.9.2`,
  `httpx==0.27.2`, `mcp`
- [ ] T004 [P] Create `.env.example` with non-secret placeholders only: `AWS_REGION=`,
  `BEDROCK_MODEL_ID=`, and commented optional `LLM_MAX_TOKENS=4096`, `LLM_READ_TIMEOUT=300`,
  `LLM_CONNECT_TIMEOUT=10`, `LLM_MAX_ATTEMPTS=4`. Confirm `.env` is still listed in `.gitignore`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The shared LLM client (contracts/llm-client.md) and the company-account
infrastructure. Every user story depends on this phase.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Shared client (code)

- [ ] T005 [P] Implement error classes in `llm/errors.py`: `LLMError(Exception)`, plus subclasses
  `LLMConfigError`, `LLMTruncatedError` (attribute `partial_text: str`) and `LLMUnavailableError`.
  Each `__str__` MUST return `f"{type(self).__name__}: {message}"`
- [ ] T006 [P] Implement `LLMSettings` (frozen dataclass) and `load_settings()` in `llm/settings.py`.
  Fields and rules come verbatim from data-model.md:
  - `model_id` comes from `BEDROCK_MODEL_ID`. It is **Required**, and a missing value raises
    `LLMConfigError` at startup.
  - `region` comes from `AWS_REGION` and is **Required**.
  - `default_max_tokens` comes from `LLM_MAX_TOKENS`, default 4096, rule "> 0".
  - `connect_timeout_s` comes from `LLM_CONNECT_TIMEOUT`, default 10, rule "> 0".
  - `read_timeout_s` comes from `LLM_READ_TIMEOUT`, default 300, rule "> 0".
  - `max_attempts` comes from `LLM_MAX_ATTEMPTS`, default 4, rule "≥ 1".
  - Any invalid value raises `LLMConfigError`.
- [ ] T007 Implement `LLMResponse` (dataclass: `text`, `stop_reason`, `input_tokens`,
  `output_tokens`, `latency_s`, `model_id`) and `LLMClient` in `llm/client.py`, per
  contracts/llm-client.md:
  - The constructor builds `boto3.client("bedrock-runtime", region_name=settings.region,
    config=botocore.config.Config(retries={"mode": "adaptive", "max_attempts":
    settings.max_attempts}, connect_timeout=..., read_timeout=...))`.
  - `complete(user, system=None, max_tokens=None, temperature=None, caller="")` calls
    `converse(modelId=settings.model_id, messages=[{"role": "user", "content": [{"text": user}]}],
    system=[{"text": system}] if system, inferenceConfig={"maxTokens": max_tokens or
    default_max_tokens, plus "temperature" only if not None})`. It never sends `topP` and never
    adds an assistant prefill.
  - It joins `output.message.content[*].text` and strips the result.
  - `stopReason == "max_tokens"` raises `LLMTruncatedError` with `partial_text` attached.
  - A `botocore.exceptions.ClientError` (Throttling, ServiceUnavailable, AccessDenied,
    Validation, ResourceNotFound), `ReadTimeoutError`, `ConnectTimeoutError` or
    `EndpointConnectionError` raises `LLMUnavailableError` with the AWS error code in the message.
  - `NoCredentialsError` raises `LLMConfigError`.
  - Each call logs one INFO line with caller, model, `latency_s`, input/output tokens and stop
    reason. Prompt and response bodies are logged only at DEBUG.
- [ ] T008 Add `async acomplete(...)` (same args; `await asyncio.to_thread(self.complete, ...)`)
  and `check(deep: bool = False) -> dict` to `llm/client.py`:
  - The result is `{"status", "provider": "bedrock", "model", "region"}`, plus `"detail"` on error.
  - `deep=False` verifies that settings loaded and that
    `boto3.Session().get_credentials()` is not None.
  - `deep=True` also calls `complete(user="ping", max_tokens=1, caller="healthcheck")` and treats
    `LLMTruncatedError` as OK.
- [ ] T009 Implement a thread-safe process singleton `get_llm_client()` (lazy, guarded by
  `threading.Lock`) and `reset_llm_client()` for tests in `llm/client.py`. Export `get_llm_client`,
  `reset_llm_client`, `LLMClient`, `LLMResponse`, `LLMError`, `LLMConfigError`,
  `LLMTruncatedError` and `LLMUnavailableError` from `llm/__init__.py`
- [ ] T010 Write unit tests in `tests/test_llm_client.py` using `botocore.stub.Stubber` on
  `client._bedrock`:
  - (a) A success case maps text, tokens and stop reason, and the request has no `topP`.
  - (b) `stopReason=max_tokens` raises `LLMTruncatedError`.
  - (c) `ThrottlingException` raises `LLMUnavailableError`, and `str(e)` starts with
    `"LLMUnavailableError:"`.
  - (d) A missing `BEDROCK_MODEL_ID` raises `LLMConfigError`.
  - (e) `check(deep=False)` returns `status == "ok"`.
  - (f) The `system` field is omitted when no system prompt is given.

  Run with `pytest tests/test_llm_client.py`.

### Company AWS infrastructure (operational; quickstart.md §1)

- [ ] T011 [P] In the company account, confirm Claude Opus 5.5 access in Amazon Bedrock and record
  the inference profile ID (`aws bedrock list-inference-profiles`, quickstart.md §1.1). Create an
  IAM role and instance profile allowing `bedrock:InvokeModel` on that profile ARN and its
  foundation-model ARNs (research R2). Record both values in the deployment notes, not in git.
- [ ] T012 Launch a CPU-only EC2 instance (`m7i.xlarge` or equivalent, x86_64, at least 4 vCPU,
  16 GB RAM, ≥ 40 GB gp3, Ubuntu 22.04) with the T011 instance profile. Open security-group
  inbound ports 3000, 8001, 8002 and 8004 (public, matching today; port 8000 stays closed).
  Set IMDSv2 with hop limit 2:
  `aws ec2 modify-instance-metadata-options --http-tokens required --http-put-response-hop-limit 2`.
  Install Docker and the Compose plugin, clone the repo and check out `001-migrate-aws-claude`
  (research R10, quickstart.md §1.3–1.4)

**Checkpoint**: `pytest tests/test_llm_client.py` passes, and the company instance can run
`aws sts get-caller-identity` using the role.

---

## Phase 3: User Story 1 - Generate a full health plan on the new platform (Priority: P1) 🎯 MVP

**Goal**: All four agents generate with Claude Opus 5.5 through the shared client and return
schema-valid outputs on the company instance.

**Independent Test**: Run `scripts/test_orchestration.py` (reference profile) 5 times on the
company instance. All 4 `agent_results` are `success`, and each output validates against
`agents/schemas.py` (quickstart V5).

### Tests for User Story 1

- [ ] T013 [P] [US1] Write `tests/test_agents_contract.py`:
  - Add a `FakeLLMClient` whose `complete()` returns `LLMResponse` objects with canned JSON,
    selected by `caller` and prompt content. Monkeypatch `llm.client.get_llm_client` (and the
    name imported into `agents.base`) to return it.
  - Stub each retrieval agent's `_retrieve` to return a fixed context string.
  - Using a reference `HealthProfile`/`UserGoals` fixture, assert that `LabAnalysisAgent`,
    `NutritionAgent` (both halves, Mon–Thu and Fri–Sun, merged into 7 `DayPlan`s),
    `TrainingAgent` and `GroceryAgent` each return an instance of `LabAnalysisOutput`,
    `NutritionOutput`, `TrainingOutput` and `GroceryOutput` respectively.
  - Assert that a fake raising `LLMTruncatedError` on the first call and returning valid JSON
    on the second makes `LabAnalysisAgent.run()` succeed. It must also make `BaseAgent.run()`
    retry.

### Implementation for User Story 1

- [ ] T014 [US1] Refactor `agents/base.py`:
  - Delete the constants `VLLM_URL`, `VLLM_MODEL`, `MAX_TOKENS` and `REQUEST_TIMEOUT`, and the
    `httpx` import. Keep `TEMPERATURE = 0.2` and `MAX_RETRIES = 2`.
  - Reimplement `_call_llm(self, prompt, attempt=1) -> str` as
    `get_llm_client().complete(system=<the existing JSON-only system prompt text, unchanged>,
    user=prompt, max_tokens=getattr(self, "MAX_TOKENS_OVERRIDE", None),
    temperature=TEMPERATURE, caller=self.AGENT_NAME).text`. Keep the signature unchanged so
    `NutritionAgent` and `GroceryAgent` work untouched.
  - In `run()`, add `LLMTruncatedError` to the caught exceptions in the retry loop. The last
    attempt raises `RuntimeError` exactly as today.
  - Do not touch `_extract_json`, `_format_profile` or `_format_goals`.
- [ ] T015 [US1] Check `agents/nutrition_agent.py`, `agents/grocery_agent.py`,
  `agents/training_agent.py` and `agents/lab_analysis_agent.py` for any `MAX_TOKENS_OVERRIDE` or
  vLLM-specific references. Remove any import of `VLLM_*` from `agents.base`. A
  `MAX_TOKENS_OVERRIDE` below 4096 must be raised to at least 4096 so Claude's longer
  half-week meal plans are not truncated (research R5). Make no prompt or schema changes unless
  T013 shows a parse failure.
- [ ] T016 [US1] In `services/api/routes/orchestrate.py`:
  - Remove the `import agents.base as _agent_base` / `_agent_base.VLLM_URL = ...` patch.
  - Remove `VLLM_URL` from the `core.config` import.
  - Update the `ThreadPoolExecutor(max_workers=1)` comment to "one pipeline at a time" (drop the
    T4 reference).
  - Leave the behavior unchanged.
- [ ] T017 [US1] In `services/api/core/config.py`, remove `VLLM_URL` and `VLLM_MODEL` and keep
  `CHROMA_URL` and `REQUEST_TIMEOUT`. Fix any remaining imports of the removed names across
  `services/api/` (`grep -rn VLLM services/api` must return nothing after T018 and T032).
- [ ] T018 [US1] In `services/api/main.py`, replace the vLLM probe in `/api/health` with
  `services["llm"] = get_llm_client().check(deep=deep)`, where `deep: bool = False` is a query
  parameter. Wrap it in try/except so an `LLMConfigError` gives `{"status": "error", "provider":
  "bedrock", "detail": str(e)}`. Keep the `chromadb` block and the
  `{"status": "ok|degraded", "services": {...}}` envelope unchanged (contracts/api.md)
- [ ] T019 [US1] In `services/api/Dockerfile`, add `COPY llm/ ./llm/` next to the existing
  `COPY agents/` and `COPY rag/` lines
- [ ] T020 [US1] Update `docker-compose.yml`:
  - Delete the entire `llm` service.
  - In `api`, remove `VLLM_URL=http://llm:8000` and `llm` from `depends_on`.
  - Add `AWS_REGION=${AWS_REGION}` and `BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID}` to `environment`.
  - Keep `vector-db`, `mcp-rag` and `frontend` unchanged.
  - Confirm `docker compose config` contains no `nvidia` or `gpu` entries (FR-011, Principle III).
- [ ] T021 [US1] Delete `services/llm/Dockerfile` (and the `services/llm/` directory) (FR-013)
- [ ] T022 [US1] Update `Makefile`:
  - Remove the `download-model` target.
  - Replace `test-llm` with `curl -s 'http://localhost:8002/api/health?deep=true' | python3 -m
    json.tool`.
  - Add `test: pytest tests/`.
  - Fix the `help` text (it currently says "Mistral 7B AWQ").
  - Update `.PHONY`.
- [ ] T023 [US1] Run `pytest tests/` locally. All tests in `tests/test_llm_client.py` and
  `tests/test_agents_contract.py` MUST pass before deploying
- [ ] T024 [US1] Deploy on the company instance (from T012): create `.env` from `.env.example`
  with `AWS_REGION` and the T011 `BEDROCK_MODEL_ID`, then run `docker compose up -d --build`.
  Verify quickstart V1 (`/api/health?deep=true` gives `services.llm.status == "ok"`), V2 (no
  secrets, no nvidia) and V11 (with an invalid `BEDROCK_MODEL_ID`, an orchestration's
  `agent_results[].error` starts with `LLMUnavailableError` within a bounded time; restore the
  valid ID afterwards). This needs US2's T029 data copy for grounded output, but V1, V2 and V11
  do not.
- [ ] T025 [US1] After US2's T029 is done, run quickstart V5–V8 on the company instance:
  - Run `python scripts/test_orchestration.py` 5 times (all 4 agents `success`).
  - Review constraint adherence (SC-007).
  - Check that `duration_s` is ≤ 420 s.
  - Run 20 consecutive orchestrations with ≥ 19 complete with 4/4 success.
  - Record the results in `specs/001-migrate-aws-claude/validation.md`.

**Checkpoint**: The full four-agent plan works on Claude Opus 5.5 in the company account.

---

## Phase 4: User Story 2 - Grounded outputs via existing retrieval (Priority: P1)

**Goal**: The same knowledge base is in the company account, agents retrieve through MCP, and a
retrieval failure is visible instead of silently ignored.

**Independent Test**: On the company instance, per-collection counts match the source (13,349
total). The 10 reference queries return identical top-5 results on old and new. Logs show
`Retrieved N chunks via MCP` for the Lab, Nutrition and Training agents (quickstart V3, V4, V9).

### Tests for User Story 2

- [ ] T026 [P] [US2] Add tests to `tests/test_agents_contract.py`. For each of
  `LabAnalysisAgent`, `NutritionAgent` and `TrainingAgent`, monkeypatch the agent's
  `_mcp_retrieve` to raise `ConnectionError`. Assert that `_retrieve()` raises
  `RetrievalUnavailableError` and that `str(e)` starts with `"RetrievalUnavailableError:"`.
  Also assert that an empty chunk list still returns `""` with no exception.

### Implementation for User Story 2

- [ ] T027 [US2] Add `class RetrievalUnavailableError(RuntimeError)` to `agents/base.py`, with a
  `__str__` that returns `f"RetrievalUnavailableError: {message}"`. In the `_retrieve()`
  overrides of `agents/lab_analysis_agent.py` (~line 104), `agents/nutrition_agent.py`
  (~line 120) and `agents/training_agent.py` (~line 90), replace `return ""` inside
  `except Exception as e:` with `raise RetrievalUnavailableError(f"[{self.AGENT_NAME}] MCP
  retrieval failed: {e}") from e`. Keep the `logger.exception` line. Leave the "no chunks
  retrieved" `return ""` path unchanged (research R11).
- [ ] T028 [US2] On the personal-account instance, record the source baseline: per-collection
  document counts from
  `GET http://localhost:8001/api/v2/tenants/default_tenant/databases/default_database/collections`
  (and each collection's count endpoint), plus the top-5 result IDs of 10 reference queries via
  `POST http://localhost:8002/api/search`. Pick 10 queries spanning the four collections
  (lab markers, meal planning, gym programming, recipes). Save to
  `specs/001-migrate-aws-claude/validation.md` under "Source baseline".
- [ ] T029 [US2] Copy the data (research R9, quickstart.md §2). On the personal instance run
  `docker compose stop vector-db`, then `tar czf chroma.tgz data/chroma` and
  `tar czf corpus.tgz data/corpus`. Transfer both to the company instance (scp/rsync, or a
  temporary S3 bucket in the company account, deleted afterwards). Extract them into the repo
  root, then run `docker compose up -d`. Never commit these files (Principle IV).
- [ ] T030 [US2] On the company instance, run quickstart V3 (counts equal the baseline, 13,349
  total) and V4 (identical top-5 for all 10 queries). If either fails, re-run
  `python scripts/ingest.py` from `data/corpus/` and repeat. Record the results in
  `specs/001-migrate-aws-claude/validation.md`.
- [ ] T031 [US2] On the company instance, run quickstart V9: during an orchestration,
  `docker compose logs api | grep -E "Retrieved [0-9]+ chunks via MCP"` shows lines for all
  three retrieval agents. Then run V10: `docker compose stop mcp-rag` and run an orchestration.
  The `agent_results` for Lab, Nutrition and Training show `failed` with errors starting with
  `RetrievalUnavailableError`. Finish with `docker compose start mcp-rag`. Record the results.

**Checkpoint**: Retrieval is identical to the personal deployment, and grounding failures are
visible.

---

## Phase 5: User Story 3 - Dashboard views keep working (Priority: P2)

**Goal**: All 6 views work. Chat & Compare uses Claude Opus 5.5 for both answers. The UI no longer
claims Llama, vLLM or LoRA.

**Independent Test**: Open `http://<company-ip>:3000`, complete the primary action in each of
the 6 views, and submit a Chat & Compare question. Both answers come back, and the labels show
Claude Opus 5.5 / Bedrock (quickstart V12).

### Implementation for User Story 3

- [ ] T032 [P] [US3] In `services/api/core/comparator.py`, rename `call_vllm(messages, label)` to
  `call_llm(messages, label)` and update both call sites in `run_comparison`.
  - Split `messages` into `system` (the `role == "system"` entry) and `user` (the `role == "user"`
    entry).
  - Call `await get_llm_client().acomplete(user=..., system=..., max_tokens=1024,
    temperature=0.3, caller=f"compare:{label}")`.
  - Return the same dict keys as today: `content=resp.text`, `latency_s=resp.latency_s`,
    `prompt_tokens=resp.input_tokens`, `completion_tokens=resp.output_tokens`, `error=None`.
  - On `LLMError`, return `content=""`, zero tokens and `error=str(e)`.
  - Remove the `VLLM_URL` and `VLLM_MODEL` imports. Keep the direct ChromaDB retrieval code
    unchanged.
- [ ] T033 [P] [US3] Update the static labels in `services/frontend/src/App.tsx`
  (contracts/api.md "Frontend"):
  - Line ~235: `StatusDot label="vLLM :8000"` becomes `label="Bedrock"`.
  - Line ~248: the `Llama 3.1 8B` badge becomes `Claude Opus 5.5`.
  - Line ~249: remove the `LoRA: health-v1` badge.
  - Line ~372: "Takes ~7 min on T4." becomes "Typically a few minutes."
  - Line ~429: "Each agent calls vLLM with RAG-augmented prompts" becomes "Each agent calls
    Claude Opus 5.5 (Amazon Bedrock) with RAG-augmented prompts".
  - Line ~743: the Model `InfoRow` value becomes "Claude Opus 5.5 (Amazon Bedrock)".
  - Line ~746: remove the LoRA adapter `InfoRow`.
  - Do not change `services/frontend/src/types.ts`. Run `npm run build` in `services/frontend` to
    confirm it compiles.
- [ ] T034 [US3] Rebuild and redeploy on the company instance
  (`docker compose up -d --build api frontend`). Run quickstart V12: all 6 views (Chat & Compare,
  Orchestrate, Analytics, Knowledge Base, Health Profile, Config) load, and each primary action
  succeeds. Check that the Chat & Compare `base_response.error` and `rag_response.error` are both
  null. Record the results in `specs/001-migrate-aws-claude/validation.md`.

**Checkpoint**: Every dashboard view works on the new deployment.

---

## Phase 6: User Story 4 - Retire the personal-account deployment (Priority: P3)

**Goal**: No resources or cost remain in the personal account.

**Independent Test**: The personal account shows no EC2, EBS, snapshot, EIP or security-group
resources for this system, and Cost Explorer shows no new charges for it (SC-008).

- [ ] T035 [US4] Confirm the gate in `specs/001-migrate-aws-claude/validation.md`: US1 (V1, V2,
  V5–V8, V11), US2 (V3, V4, V9, V10) and US3 (V12) all pass. Also confirm `data/chroma/` and
  `data/corpus/` exist on the company instance (FR-014). Do not continue unless every check
  passes.
- [ ] T036 [US4] In the personal AWS account, terminate the EC2 instance. Then delete any
  detached EBS volumes and snapshots, release the Elastic IP (if any), delete the security
  groups and any S3 transfer bucket, and record the deleted resource IDs in
  `specs/001-migrate-aws-claude/validation.md`. No rollback window is needed (clarification
  2026-10-07).
- [ ] T037 [US4] After 48 h, check the personal account's Cost Explorer and the EC2 console in
  all used regions. Record that there are no remaining resources and no new charges (SC-008) in
  `specs/001-migrate-aws-claude/validation.md`.

**Checkpoint**: The migration is complete and the personal account is clean.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: The remaining model call site, documentation and consistency

- [ ] T038 [P] Migrate `rag/pipeline.py`:
  - Remove the `vllm_url` and `model_name` constructor args and the `requests` call.
  - Rename `_call_vllm(system_prompt, user_message)` to `_call_llm`. It now returns
    `get_llm_client().complete(user=user_message, system=system_prompt,
    max_tokens=self.max_tokens, temperature=self.temperature, caller="rag_pipeline").text`.
  - Update its single call site and the docstrings that mention vLLM.
  - In `scripts/test_rag.py`, remove the `--vllm-url` argument and the `vllm_url=` kwarg.
  - Check that `rag/evaluate.py` still constructs `HealthRAGPipeline` with valid args.
- [ ] T039 [P] Update `README.md` (FR-015):
  - Revise the architecture diagram: remove the vLLM container, add Bedrock outside the
    instance, and show the CPU instance.
  - Replace the Infrastructure & Hardware table and drop the GPU memory table.
  - In Prerequisites and the fresh-server setup, remove the CUDA, driver and model-download steps
    and add the IAM role, IMDS hop limit 2, `.env` (`AWS_REGION`, `BEDROCK_MODEL_ID`) and model
    access steps from quickstart.md §1.
  - Update the Services list to remove the vLLM service and the `services.llm` health entry.
  - Update the Makefile reference.
  - Update "Migrating to a New Server": remove the Llama and LoRA rows and add data-copy steps.
  - Add Troubleshooting entries for `NoCredentialsError` (IMDS hop limit),
    `AccessDeniedException` and `ResourceNotFoundException` (model or profile ID), and
    `ThrottlingException`.
  - Add a note that `training/` and `scripts/download_model.sh` are Module 2 reference only and
    not used at runtime (FR-013).
- [ ] T040 [P] Correct the validation wording in `specs/001-migrate-aws-claude/quickstart.md`
  V10/V11 and `specs/001-migrate-aws-claude/data-model.md` "State transitions". Because the
  orchestrator catches agent exceptions, the job ends `complete` with the affected
  `agent_results[].status == "failed"` and `error` naming the error class. The job does not end
  in a `failed` state.
- [ ] T041 Repository-wide consistency check:
  - `grep -rniE "vllm|VLLM_URL|/models/llama" --include=*.py --include=*.yml --include=*.ts
    --include=*.tsx --include=Makefile . | grep -v -e node_modules -e training/ -e specs/`
    returns nothing.
  - `git status` shows no weights, `data/chroma` or `data/corpus` files, or `.env`.
  - Running `pytest tests/` passes.
- [ ] T042 Final quickstart pass on the company instance: re-run V1, V5 (one run) and V12 after all
  polish changes are deployed (`docker compose up -d --build`), and update
  `specs/001-migrate-aws-claude/validation.md`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies
- **Foundational (Phase 2)**: depends on Setup. T005–T010 (code) and T011–T012 (infra) can run
  in parallel tracks. Blocks all user stories.
- **US1 (Phase 3)**: depends on Phase 2. T025 also needs US2's T029 (the data copy) to produce
  grounded runs.
- **US2 (Phase 4)**: depends on Phase 2 (T012 instance). T026–T027 (code) are independent of US1
  code. T030–T031 need a running stack (T024).
- **US3 (Phase 5)**: depends on Phase 2. T032 needs T009. T034 needs a running stack (T024).
- **US4 (Phase 6)**: depends on US1, US2 and US3 validation all passing (T025, T030, T031, T034).
- **Polish (Phase 7)**: T038–T040 can start after Phase 2. T041 and T042 come last.

### Task-level dependencies

- T005, T006 → T007 → T008 → T009 → T010
- T009 → T013, T014, T018, T032, T038
- T014 → T015 → T023. T016, T017 and T018 → T023. T019, T020 and T021 → T024
- T013 + T026 → T023 (all tests green before deploy)
- T011 → T012 → T024. T028 → T029 → T030
- T024 + T029 → T025, T031. T024 → T034

### Parallel Opportunities

- Setup: T002, T003, T004 together
- Foundational: T005 ∥ T006; T011 (infra) ∥ all code tasks
- Once T009 is done: T013 (US1 tests) ∥ T026 (US2 tests) ∥ T032, T033 (US3) ∥ T038, T039, T040
  (polish), since they touch different files
- US2 T027 (agent retrieval overrides) ∥ US1 T014 (base `_call_llm`). Both edit `agents/base.py`,
  so apply T027's new class after T014 or merge the edits carefully.

## Parallel Example: after Phase 2

```text
Task: "T013 [US1] Agent contract tests in tests/test_agents_contract.py"
Task: "T032 [US3] Comparator → shared client in services/api/core/comparator.py"
Task: "T033 [US3] Frontend labels in services/frontend/src/App.tsx"
Task: "T038 Migrate rag/pipeline.py and scripts/test_rag.py"
Task: "T039 README.md updates"
```

## Implementation Strategy

### MVP (User Story 1 + data copy)

1. Phase 1 → Phase 2 (client and tests green; company instance ready)
2. Phase 3 code tasks T013–T023 → deploy T024
3. Copy the data (US2 T028–T029), then T025
4. **Stop and validate**: four-agent plan on Claude in the company account (SC-001, SC-003)

### Incremental Delivery

1. MVP above
2. US2: retrieval failure visibility + V3, V4, V9, V10 → grounded and equivalent
3. US3: comparator + labels + V12 → all dashboard views verified
4. Polish: `rag/pipeline.py`, README, docs corrections, consistency sweep
5. US4: decommission the personal account (last; irreversible)

## Notes

- `agents/schemas.py` is never edited. If an agent output fails validation with Claude, fix the
  agent's prompt or parsing, not the schema.
- `services/api/routes/search.py` and the comparator's direct ChromaDB access are API-layer code
  and stay unchanged (plan.md Constitution Check).
- `specs/001-migrate-aws-claude/validation.md` is created by the first task that records results
  (T028) and appended to afterwards.
