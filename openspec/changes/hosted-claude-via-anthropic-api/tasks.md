# Tasks

## 1. Dependencies and test scaffolding

- [ ] 1.1 Add `anthropic` to `services/api/requirements.txt`. Verify with `docker compose build api`, which should succeed.
- [ ] 1.2 Create `requirements-dev.txt` with `pytest`, `anthropic`, `pydantic` and `httpx`, and add a `make test` target that runs `pytest tests/`. Verify that `pip install -r requirements-dev.txt && make test` runs; it may collect 0 tests at this stage.
- [ ] 1.3 Add `.env.example` with empty `ANTHROPIC_API_KEY=` and `ANTHROPIC_MODEL=` entries. Verify that `git check-ignore .env` still matches and `.env.example` is tracked.

## 2. Shared LLM client

- [ ] 2.1 Create `agents/llm_client.py`, implementing D3 (lazy config, `DEFAULT_MODEL`, `LLMConfigError`) and D4 (`complete`, `acomplete`, `acomplete_with_usage`, text-block join, optional temperature, `max_tokens` truncation warning, 600 s timeout). Verify with `python -c "import agents.llm_client"` with no key set, which should succeed without raising.
- [ ] 2.2 Implement `check_health()` with `models.retrieve` (D6). It returns `{"status":"ok","model":id}`, or an error dict that names `ANTHROPIC_API_KEY` when the key is missing. Verify with the tests in 2.3.
- [ ] 2.3 Write `tests/test_llm_client.py` with a mocked SDK covering every D10 case: default and overridden model, missing-key error text, text-block joining, temperature omitted when `None`, truncation warning, health ok/error, and the key never appearing in logs or health output. Verify that `make test` passes.
- [ ] 2.4 Confirm the exact Claude Opus 5.5 model ID with `client.models.list()` using a real key, and set `DEFAULT_MODEL`. Verify that `python -c "from agents import llm_client; print(llm_client.check_health())"` prints `status: ok`.

## 3. Agents and Orchestration

- [ ] 3.1 Rewrite `BaseAgent._call_llm` in `agents/base.py` to delegate to `llm_client.complete`. Keep the current system prompt, temperature 0.2, the `MAX_TOKENS_OVERRIDE` lookup and the logging, and remove `VLLM_URL`, `VLLM_MODEL` and the `httpx` import. Verify with a new test in `tests/test_base_agent.py` that a stub agent's `_call_llm` passes its `MAX_TOKENS_OVERRIDE` and system prompt to the mocked client, and that `make test` passes.
- [ ] 3.2 Remove the `VLLM_URL` monkey-patch and import from `services/api/routes/orchestrate.py`, and update the "on T4" comment. Verify that `grep -rn VLLM agents services/api` returns nothing.
- [ ] 3.3 Add a test in which `run()` raises an error naming `ANTHROPIC_API_KEY` when the key is unset. Verify that `make test` passes.

## 4. RAG pipeline and CLI

- [ ] 4.1 In `rag/pipeline.py`, rename `_call_vllm` to `_call_llm` and route it through `llm_client.complete` (temperature 0.3). Drop the `vllm_url` and `model_name` constructor arguments and the `requests` import, and report the configured model ID wherever the model name was logged or returned. Verify that `python scripts/test_rag.py --help` runs.
- [ ] 4.2 Remove `--vllm-url` from `scripts/test_rag.py`. Verify that `python scripts/test_rag.py --help` no longer lists it, and that `python scripts/test_rag.py --type meal_plan` returns parsed JSON against the running stack.
- [ ] 4.3 In `rag/prompt_builder.py`, replace "local, privacy-preserving" with neutral wording and update the Llama/vLLM comments (D12). Verify that `grep -rniE "vllm|llama" rag/` returns nothing.

## 5. API service

- [ ] 5.1 Remove `VLLM_URL` and `VLLM_MODEL` from `services/api/core/config.py`. Rename `comparator.call_vllm` to `call_llm`, built on `acomplete_with_usage`, keeping the result dict shape and taking token counts from `usage`. Verify that `/api/chat` returns `base_response` and `rag_response` with non-empty content and an `error` of `null`.
- [ ] 5.2 Replace the `vllm` check in `/api/health` with `services.llm` from `check_health()`. Verify that `curl localhost:8002/api/health` shows `services.llm.status == "ok"` and the model ID, and has no `vllm` key. Then, with `ANTHROPIC_API_KEY` unset, verify that `llm.status` is `error` and the overall status is `degraded`.
- [ ] 5.3 Verify the missing-key behavior end to end: with no key, start an Orchestration job through `/api/orchestrate`. The job status should be `failed`, with an error naming `ANTHROPIC_API_KEY`.

## 6. Infrastructure and removals

- [ ] 6.1 In `docker-compose.yml`, delete the `llm` service, remove `llm` from `api.depends_on`, drop `VLLM_URL`, and add the `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL` passthrough (D8). Verify that `docker compose config` shows no GPU reservation or `llm` service, and that `make start` brings up all services.
- [ ] 6.2 Delete `services/llm/`, `scripts/download_model.sh` and `training/`. Remove `make download-model` and the `training/evaluation_results.json` line from `.gitignore`. Verify that `make help` no longer lists `download-model`, and that `grep -rn "download_model\|training/" Makefile docker-compose.yml .gitignore` is empty.
- [ ] 6.3 Add `scripts/test_llm.py` (it loads `.env` and calls `llm_client.complete`) and point `make test-llm` at it (D9). Verify that `make test-llm` prints a reply and exits 0 with a valid key, and exits non-zero naming `ANTHROPIC_API_KEY` without one.

## 7. Frontend

- [ ] 7.1 In `services/frontend/src/App.tsx`, update the following:
  - the status label: replace "vLLM :8000" with an Anthropic API label
  - the header badges: replace "Llama 3.1 8B" with "Claude Opus 5.5" and remove the LoRA badge
  - the System info rows: drop quantization, LoRA, GPU and the T4 context window
  - the "~7 min on T4" and "20–60s on T4" hints

  Verify that `npm run build` and `npm run lint` pass in `services/frontend`, and that `grep -niE "vllm|llama|lora|T4" services/frontend/src/App.tsx` is empty.

## 8. Documentation

- [ ] 8.1 Update the README:
  - Rewrite the architecture diagram, services, prerequisites, setup, commands table and troubleshooting for the Anthropic API.
  - State that Health Profiles are sent to Anthropic.
  - Replace the vLLM, LoRA and QLoRA sections with a short pointer to ADR 0001 and git history.

  Verify that `grep -niE "vllm|download-model|g4dn|T4 " README.md` returns only the historical pointer, and that the documented commands run as written.
- [ ] 8.2 Update AGENTS.md to match ADR 0001: remove Bedrock, `LLM_PROVIDER`, `BEDROCK_MODEL_ID`, `AWS_REGION`, `make start-vllm` and the vLLM service line, and list `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`. Verify that `grep -niE "bedrock|vllm|LLM_PROVIDER" AGENTS.md` is empty.
- [ ] 8.3 Create `docs/aws-runbook.md` covering:
  - a CPU-only EC2 instance (size confirmed with `docker stats` after the stack is running)
  - a security group restricted to the owner's IP on 3000 and 8002
  - `.env` setup from `.env.example`
  - an Anthropic Console spend limit
  - the rollback steps from design.md

  Verify by following the runbook on the target instance up to a healthy `/api/health`.

## 9. Integration verification

- [ ] 9.1 On a CPU-only host, run `make start`, `make test-llm`, `curl localhost:8002/api/health` and `python scripts/test_orchestration.py`. Verify that all four agents succeed and the output contains a Lab Analysis, Meal Plan, Training Program and Grocery List. Check the logs for `max_tokens` truncation warnings, and raise an agent's `MAX_TOKENS_OVERRIDE` only if any appear.
- [ ] 9.2 Run a repo-wide check: `grep -rniE "vllm|VLLM_URL|/models/llama" --exclude-dir={node_modules,dist,.git,openspec} .` returns matches only in `REFLECTION.md`, `docs/adr/` and the README history pointer. Also verify that `make test` passes.

## Workflow follow-up

- Archive the change once it is reviewed and deployed.
- Stop or terminate the g4dn.xlarge after the CPU-only deployment is confirmed.
