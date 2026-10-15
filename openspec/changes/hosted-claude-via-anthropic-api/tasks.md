# Tasks

## 1. Dependencies, cleanup and offline test harness

- [x] 1.1 Add `anthropic` to `services/api/requirements.txt`, pinned to a minimum version that supports structured outputs and `output_config.effort` (D2). Verify that `docker compose build api` succeeds.
- [x] 1.2 Create `requirements-dev.txt` (`pytest`, `anthropic`, `pydantic`, `httpx`) and add a `make test` target that runs `pytest tests/`. Verify that `pip install -r requirements-dev.txt && make test` runs.
- [x] 1.3 Add `tests/conftest.py` with the autouse offline fixture from D12: a dummy key, a singleton reset, and `socket.connect` blocked. Verify with a test that opens a socket and confirms the fixture makes it fail, and that `make test` passes with networking disabled and `ANTHROPIC_API_KEY` unset.
- [x] 1.4 Add `.env.example` with empty `ANTHROPIC_API_KEY=` and `ANTHROPIC_MODEL=`. Verify that `git check-ignore .env` still matches and that `.env.example` is tracked.
- [x] 1.5 Delete the top-level `llm/` folder and every `__pycache__` directory outside `node_modules` (decision 1). Verify that `find . -name __pycache__ -not -path "*/node_modules/*"` and `ls llm` both return nothing.

## 2. Shared LLM client

- [x] 2.1 Create `agents/llm_client.py` with lazy configuration (`DEFAULT_MODEL = "claude-opus-5-5"`, `LLMConfigError`), the D4 request settings (`effort` `medium`; no `temperature`, `top_p`, `top_k` or `thinking`; `text` blocks selected by type; 600 s timeout) and `complete`/`acomplete`. Verify that `python -c "import agents.llm_client"` succeeds with no key set.
- [x] 2.2 Add `generate(system, user, *, schema, max_tokens)` using `messages.create` with the schema as `output_config.format`, then `schema.model_validate_json` (D5). It raises `LLMTruncatedError` when `stop_reason` is `max_tokens` and an error when it is `refusal`, and lets `ValidationError` propagate (D6). Verify with the tests in 2.4.
- [x] 2.3 Add `check_health()`: check that the key is set, then call `models.retrieve` (D10). Verify with the tests in 2.4.
- [x] 2.4 Write `tests/test_llm_client.py` against a fake SDK client. It must cover:
  - the default `claude-opus-5-5` model and the `ANTHROPIC_MODEL` override
  - the missing-key error, which must name the variable
  - the request containing `effort` `medium` and no `temperature`/`top_p`/`top_k`/`thinking`
  - thinking blocks being ignored
  - the parsed model being returned from `generate`
  - `LLMTruncatedError` on `max_tokens`
  - the refusal error
  - the `check_health` ok and error shapes, never containing the key, and never calling `messages.create`

  Verify that `make test` passes offline.

## 3. Agents: structured output and output limits

- [x] 3.1 Add the LLM-facing wrapper models `HalfWeekPlan`, `ConsolidatedItem`/`ConsolidatedItems` and `ShoppingTips` to `agents/schemas.py`, leaving the public contracts unchanged (D5). Verify that `python -c "from agents.schemas import HalfWeekPlan, ConsolidatedItems, ShoppingTips"` succeeds.
- [x] 3.2 Rework `BaseAgent` (D5/D6):
  - `_call_llm` delegates to `llm_client.generate` with the agent's schema and `MAX_TOKENS`, keeping the existing system-prompt intent without the "JSON only" wording.
  - `run()` retries only on `ValidationError` and never on `LLMTruncatedError`.
  - Replace `_parse_output` with `_postprocess`.
  - Remove `_extract_json`, `VLLM_*`, `TEMPERATURE`, `MAX_TOKENS_OVERRIDE` and `httpx`.

  Verify with `tests/test_base_agent.py`, using a stub agent and a fake client. It must show that the schema and `max_tokens` are forwarded, that a validation failure is retried and then fails, and that a truncation fails with no retry. `make test` must pass.
- [x] 3.3 Move the Lab Analysis and Training agents to the `LabAnalysisOutput` and `TrainingOutput` schemas, with `MAX_TOKENS` 4096. Remove the JSON-format instructions and schema examples from their prompts. Lab Analysis keeps the guard that clears `analysed_markers` when there are no Blood Markers; drop the list flattening. Verify with tests showing that the no-markers guard still clears the hallucinated markers, and that `grep -nE "_extract_json|JSON only|schema_example" agents/lab_analysis_agent.py agents/training_agent.py` is empty.
- [x] 3.4 Move the Nutrition agent to `HalfWeekPlan` for each half-week call, with `MAX_TOKENS` 8192, and keep the merge and average calculation. Verify with a test where two fake half replies merge into a valid 7-day `NutritionOutput`, and a test where a truncated half fails the agent.
- [x] 3.5 Move the Grocery agent to `ConsolidatedItems` and `ShoppingTips` with `MAX_TOKENS` 2048, removing the regex extraction and keeping the raw-item and default-tip fallbacks. Verify with tests for the success path and for the fallback when a call fails or is truncated, and that `grep -n "re.search" agents/grocery_agent.py` is empty.
- [x] 3.6 Add a test in which `run()` fails with an error naming `ANTHROPIC_API_KEY` when the key is unset. Verify that `make test` passes.

## 4. Orchestration scheduling

- [x] 4.1 In `agents/orchestrator.py`, run Nutrition and Training in parallel after Lab Analysis (D7). Each gets its own context copy and runs through `_run_agent`. Their results are appended in the order Nutrition, then Training; Grocery waits for Nutrition. Remove the T4 comment. Verify with `tests/test_orchestrator.py` using stub agents, showing:
  - both agents start before either finishes (a `threading.Barrier` that would time out if they ran one after the other)
  - a Training failure does not stop Nutrition or Grocery
  - Grocery is reported `skipped` when Nutrition fails
  - the result order is fixed even when Training finishes first
- [x] 4.2 In `services/api/routes/orchestrate.py`, remove the `VLLM_URL` import and monkey-patch, keep `max_workers=1`, and change its comment to the cost guard (D8). Verify with a test that submits two jobs to the route's executor with a stub orchestrator and sees the second stay `pending` until the first finishes. `grep -rn VLLM agents services/api` must return nothing.

## 5. Comparator, legacy pipeline and CLI

- [x] 5.1 Rename `comparator.call_vllm` to `call_llm`, built on `acomplete` with `max_tokens` 2048. Keep the two calls per chat and the result dict shape, and take token counts from `usage`. Remove `VLLM_URL` and `VLLM_MODEL` from `core/config.py`. Verify with an offline test in which the fake client produces both responses and a truncation sets `error`. Then, live, `/api/chat` must return two non-empty answers with `error` `null`.
- [x] 5.2 In `rag/pipeline.py`, rename `_call_vllm` to `_call_llm` via `complete` (`max_tokens` 4096). Drop the `vllm_url`, `model_name`, `temperature` and `requests` constructor arguments, and keep the file (decision 10). Remove `--vllm-url` from `scripts/test_rag.py`. Verify that `python scripts/test_rag.py --help` does not list it, and that `python scripts/test_rag.py --type meal_plan` returns parsed JSON against the running stack.
- [x] 5.3 Update `rag/prompt_builder.py` wording and comments (D13). Verify that `grep -rniE "vllm|llama|privacy-preserving" rag/` is empty.

## 6. API health and smoke test

- [x] 6.1 Replace the `vllm` check in `/api/health` with `services.llm` from `check_health()`. Verify with an offline FastAPI `TestClient` test covering the ok, missing-key and invalid-model cases, all with no `vllm` key. Then, live, `curl localhost:8002/api/health` must show `services.llm.status == "ok"` and `claude-opus-5-5`.
- [x] 6.2 Add `scripts/test_llm.py` and point `make test-llm` at it (D10). Verify that `make test-llm` prints a reply and exits 0 with a valid key, and exits non-zero naming `ANTHROPIC_API_KEY` without one.
- [x] 6.3 Verify the missing-key path end to end: with no key, start a job through `/api/orchestrate`. The agents must fail with errors naming `ANTHROPIC_API_KEY`, and `/api/health` must report `degraded`.

## 7. Infrastructure removals

- [x] 7.1 In `docker-compose.yml`, delete the `llm` service, remove it from `api.depends_on`, drop `VLLM_URL` and add the Anthropic passthrough (D11). Verify that `docker compose config` shows no GPU reservation and no `llm` service, and that `make start` brings every service up.
- [x] 7.2 Delete `services/llm/`, `scripts/download_model.sh` and `training/`. Remove `make download-model` and the `training/evaluation_results.json` line from `.gitignore`. Verify that `make help` no longer lists `download-model`, and that `grep -rn "download_model\|training/" Makefile docker-compose.yml .gitignore` is empty.

## 8. Frontend

- [x] 8.1 In `services/frontend/src/App.tsx`, update:
  - the status label to show the Anthropic API
  - the model badge to "Claude Opus 5.5", removing the LoRA badge
  - the System info rows, dropping quantization, LoRA, GPU and the T4 context window
  - the T4 timing hints

  Verify that `npm run build` and `npm run lint` pass, and that `grep -niE "vllm|llama|lora|T4" services/frontend/src/App.tsx` is empty.

## 9. Documentation

- [x] 9.1 Update AGENTS.md to match ADR 0001 (decision 1): remove Bedrock, `LLM_PROVIDER`, `BEDROCK_MODEL_ID`, `AWS_REGION`, `make start-vllm`, the vLLM service and `docs/aws-runbook.md`, and add `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`. Verify that `grep -niE "bedrock|vllm|LLM_PROVIDER|aws-runbook" AGENTS.md` is empty and that `docs/aws-runbook.md` does not exist.
- [x] 9.2 Add the decision 12 consequence to ADR 0001: Blood Markers, medical history and medications are sent to Anthropic unredacted, with no zero-data-retention agreement, which is accepted for personal/demo use. Verify by reading the ADR's Consequences section.
- [x] 9.3 Amend decision 5 in `docs/decisions.md`: Opus 5.5 rejects non-default `temperature` and disabling thinking, so requests omit both and pin `effort` to `medium`. Verify that the decision text matches design D4.
- [x] 9.4 Rewrite the README for the Anthropic API. Cover the architecture, services, prerequisites, setup, commands (`make test`, `make test-llm`) and troubleshooting. Add a "Deployment (personal/demo)" section on the security group, the spend limit and the health data sent to Anthropic, and replace the vLLM/LoRA sections with a pointer to ADR 0001. Verify that `grep -niE "vllm|download-model|g4dn|bedrock" README.md` returns only the history pointer, and that the documented commands run as written.

## 10. Integration verification

- [ ] 10.1 On a CPU-only host with a real key, run `make start`, `make test-llm`, `curl localhost:8002/api/health` and `python scripts/test_orchestration.py`. Verify that:
  - all four agents succeed
  - the logs show Nutrition and Training overlapping
  - the output contains a Lab Analysis, Meal Plan, Training Program and Grocery List

  If any `LLMTruncatedError` appears, raise only that agent's `MAX_TOKENS` and re-run.
- [x] 10.2 Run `make test` with networking disabled; it must pass. Run `grep -rniE "vllm|VLLM_URL|/models/llama|_extract_json" --exclude-dir={node_modules,dist,.git,openspec} .`; it must match only `REFLECTION.md`, `docs/` and the README history pointer.

## Workflow follow-up

- Archive the change once it is reviewed and deployed.
- Stop or terminate the g4dn.xlarge after the CPU-only deployment is confirmed.
