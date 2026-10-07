# Tasks

## 1. Bedrock LLM client

- [x] 1.1 Add `boto3` to `services/api/requirements.txt` and `pytest` as a dev dependency. Verify `pip install -r services/api/requirements.txt` succeeds.
- [x] 1.2 Create the `llm/` package with config loaded from env (`AWS_REGION`, `BEDROCK_MODEL_ID`, `LLM_MAX_TOKENS`, `LLM_READ_TIMEOUT_S`, `LLM_MAX_RETRIES`), a shared `bedrock-runtime` client using botocore adaptive retries and timeouts (design D3, D5), and `LLMProviderError`. Verify with unit tests that defaults apply when the env is unset and env overrides are honored.
- [x] 1.3 Implement `generate_text(system, user, max_tokens?)` on the Converse API, returning content plus `prompt_tokens`/`completion_tokens` (D8). Verify with a botocore `Stubber` unit test, including a throttling response that is retried and an exhausted-retries case that raises `LLMProviderError`.
- [x] 1.4 Implement `generate_structured(system, user, output_model, max_tokens?)`: inline the Pydantic `$ref`s, force a single tool call, return the validated model, and raise on `ValidationError` (D4). Verify with Stubber unit tests for valid tool input and for schema-violating tool input.
- [ ] 1.5 Spike: with real credentials, send each of `LabAnalysisOutput`, `NutritionOutput`, `TrainingOutput`, `GroceryOutput` and the grocery sub-schemas to Bedrock through `generate_structured`. Verify every schema is accepted and returns valid output. Record any schema adjustments in design.md.
- [ ] 1.6 Add `COPY llm/ ./llm/` to `services/api/Dockerfile`. Verify `docker compose build api` succeeds and `python -c "import llm"` works in the container.

## 2. Agents use structured output

- [x] 2.1 Replace `BaseAgent._call_llm` and the parse-retry loop with `generate_structured`, retrying on validation failure up to the configured attempts. Remove the `VLLM_*` constants. Verify with a unit test using a stubbed client: a valid result is returned, and an invalid result is retried and then reported with the validation error.
- [x] 2.2 Port the Lab Analysis and Training agents' `_parse_output` post-processing to run on the validated output. Verify with unit tests using stubbed tool inputs that derived fields (for example calorie targets) match the current behavior.
- [x] 2.3 Rewrite the Nutrition agent to generate all seven days in one `generate_structured` call with a raised `MAX_TOKENS_OVERRIDE`, and remove the half-week split and merge. Verify with a unit test that exactly one LLM call is made and the output contains Monday through Sunday.
- [x] 2.4 Convert Grocery category consolidation and shopping notes to `generate_structured` with small schemas, and fan the category calls out over a small thread pool. Verify with a unit test that per-category results are aggregated correctly.
- [x] 2.5 Delete `_extract_json` and any other dead JSON-recovery code. Verify `grep -rn "_extract_json" agents/` returns nothing and the unit tests pass.

## 3. Orchestration concurrency

- [x] 3.1 Run Nutrition and Training concurrently after Lab Analysis, each with a shallow copy of `context`. Run Grocery only after Nutrition succeeds. Keep per-agent failure isolation and the Lab-failure fallback. Verify with unit tests using fake agents that cover: concurrent start (both started before either finishes), Nutrition failure (Grocery skipped, Training returned), and Lab failure (Nutrition and Training still run).
- [x] 3.2 Make the job executor size come from `MAX_CONCURRENT_JOBS` (default 3), and remove the `VLLM_URL` monkey-patch in `routes/orchestrate.py`. Verify with a test that two jobs reach `running` at the same time, and that a job over the limit stays `pending` until a slot frees.

## 4. API, RAG pipeline and comparator

- [x] 4.1 Route `rag/pipeline.py` generation through `generate_text`, remove the `vllm_url`/`model_name` parameters, and drop `--vllm-url` from `scripts/test_rag.py`. Verify `python scripts/test_rag.py --help` runs and shows no vLLM options.
- [x] 4.2 Make `core/comparator.py` call `generate_text` via `asyncio.to_thread`, keeping the parallel base and RAG calls and the token fields. Remove `VLLM_URL`/`VLLM_MODEL` from `core/config.py`. Verify a stubbed comparator test returns both responses with usage.
- [x] 4.3 Replace the `/api/health` vLLM check with a cached (60s) minimal Converse probe reported as `services.llm` (D7). Verify with tests: stubbed success gives `llm.status = "ok"` with the model ID, and a stubbed `AccessDeniedException` gives `llm.status = "error"` and overall `degraded`.

## 5. Remove the local LLM and training

- [x] 5.1 Remove the `llm` service, GPU reservation and `depends_on: llm` from `docker-compose.yml`, and add the Bedrock env vars to `api`. Delete `services/llm/` and `scripts/download_model.sh`. Verify `docker compose config` is valid and lists no `llm` service.
- [x] 5.2 Delete `training/` and any `models/` usage that is no longer needed. Verify `grep -rn "training/\|models/llama\|adapters" --include=*.py --include=*.yml --include=Makefile .` returns nothing.
- [x] 5.3 Update the Makefile: remove `download-model`, and repoint `test-llm` to `curl /api/health`. Fix the help text. Verify `make help` and `make test-llm` (against a running stack) work.
- [x] 5.4 Update the frontend: replace the "vLLM :8000" status, the Llama/LoRA badges and info rows, and the T4 timing hints with Bedrock/Opus equivalents, driven by `/api/health` `llm`. Verify `npm run build` and `npm run lint` pass in `services/frontend` and that a grep for `vLLM|LoRA|T4|Llama` in `src/` returns nothing.

## 6. Documentation

- [x] 6.1 Update the README: architecture diagram, hardware, prerequisites, model download (removed), running, Makefile reference, troubleshooting, module progress note on the retired Modules 1–2 artifacts, and design decisions. Verify a grep for `vLLM|g4dn|T4|download_model` only matches historical notes.
- [ ] 6.2 Add a "Deploying to the company AWS account (manual)" section: enable Opus 5.5 model access, find the inference profile ID, IAM role policy (inference profile plus underlying model ARNs), CPU instance sizing, IMDS hop limit 2, security group ports, data migration (`scp` or `ingest.py`), env vars, and the `/api/health` check. Verify by following it during group 7.

## 7. Integration verification (company account)

- [ ] 7.1 With Opus 5.5 access enabled and the instance provisioned per the README, run `docker compose up -d`. Verify `/api/health` reports `llm: ok` with the configured model ID and the expected Chroma collection count.
- [ ] 7.2 Run `scripts/test_orchestration.py` and `scripts/test_rag.py` against the deployment. Verify all four agents succeed, the meal plan has seven days, and the RAG output parses.
- [ ] 7.3 Submit two orchestration jobs from the UI at the same time and run one base-vs-RAG comparison. Verify both jobs complete, the comparison shows token counts, and there are no unrecovered throttling errors in the logs. Tune the default token caps and `MAX_CONCURRENT_JOBS` if needed and record the values in design.md.
- [ ] 7.4 Decommission the personal-account instance after verification, keeping an EBS snapshot. Verify the instance is stopped and the snapshot exists.

## Workflow follow-up

- Archive the change after verification, which syncs the three new capability specs into `openspec/specs/`.
