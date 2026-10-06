# Research: How the AI Health Orchestration codebase works today

- Date: 2026-10-06
- Git HEAD: `0bb0dec` (feat: add Reflection note)
- Scope: components, how they interact, data models, and how the system is built, run, and tested.

## Summary

The repo is a Docker Compose stack of five services. **llm** runs vLLM serving Llama 3.1 8B. **vector-db** runs ChromaDB. **mcp-rag** is an MCP server that wraps the ChromaDB retriever. **api** is a FastAPI service that hosts the multi-agent orchestrator. **frontend** is a React/Vite dashboard.

A request from the "Orchestrate" tab goes to `POST /api/orchestrate`. The API runs a sequential four-agent pipeline (Lab Analysis → Nutrition → Training → Grocery) in a background thread, and the browser polls for the result. Each agent builds a prompt from the user's `HealthProfile`/`UserGoals`, adds RAG context, calls vLLM, and parses the JSON reply into a Pydantic model from `agents/schemas.py`. Three of the agents fetch RAG context through the MCP server; the Grocery agent does no retrieval.

A second, separate path ("Chat & Compare") calls `POST /api/chat`. That path sends the same question to vLLM twice, once with RAG context and once without, and returns a keyword-overlap "grounding score". It talks to ChromaDB directly over HTTP.

Outside the running stack, standalone scripts handle corpus ingestion (`scripts/ingest.py`), RAG experiments and evaluation (`rag/`, `scripts/test_rag.py`), and QLoRA fine-tuning (`training/`). There is no automated unit-test suite. Testing is done with manual CLI scripts run against live services.

---

## 1. Runtime topology (Docker Compose)

`docker-compose.yml:1-70`

| Service | Build / image | Host port → container | Key config |
|---|---|---|---|
| `llm` | `services/llm/Dockerfile` (`vllm/vllm-openai:latest`) | 8000 → 8000 | `--model /models/llama --quantization bitsandbytes --load-format bitsandbytes --max-model-len 16384 --gpu-memory-utilization 0.85 --max-num-seqs 8 --enforce-eager --dtype float16` (`docker-compose.yml:5-14`); mounts `./models/llama` (`:17-18`); reserves 1 NVIDIA GPU (`:19-25`) |
| `vector-db` | `chromadb/chroma:latest` | 8001 → 8000 | volume `./data/chroma:/data` (`:28-34`) |
| `api` | `services/api/Dockerfile`, context = repo root | 8002 → 8002 | env `VLLM_URL=http://llm:8000`, `CHROMA_URL=http://vector-db:8000`, `MCP_RAG_URL=http://mcp-rag:8004/mcp`; depends on llm, vector-db, mcp-rag (`:36-50`) |
| `frontend` | `services/frontend/Dockerfile` | 3000 → 3000 | depends on api (`:52-58`) |
| `mcp-rag` | `services/mcp-rag/Dockerfile`, context = repo root | 8004 → 8004 | env `CHROMA_HOST=vector-db`, `CHROMA_PORT=8000` (`:60-70`) |

Container images:
- **api** (`services/api/Dockerfile:1-9`): `python:3.12-slim`. It installs `services/api/requirements.txt` (fastapi, uvicorn, httpx, pydantic 2.9.2, sentence-transformers, torch 2.5.1, chromadb 1.5.8, mcp), then copies `services/api/`, `agents/` and `rag/` into `/app`. It runs `uvicorn main:app --reload` on port 8002.
- **mcp-rag** (`services/mcp-rag/Dockerfile:1-20`): reuses the API requirements file, but installs CPU-only torch first, then `mcp[cli]`. It copies `rag/` and `mcp_server.py` and runs `python mcp_server.py`.
- **frontend** (`services/frontend/Dockerfile:1-12`): `node:20-slim`, `npm ci`, then `npm run dev` (Vite dev server, not a production build).
- **llm** (`services/llm/Dockerfile:1-16`): the same vLLM args as compose, set as the default `CMD`.

Internal vs. host addressing: containers use service DNS names (`llm:8000`, `vector-db:8000`, `mcp-rag:8004`). Host-side scripts use `localhost:8000` for vLLM and `localhost:8001` for ChromaDB.

## 2. Data models (`agents/schemas.py`)

All pipeline contracts are Pydantic v2 models. The docstring data-flow diagram is at `agents/schemas.py:1-16`. Public re-exports are in `agents/__init__.py:1-13`.

**Enums** (`:27-60`)
- `FitnessGoal`: fat_loss, muscle_gain, maintenance, performance
- `ActivityLevel`: sedentary … extra_active
- `FitnessLevel`: beginner, intermediate, advanced
- `MarkerStatus`: optimal, normal, borderline, borderline-high/low, elevated, high, low, deficient, critical
- `AgentStatus` (`:297-300`): success, failed, skipped

**Inputs**
- `BloodMarker{name, value, unit}` (`:63-67`)
- `ScaleMetrics{weight_kg, body_fat_pct?, muscle_mass_kg?, bmi?}` (`:70-75`)
- `HealthProfile` (`:78-106`): fields age (18–100), sex, height_cm (100–250), scale_metrics, blood_markers[], medical_history[], medications[], allergies[]
- `UserGoals` (`:109-118`): fields primary_goal, activity_level, fitness_level, training_days_per_week (1–7), dietary_preferences[], notes?

**Agent outputs**
- `LabAnalysisOutput` (`:142-179`): analysed_markers (`AnalysedMarker`, `:124-139`), dietary_constraints[], training_constraints[], estimated_tdee?, recommended_calories?, overall_health_summary, sources_used[]
- `NutritionOutput` (`:205-226`): weekly_plan is a list of `DayPlan` (`:195-202`) with `min_length=7, max_length=7`, and each `DayPlan` contains `Meal` items (`:182-192`). Also holds averages, caloric_strategy, constraint_adherence, sources_used.
- `TrainingOutput` (`:259-268`): weekly_program is a list of `TrainingSession` (`:241-248`), each with `Exercise` items (`:232-238`); also rest_days and progression_scheme (`ProgressionScheme`, weeks 2–4, `:251-256`).
- `GroceryOutput` (`:281-291`): items_by_category is a `dict[str, list[GroceryItem]]` (`:273-278`); also total_items, shopping_notes, estimated_weekly_cost_eur.

**Aggregate**
- `AgentResult{agent, status, duration_s, error?}` (`:303-308`)
- `OrchestrationResult` (`:311-324`): four optional agent outputs, agent_results[], total_duration_s, and pipeline_version="1.0"

## 3. Agent layer (`agents/`)

### 3.1 BaseAgent (`agents/base.py`)
- Module constants (`:24-29`): `VLLM_URL="http://localhost:8000/v1/chat/completions"`, `MAX_TOKENS=2048`, `TEMPERATURE=0.2`, `REQUEST_TIMEOUT=600`, `MAX_RETRIES=2`.
- `run()` (`:40-61`) is a template method: `_retrieve` → `_build_prompt` → loop of `_call_llm` + `_parse_output`. A `JSONDecodeError`/`ValueError` is retried up to `MAX_RETRIES` times, after which it raises `RuntimeError`.
- Subclasses must implement three abstract hooks: `_build_query`, `_build_prompt` and `_parse_output` (`:63-73`).
- Default `_retrieve()` (`:75-111`):
  1. Calls `self.retriever.retrieve(query_type=self.COLLECTION_KEY, n_results_per_collection=6, max_distance=0.8)`.
  2. Formats the chunks into a `=== RETRIEVED CLINICAL/NUTRITIONAL GUIDELINES ===` block.
  3. Returns `""` if there is no retriever or on error.
- `_call_llm()` (`:113-145`) does a synchronous `httpx` POST to the OpenAI-compatible vLLM endpoint. The system message is fixed to "valid JSON only". `max_tokens` can be overridden through `MAX_TOKENS_OVERRIDE` (`:131`).
- `_extract_json()` (`:147-159`) tries three strategies in order: a fenced ```json block, then a direct `json.loads`, then the substring from the first `{` to the last `}`.
- `_format_profile()` / `_format_goals()` (`:161-196`) render the inputs as labelled text blocks.

### 3.2 LabAnalysisAgent (`agents/lab_analysis_agent.py`)
- `COLLECTION_KEY="lab_analysis"`, which routes to `public_health_recommendations` (`:63-66`).
- The query lists the submitted marker names, or uses a generic biomarker query if there are none (`:70-89`).
- `_retrieve` is overridden to go through MCP (`:93-125`). `_mcp_retrieve` (`:127-140`) opens `streamablehttp_client(MCP_RAG_URL)` and calls tool `retrieve_context`, reading `result.structuredContent["result"]`.
- TDEE and calorie target are computed deterministically (`_calculate_calories`, `:269-293`):
  - BMR uses Mifflin-St Jeor.
  - It is multiplied by `ACTIVITY_MULTIPLIERS` (`:47-53`).
  - `GOAL_CALORIE_ADJUSTMENTS` (`:55-60`) then adds −400, +300, 0 or +200 kcal depending on the goal.
  - The results are injected into the prompt as values the LLM must use.
- The prompt uses a placeholder-only schema example (`:142-242`). It has separate instruction branches for when markers are present and when they are absent (`:207-228`).
- `_parse_output` (`:244-265`) does two things:
  - If no markers were submitted, it forces `analysed_markers=[]`.
  - It flattens list-typed `dietary_implication`/`training_implication`/`interpretation` fields to strings.

### 3.3 NutritionAgent (`agents/nutrition_agent.py`)
- `COLLECTION_KEY="meal_plan"`, which routes to `nutrition_guidelines` and `food_and_recipes` (`:48-51`).
- It overrides `run()` (`:55-87`): one retrieval, then **two** LLM calls (Mon–Thu, then Fri–Sun, `:44-45`), then `_merge_and_parse`. The base retry loop is not used.
- Target calories come from `lab.recommended_calories`. If that is missing, it falls back to 2000 kcal plus a goal adjustment (`:168-172`).
- `_build_half_prompt` (`:179-236`) includes RAG context, profile, goals, lab dietary constraints and a JSON example with a `days` array.
- `_merge_and_parse` (`:238-289`):
  - Concatenates the days and requires at least 7.
  - Computes the averages.
  - Generates `caloric_strategy`/`constraint_adherence` text in code.
  - Sets `sources_used=[]`.
- MCP retrieval: `_retrieve` (`:108-141`). `_mcp_retrieve` is defined **twice**: `:143-156` parses `result.content[0].text`, and `:291-303` uses `result.structuredContent`. Python keeps the later definition, so `:291` is the one that runs.
- `_build_prompt`/`_parse_output` are stubs that just `pass` (`:158-164`).

### 3.4 TrainingAgent (`agents/training_agent.py`)
- `COLLECTION_KEY="gym_program"`, which routes to `gym_programming` (`:55-58`).
- The query combines `GOAL_TRAINING_STYLE[goal]` (`:47-52`), fitness level, days per week, and up to two lab training constraints (`:60-77`).
- MCP retrieval: `_retrieve`/`_mcp_retrieve` (`:79-126`).
- The prompt (`:128-217`) asks for exactly N sessions, 4–6 exercises each, and a four-week progression where week 4 is a deload. It includes a concrete JSON example.
- `_parse_output` is `TrainingOutput(**_extract_json(raw))` (`:219-221`). It uses the BaseAgent retry loop.

### 3.5 GroceryAgent (`agents/grocery_agent.py`)
- `COLLECTION_KEY="grocery"`. It overrides `run()` (`:123-171`) and never calls `_retrieve`, so it does no RAG.
- Steps:
  1. Flatten all `meal.ingredients` from `context["nutrition"]`. It raises if nutrition output is missing.
  2. Keyword pre-bucket the ingredients into 8 `CATEGORIES` (`:37-46`) using `CATEGORY_KEYWORDS` (`:49-83`) and `_pre_bucket` (`:86-103`). Matching is first-match in dict order.
  3. Make one LLM call per non-empty category to consolidate and sum quantities (`_consolidate_category`, `:175-236`). Arrays are extracted with the regex `\[.*\]`. If that fails, it falls back to the raw items, capped at 10, with `total_quantity="see meal plan"`.
  4. Make one LLM call for four shopping tips (`_get_shopping_notes`, `:238-269`). There is a hard-coded fallback list.
- `estimated_weekly_cost_eur` is always `None` (`:163-168`). Per-item costs come from the LLM.

### 3.6 Orchestrator (`agents/orchestrator.py`)
- The constructor (`:60-86`) builds one shared `HealthRetriever(chroma_host, chroma_port)`. That loads the embedding model and connects to ChromaDB. It then passes the retriever into all four agents.
- `run()` (`:88-212`) executes the agents **sequentially**, sharing a mutable `context` dict:
  1. LabAnalysis. On success it sets `context["lab_analysis"]` (`:110-128`).
  2. Nutrition always runs. On success it sets `context["nutrition"]` (`:130-148`).
  3. Training always runs and reads `lab_analysis` from context (`:150-167`).
  4. Grocery runs only if nutrition succeeded. Otherwise it records a `SKIPPED` `AgentResult` (`:169-195`).
- `_run_agent()` (`:216-246`) wraps each agent call, catches every exception, and records a `SUCCESS`/`FAILED` status plus duration. The pipeline itself does not raise.

Data flow between agents:

```
HealthProfile + UserGoals ─┬─> LabAnalysisAgent ──> context["lab_analysis"]
                           │                              │ dietary_constraints, recommended_calories
                           ├─> NutritionAgent  <──────────┤ ──> context["nutrition"]
                           │                              │ training_constraints
                           ├─> TrainingAgent   <──────────┘
                           └─> GroceryAgent    <── context["nutrition"].weekly_plan[*].meals[*].ingredients
```

## 4. RAG layer (`rag/`)

### 4.1 HealthRetriever (`rag/retriever.py`)
- There are four collections (`ALL_COLLECTIONS`, `:35-40`): `nutrition_guidelines`, `food_and_recipes`, `gym_programming`, `public_health_recommendations`.
- `COLLECTION_MAP` (`:22-33`) maps a `query_type` to its collection(s). Examples: `meal_plan` → nutrition + food, `lab_analysis` → public_health, `grocery` → food, and `full` → all four.
- The constructor (`:64-83`) creates `chromadb.HttpClient`, loads `SentenceTransformer("all-MiniLM-L6-v2")` on CPU, and caches collection handles (`:85-93`).
- `query_collection()` (`:99-144`) embeds the query, calls `col.query(...)`, and maps each hit to a `RetrievedChunk{content, collection, source, category, distance, chunk_index}` (`:43-51`).
- `retrieve()` (`:146-198`) routes the query, filters out chunks with `distance > max_distance`, sorts globally by distance, and de-duplicates on the first 100 characters of content.
- `health_check()` (`:200-204`) returns document counts per collection.

### 4.2 HealthPromptBuilder (`rag/prompt_builder.py`)
- `PromptStrategy.STRUCTURED` / `NAIVE` (`:22-24`). `SYSTEM_PROMPT` (`:39-54`).
- `build()` (`:88-128`) trims the chunks to a `max_context_chars` budget (default 12,000; `_trim_to_budget`, `:149-162`), then formats them:
  - Structured: one `[CONTEXT: <label>]` section per collection (`:164-207`).
  - Naive: plain concatenation (`:209-225`).
- `build_no_rag()` (`:130-145`) builds the baseline prompt with no context.

### 4.3 HealthRAGPipeline (`rag/pipeline.py`)
This is the single-call RAG pipeline. The agents and the API do not use it; it is used by `scripts/test_rag.py` and `rag/evaluate.py`.
- `OUTPUT_TYPE_CONFIG` (`:41-84`) defines the retrieval query templates per output type. `OUTPUT_SCHEMAS` (`:87-171`) holds the JSON schemas as strings. These schemas differ from the shapes in `agents/schemas.py`.
- `query()` (`:228-314`) runs these steps: build retrieval query → retrieve (`max_distance=0.85`) → build user request → build prompt → `requests.post` to vLLM (`_call_vllm`, `:373-398`) → `_parse_json` (strips fences, `:400-416`). It returns a `RAGResult` (`:174-186`).

### 4.4 RAG evaluation (`rag/evaluate.py`)
- `EVAL_CASES` (`:35-~100`) contains 5 named cases.
- `run_evaluation()` (`:196-~321`) runs each case with RAG and without it. It compares JSON-parse success and schema field counts (`_count_schema_fields` `:148`, `_compare_results` `:168`), then writes `rag/evaluation_results.json`.

## 5. MCP RAG server (`services/mcp-rag/mcp_server.py`)
- `FastMCP("health-rag-corpus", host="0.0.0.0", port=8004)` (`:22`). It is served with `transport="streamable-http"` (`:93-94`), so the endpoint is `/mcp`.
- A single module-level `HealthRetriever` is created at startup using `CHROMA_HOST`/`CHROMA_PORT` from the environment (`:19-25`).
- Tools:
  - `retrieve_context(query_text, query_type="full", n_results_per_collection=4, max_distance=1.0)` (`:42-60`)
  - `search_nutrition_guidelines`, `search_food_and_recipes`, `search_gym_programming`, `search_public_health_recommendations` (`:63-84`), each taking `(query, top_k=5)`
  - `corpus_health_check()` (`:87-90`)
- Results are serialized to dicts by `_serialize` (`:28-39`).
- The only consumers are the LabAnalysis, Nutrition and Training agents, and they call only `retrieve_context`. They pass `n_results_per_collection=6` and `max_distance=0.8`.

## 6. API service (`services/api/`)

- `main.py:14-25` creates the FastAPI app (v0.4.0) with fully open CORS. It mounts the `chat`, `search` and `orchestrate` routers under `/api`.
- `GET /api/health` (`main.py:28-54`) checks two things: vLLM `/v1/models`, and ChromaDB `/api/v2/heartbeat` plus the collection list. It returns `ok` or `degraded`.
- `core/config.py:1-4` hard-codes `VLLM_URL="http://llm:8000"`, `CHROMA_URL="http://vector-db:8000"`, `VLLM_MODEL`, and `REQUEST_TIMEOUT=600`. The `VLLM_URL`/`CHROMA_URL` environment variables set in compose are not read here.

### 6.1 Orchestration endpoints (`routes/orchestrate.py`)
- An in-memory job store `_jobs` and a `ThreadPoolExecutor(max_workers=1)` (`:42-51`).
- At import time, `agents.base.VLLM_URL` is monkey-patched to `{VLLM_URL}/v1/chat/completions` (`:66-67`). The Chroma host and port are parsed from `CHROMA_URL` (`:56-63`).
- `get_orchestrator()` (`:72-80`) creates the `Orchestrator` lazily as a singleton.
- Endpoints:
  - `POST /api/orchestrate` (`:112-128`) takes `OrchestrateRequest{profile: HealthProfile, goals: UserGoals}` (`:105-107`). It creates an 8-character job id, submits `_run_pipeline` (`:85-101`) to the executor, and returns `{job_id, status:"pending"}`.
  - `GET /api/orchestrate/status/{job_id}` (`:132-143`) returns the status and duration.
  - `GET /api/orchestrate/result/{job_id}` (`:146-154`) returns an `OrchestrationResult`, or 400 if the job is not complete.
  - `GET /api/orchestrate/sample-request` (`:157-186`) returns a canned profile with 7 markers.

### 6.2 Chat / compare endpoint (`routes/chat.py`, `core/comparator.py`)
- `POST /api/chat` takes `{query, health_profile: dict, query_type, top_k}` (`routes/chat.py:8-23`) and calls `run_comparison`.
- `comparator.py` does **not** use `rag/retriever.py`. Instead:
  - It has its own embedder singleton (`:24-28`).
  - It caches collection name→UUID via the ChromaDB v2 REST API (`:31-42`).
  - It queries each collection over HTTP (`retrieve_chunks`, `:45-82`) and returns `{collection, excerpt (first 400 chars), score = 1 - distance, metadata}`.
  - It has its own `COLLECTION_MAP` (`:15-21`), whose `lab_analysis` entry also includes `nutrition_guidelines`.
- `run_comparison` (`:114-177`):
  1. Runs retrieval and the base LLM call concurrently with `asyncio.gather`.
  2. Builds `[CONTEXT: …]` blocks and makes the RAG LLM call (`call_vllm`, `:85-111`, max_tokens 1024, temperature 0.3).
  3. Scores both answers with `compute_grounding_score`.
- `core/metrics.py:4-31` defines the grounding score as the share of the response's significant words (more than 3 characters, not a stop word) that appear in the retrieved excerpts. `count_tokens_approx` is `len/4` (`:34-36`).

### 6.3 Search endpoint (`routes/search.py:7-18`)
- `POST /api/search {query, top_k=8, collections=[all four]}` → `retrieve_chunks` → `{sources, total}`.

## 7. Frontend (`services/frontend/`)

- Stack: React 19, Vite 8, Tailwind 4, recharts and lucide-react (`package.json:12-33`). The scripts are `dev`, `build` (`tsc -b && vite build`), `lint` and `preview`.
- The Vite dev server listens on `0.0.0.0:3000` and proxies `/api` to `http://api:8002` with 600-second timeouts (`vite.config.ts:6-17`).
- `src/main.tsx:1-10` mounts `<App/>`.
- `src/types.ts:1-44` defines `Source`, `LLMResponse`, `ChatResult` and the form-level `HealthProfile`. This form type is flat: weight, height, body fat, LBM, LDL, HDL, glucose, creatinine, goal and notes. It is different from the backend `HealthProfile`.
- `src/useChat.ts:11-50` picks `query_type` with regex keyword matching (`detectQueryType`), then POSTs to `/api/chat`.
- `src/App.tsx`:
  - It has a single component with six views selected by `view` state (`:18`, `:198-205`): chat (`:256`), orchestrate (`:362`), analytics (`:604`), kb (`:641`), profile (`:683`) and config (`:731`).
  - `buildOrchestrationRequest` (`:65-100`) maps the form profile to the backend schema:
    - It hard-codes `age 38`, `sex male`, `moderately_active`, `intermediate`, and `4` training days.
    - Defaults: height 178, weight 80.
    - Only LDL, HDL and glucose become `blood_markers`; creatinine is not sent.
    - `notes` becomes `medical_history`.
  - `handleOrchestrate` (`:155-196`) POSTs `/api/orchestrate`, polls `/status/{id}` every 10 s, then GETs `/result/{id}`.
  - The local TypeScript `OrchestrationResult` interface is defined at `:22-60`.
  - `handleKbSearch` (`:137-153`) calls `/api/search`.
  - The analytics view charts per-session latency and grounding history kept in React state (`:110`, `:128-135`, `:604-637`).
  - The config view shows static system info and a `topK` input (`:731-755`).

## 8. Corpus ingestion (`scripts/ingest.py`)
- Paths are hard-coded: `CORPUS_BASE=/home/ubuntu/ai-health-orchestration/data/corpus` (`:27`), with one subfolder per collection (`:29-34`). ChromaDB is reached at `localhost:8001` (`:36-37`).
- Embedding model: `all-MiniLM-L6-v2`.
- Collections are created with `hnsw:space=cosine` (`:350-353`).
- Chunking:
  - Sliding window of 400 words with 60-word overlap (`chunk_text`, `:94-110`) for nutrition and gym PDF/DOCX files.
  - Paragraph chunking (`chunk_by_paragraph`, `:113-134`) for public-health documents.
  - One chunk per row for `All_Diets.csv` recipes (`:217-251`).
  - One chunk per USDA food, with key nutrients, from the Foundation and SR Legacy CSVs (`:142-214`, `:298-324`).
- `main()` (`:378-446`) processes the four categories in order and upserts in batches of 100. Ids follow the pattern `{collection}_{i}`.
- Extra Python dependencies used here but not listed in any requirements file: PyMuPDF (`fitz`), `python-docx`, `pandas`, `tqdm`.
- The `data/corpus/` and `data/chroma/` folders are git-ignored (`.gitignore:3,11,16-17`).

## 9. Model and fine-tuning (`training/`, `scripts/download_model.sh`)
- `scripts/download_model.sh:16-65` needs `HF_TOKEN`. It creates `~/.venv-hf` and runs `snapshot_download` to fetch `meta-llama/Meta-Llama-3.1-8B-Instruct` into `./models/llama`. Run it with `make download-model` (`Makefile:11-12`); note the Makefile help text still says "Mistral 7B AWQ" (`Makefile:5`).
- `training/config.yaml:1-59` configures QLoRA:
  - 4-bit NF4 with double quantization.
  - LoRA with r=16, α=32 and dropout 0.05 on all attention and MLP projections.
  - 3 epochs, learning rate 2e-4 with a cosine schedule.
  - Output goes to `./models/adapters/health-v1`.
- `training/train.py` loads the config (`:41`) and formats JSONL examples into Llama 3.1 chat templates (`:51-87`). It then loads the model in 4-bit, applies LoRA, trains with `trl.SFTTrainer` (`:190`), and runs an inference smoke test. It requires CUDA (`:271-299`).
- Dataset: `training/dataset/processed/health_training_data.jsonl`, 20 lines of `{instruction, input, output}`.
- `training/evaluate.py` compares the base model and base + adapter on `TEST_PROMPTS` (`:27`), checking JSON validity (`:104-200`).
- The vLLM service command in `docker-compose.yml` does not pass LoRA flags, so it serves the base model at `/models/llama`. The config view in the UI does display "LoRA adapter health-v1" (`App.tsx:749`).

## 10. Build, run, and test

**Build / run** (`Makefile:1-27`, `AGENTS.md:15-19`)
- `make download-model`: fetches the model weights.
- `make start`: `docker compose up -d`. `make stop`: `docker compose down`. `make logs`: `docker compose logs -f`.
- `make test-llm`: sends a curl chat completion to `localhost:8000`.
- The corpus is populated with `python scripts/ingest.py`, run on the host against `localhost:8001`.

**Testing.** There is no pytest/unittest suite and no CI workflow; `.github/` contains only `skills/`. Verification uses these tools against the live stack:
- Health: `curl http://localhost:8002/api/health` (`services/api/main.py:28`).
- RAG: `scripts/test_rag.py` (`:1-268`). Its modes are `--health-check`, `--type {meal_plan,gym_program,grocery_list,lab_analysis}`, `--compare` (RAG vs no-RAG) and `--evaluate` (runs `rag.evaluate.run_evaluation`). Defaults are vLLM `http://localhost:8000` and Chroma host `localhost`.
- Full pipeline: `scripts/test_orchestration.py` (`:1-220`).
  - It builds a full profile (`:49`) or a minimal one (`--minimal`, `:85`).
  - It runs `Orchestrator(chroma_host="localhost", chroma_port=8001)` in-process (`:199-200`) and prints a summary (`:112`).
  - `--output file.json` saves the result.
  - It exits with code 1 if any agent `FAILED` (`:210-217`).
  - Because the agents' MCP URL defaults to `http://mcp-rag:8004/mcp` and `BaseAgent.VLLM_URL` defaults to `localhost:8000`, running it on the host uses localhost vLLM for generation. MCP retrieval resolves only if the `MCP_RAG_URL` environment variable points at a reachable server (`agents/*_agent.py:41-42`).
- Frontend: `npm run lint` and `npm run build` (`tsc -b`) are available. There are no frontend tests.
- Model: `training/evaluate.py` (base vs fine-tuned) and `rag/evaluate.py` (RAG vs no-RAG).

## 11. Cross-component request flows

**Orchestrate (UI → result)**
```
Browser (App.tsx handleOrchestrate)
  └─POST /api/orchestrate ─> Vite proxy :3000 ─> api:8002 routes/orchestrate.py
        └─ThreadPoolExecutor ─> Orchestrator.run
             ├─ LabAnalysisAgent ─MCP retrieve_context─> mcp-rag:8004 ─> HealthRetriever ─> vector-db:8000
             │                    └─httpx─> llm:8000 /v1/chat/completions
             ├─ NutritionAgent   ─MCP─> mcp-rag ; 2× vLLM calls
             ├─ TrainingAgent    ─MCP─> mcp-rag ; 1–2× vLLM calls (retry)
             └─ GroceryAgent     ─(no RAG)─ ; ≤9× vLLM calls (8 categories + notes)
  ◄─poll GET /status/{id} every 10s ; GET /result/{id} ─> OrchestrationResult JSON
```

**Chat & Compare**
```
Browser useChat ─POST /api/chat─> comparator.run_comparison
   ├─ asyncio.gather( retrieve_chunks ─REST─> vector-db , call_vllm(base) ─> llm )
   └─ call_vllm(rag) ─> llm ; grounding scores ─> ChatResult
```

**Knowledge-base search**: `POST /api/search` → `comparator.retrieve_chunks` → ChromaDB REST.

There are three independent ChromaDB access paths:
1. `rag/retriever.HealthRetriever` (Python client). Used by the MCP server, the Orchestrator constructor, and `HealthRAGPipeline`.
2. `services/api/core/comparator.py` (raw REST v2). Used by `/api/chat` and `/api/search`.
3. `scripts/ingest.py` (Python client, write path).
