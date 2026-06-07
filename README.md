# AI Health Orchestration System

> A modular, production-style AI system that ingests structured health data and user goals to generate grounded, structured outputs including weekly meal plans, grocery lists, and gym programs.

Built as part of the **AI Technical Deep Dive** mentorship program.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Infrastructure & Hardware](#infrastructure--hardware)
- [Repository Structure](#repository-structure)
- [Services](#services)
- [Frontend Dashboard](#frontend-dashboard)
- [Prerequisites](#prerequisites)
- [Fresh Server Setup Guide](#fresh-server-setup-guide)
- [Model Download](#model-download)
- [RAG Corpus Ingestion](#rag-corpus-ingestion)
- [Running the System](#running-the-system)
- [Testing & Verification](#testing--verification)
- [Makefile Reference](#makefile-reference)
- [Migrating to a New Server](#migrating-to-a-new-server)
- [Troubleshooting](#troubleshooting)
- [Module Progress](#module-progress)
- [Design Decisions](#design-decisions)

---

## Project Overview

This system implements a multi-module AI pipeline across 6 engineering modules, plus a professional frontend dashboard:

| Module | Topic | Status |
|--------|-------|--------|
| Module 1 | Local LLM Deployment & Quantization | ✅ Complete |
| Module 2 | Parameter-Efficient Fine-Tuning (LoRA) | ✅ Complete |
| Module 3 | Vector Database & RAG Corpus Ingestion | ✅ Complete |
| Module 4 | Retrieval-Augmented Generation (RAG) | ✅ Complete |
| Module 5 | Agentic AI / Multi-Agent Orchestration | ✅ Complete |
| Module 6 | Model Context Protocol (MCP) | 🔜 Planned |
| Frontend | Analytics Dashboard & Orchestration UI | ✅ Complete |

**Core capabilities:**
- Ingest structured health data (scale metrics, blood tests, medical history)
- Accept user-defined health and fitness goals
- Generate weekly meal plans with calories and macros
- Generate structured gym programs (sets, reps, progression)
- Generate aggregated grocery lists
- Produce structured JSON outputs suitable for automation pipelines

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EC2 g4dn.xlarge                          │
│                     Ubuntu 22.04 + CUDA 12.2                    │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │   vLLM Container     │    │    ChromaDB Container         │   │
│  │   Port: 8000         │    │    Port: 8001 (host)          │   │
│  │                      │    │    Port: 8000 (internal)      │   │
│  │  Llama 3.1 8B        │    │  4 Collections (13,349 docs)  │   │
│  │  Instruct (fp16)     │    │  - public_health_recs         │   │
│  │  GPU: Tesla T4       │    │  - nutrition_guidelines       │   │
│  └──────────┬───────────┘    │  - gym_programming            │   │
│             │                │  - food_and_recipes           │   │
│             ▼                └──────────────────────────────┘   │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │   FastAPI Container  │    │   React/Vite Container        │   │
│  │   Port: 8002         │    │   Port: 3000                  │   │
│  │                      │    │                               │   │
│  │  RAG pipeline        │    │  Chat & Compare               │   │
│  │  Multi-agent orch.   │    │  Orchestrate (Module 5)       │   │
│  │  Async job system    │    │  Analytics charts             │   │
│  │  ChromaDB retrieval  │    │  Knowledge Base search        │   │
│  └──────────────────────┘    │  Health Profile form          │   │
│                              └──────────────────────────────┘   │
│    ./models/llama/       (mounted volume, not in git)           │
│    ./models/adapters/    (LoRA adapters, not in git)            │
│    ./data/chroma/        (mounted volume, persisted to disk)    │
│    ./data/corpus/        (corpus files, not in git)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Multi-agent pipeline (Module 5):**

```
HealthProfile + UserGoals
    │
    ▼
Orchestrator
    │
    ├── 1. LabAnalysisAgent  → public_health_recommendations
    │       interprets blood markers against clinical reference ranges
    │       outputs: dietary_constraints, training_constraints, TDEE, target_calories
    │
    ├── 2. NutritionAgent    → nutrition_guidelines + food_and_recipes
    │       generates 7-day meal plan (split: Mon–Thu + Fri–Sun)
    │       consumes: dietary_constraints, recommended_calories from Lab Analysis
    │
    ├── 3. TrainingAgent     → gym_programming
    │       generates weekly gym program (sets, reps, progression)
    │       consumes: training_constraints from Lab Analysis
    │
    └── 4. GroceryAgent      → food_and_recipes
            aggregates ingredients across 7 days into shopping list
            uses: per-category LLM consolidation (fuzzy deduplication)
            consumes: weekly_plan from Nutrition Agent
```

**RAG pipeline (Module 4):**

```
User Query (health profile + goal)
    │
    ▼
HealthRAGPipeline (rag/pipeline.py)
    │
    ├── 1. Embed query → sentence-transformers/all-MiniLM-L6-v2 (CPU)
    ├── 2. Retrieve → ChromaDB (port 8001) — collection routed by query type
    ├── 3. Build augmented prompt → STRUCTURED strategy (labelled context blocks)
    └── 4. Generate → vLLM API (port 8000) → Structured JSON response
```

---

## Infrastructure & Hardware

| Component | Specification |
|-----------|--------------|
| **Instance** | AWS EC2 g4dn.xlarge |
| **GPU** | NVIDIA Tesla T4 — 16 GB VRAM |
| **CPU** | 4 vCPUs |
| **RAM** | 16 GB |
| **Storage** | 80 GB SSD (NVMe) |
| **OS** | Ubuntu 22.04 LTS |
| **CUDA** | 12.2 |
| **Driver** | 535.288.01 |

**GPU Memory Usage (Llama 3.1 8B fp16 via vLLM):**

| Component | VRAM Used |
|-----------|-----------|
| Llama 3.1 8B (bitsandbytes 4-bit) | ~5.65 GB |
| KV Cache (PagedAttention) | ~6.8 GB |
| System overhead | ~0.5 GB |
| Safety buffer (15%) | ~2.1 GB |
| **Total available** | **14.58 GB** |

---

## Repository Structure

```
ai-health-orchestration/
│
├── docker-compose.yml              # Orchestrates all 4 services
├── Makefile                        # Command interface (start, stop, test, etc.)
├── .gitignore
├── README.md
│
├── agents/                         # ✅ Module 5: Multi-agent orchestration
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic I/O contracts for all agents
│   ├── base.py                     # Abstract BaseAgent: RAG, LLM, retry, JSON parsing
│   ├── orchestrator.py             # Pipeline coordinator — sequential execution
│   ├── lab_analysis_agent.py       # Agent 1: blood marker interpretation
│   ├── nutrition_agent.py          # Agent 2: 7-day meal plan (split generation)
│   ├── training_agent.py           # Agent 3: weekly gym program
│   └── grocery_agent.py            # Agent 4: per-category shopping list
│
├── services/
│   ├── llm/
│   │   └── Dockerfile              # vLLM container configuration
│   ├── api/                        # ✅ FastAPI middleware
│   │   ├── Dockerfile              # Copies agents/ and rag/ from project root
│   │   ├── main.py                 # FastAPI app — registers all routers
│   │   ├── routes/
│   │   │   ├── chat.py             # POST /api/chat (parallel RAG + base)
│   │   │   ├── search.py           # POST /api/search (Knowledge Base)
│   │   │   └── orchestrate.py      # POST /api/orchestrate (async job system)
│   │   ├── core/
│   │   │   ├── comparator.py       # asyncio.gather() RAG vs base runner
│   │   │   ├── metrics.py          # Grounding score, token count helpers
│   │   │   └── config.py           # Service URLs, timeouts
│   │   └── requirements.txt
│   └── frontend/                   # ✅ React/Vite dashboard
│       ├── Dockerfile
│       ├── package.json
│       ├── vite.config.ts
│       └── src/
│           ├── App.tsx             # Full dashboard — 6 views including Orchestrate
│           ├── useChat.ts
│           ├── types.ts
│           ├── main.tsx
│           └── index.css
│
├── scripts/
│   ├── download_model.sh           # Downloads Llama 3.1 8B from HuggingFace
│   ├── ingest.py                   # Module 3: RAG corpus ingestion pipeline
│   ├── test_rag.py                 # Module 4: RAG pipeline CLI test tool
│   └── test_orchestration.py       # Module 5: full pipeline CLI test tool
│
├── rag/                            # Module 4: RAG pipeline
│   ├── __init__.py
│   ├── retriever.py                # ChromaDB query logic + collection routing
│   ├── prompt_builder.py           # Prompt assembly (STRUCTURED / NAIVE strategies)
│   ├── pipeline.py                 # Main RAG orchestration class
│   └── evaluate.py                 # RAG vs no-RAG evaluation suite
│
├── training/
│   ├── train.py                    # QLoRA fine-tuning script
│   ├── evaluate.py                 # Base vs fine-tuned comparison
│   ├── config.yaml                 # Training hyperparameters
│   ├── evaluation_results.json
│   └── dataset/
│       └── processed/
│           └── health_training_data.jsonl
│
├── models/                         # ⚠️ NOT IN GIT
│   ├── llama/                      # Llama 3.1 8B Instruct fp16 (~15 GB)
│   └── adapters/
│       └── health-v1/              # LoRA adapter from Module 2 (~161 MB)
│
└── data/
    ├── chroma/                     # ⚠️ NOT IN GIT — vector DB persistent storage
    └── corpus/                     # ⚠️ NOT IN GIT — raw corpus files
```

---

## Services

### LLM Service — vLLM + Llama 3.1 8B Instruct

- **Image:** `vllm/vllm-openai:latest`
- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (fp16, on-the-fly 4-bit via BitsAndBytes)
- **Port:** `8000`
- **API:** OpenAI-compatible (`/v1/chat/completions`, `/v1/models`)
- **Runtime flags:** `--dtype float16` · `--max-model-len 16384` · `--gpu-memory-utilization 0.85` · `--enforce-eager` · `--quantization bitsandbytes`

### Vector Database — ChromaDB

- **Image:** `chromadb/chroma:latest` (v1.5.8)
- **Port:** `8001` (host) → `8000` (internal)
- **Storage:** `./data/chroma:/data`
- **Collections:** 4 domain-specific collections (13,349 documents)

### API Middleware — FastAPI

- **Port:** `8002`
- **Endpoints:**
  - `POST /api/chat` — parallel RAG + base via `asyncio.gather()`
  - `POST /api/search` — ChromaDB vector search
  - `POST /api/orchestrate` — start async pipeline job, returns `{job_id}`
  - `GET /api/orchestrate/status/{job_id}` — poll job status
  - `GET /api/orchestrate/result/{job_id}` — fetch completed result
  - `GET /api/health` — liveness check

### Frontend Dashboard — React + Vite

- **Port:** `3000`
- **Views:** Chat & Compare · Orchestrate · Analytics · Knowledge Base · Health Profile · Config

---

## Frontend Dashboard

### Orchestrate Tab (Module 5)

The Orchestrate tab runs the full 4-agent pipeline from the browser:

1. Browser POSTs to `/api/orchestrate` → receives `{job_id}` immediately
2. Browser polls `/api/orchestrate/status/{job_id}` every 10 seconds
3. When status is `"complete"`, browser fetches `/api/orchestrate/result/{job_id}`
4. Results display in collapsible panels per agent

This async job pattern avoids long-held HTTP connections through the Vite proxy, which cannot hold connections open for 7+ minutes.

**Agent output panels:**
- **🔬 Lab Analysis** — TDEE, caloric target, dietary constraints, training constraints, per-marker interpretation with status and reference ranges
- **🥗 Nutrition Plan** — 7-day meal plan with daily macro totals and per-meal breakdown
- **🏋️ Training Program** — weekly sessions with exercises, sets/reps, and 4-week progression scheme
- **🛒 Grocery List** — consolidated shopping list by category with quantities

### API Contract (Orchestration)

#### `POST /api/orchestrate`
```json
{
  "profile": {
    "age": 38, "sex": "male", "height_cm": 178.0,
    "scale_metrics": { "weight_kg": 88.5, "body_fat_pct": 22.0 },
    "blood_markers": [
      { "name": "LDL Cholesterol", "value": 3.8, "unit": "mmol/L" }
    ],
    "medical_history": [], "medications": [], "allergies": []
  },
  "goals": {
    "primary_goal": "fat_loss",
    "activity_level": "moderately_active",
    "fitness_level": "intermediate",
    "training_days_per_week": 4,
    "dietary_preferences": []
  }
}
```

Returns: `{ "job_id": "a1b2c3d4", "status": "pending" }`

#### `GET /api/orchestrate/status/{job_id}`
Returns: `{ "job_id": "...", "status": "running|complete|failed", "duration_s": null }`

#### `GET /api/orchestrate/result/{job_id}`
Returns: Full `OrchestrationResult` with `lab_analysis`, `nutrition`, `training`, `grocery`, `agent_results`, `total_duration_s`.

---

## Prerequisites

1. **Ubuntu 22.04 LTS**
2. **NVIDIA GPU drivers** (535+) and **CUDA 12.2**
3. **Docker** + **Docker Compose plugin**
4. **NVIDIA Container Toolkit**
5. **Git**
6. **Python 3.12 + python3.12-venv**
7. **HuggingFace account** with Meta Llama 3.1 license accepted

---

## Fresh Server Setup Guide

### Step 1 — Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2 — Verify GPU
```bash
nvidia-smi
```

### Step 3 — Install Docker
```bash
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

### Step 4 — Install NVIDIA Container Toolkit
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 5 — Git + GitHub SSH
```bash
sudo apt install -y git
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
ssh-keygen -t ed25519 -C "your@email.com"
cat ~/.ssh/id_ed25519.pub   # Add to GitHub → Settings → SSH keys
```

### Step 6 — Python venv
```bash
sudo apt install -y python3.12-venv python-is-python3
```

### Step 7 — Clone
```bash
git clone git@github.com:gharaehs/ai-health-orchestration.git
cd ai-health-orchestration
```

---

## Model Download

```bash
export HF_TOKEN="hf_your_token_here"
bash scripts/download_model.sh
```

Downloads `meta-llama/Meta-Llama-3.1-8B-Instruct` (~15 GB) to `./models/llama/`.

---

## RAG Corpus Ingestion

```bash
python3 -m venv ~/.venv-rag
source ~/.venv-rag/bin/activate
pip install chromadb sentence-transformers pymupdf pandas tiktoken tqdm python-docx
make start   # ChromaDB must be running
python3 scripts/ingest.py
```

| Collection | Documents | Content |
|------------|-----------|---------|
| `public_health_recommendations` | 4,246 | Lab reference ranges, clinical guidelines |
| `nutrition_guidelines` | 881 | Research papers, DRI tables |
| `gym_programming` | 416 | ACSM/NSCA position stands |
| `food_and_recipes` | 7,806 | Recipes + USDA nutrient data |
| **Total** | **13,349** | |

---

## Running the System

```bash
make start    # Build and start all 4 services
make logs     # Watch startup logs
make stop     # Stop all services
```

Access: `http://<your-ec2-public-ip>:3000`

---

## Testing & Verification

### Health check
```bash
curl http://localhost:8002/api/health | python3 -m json.tool
```

### RAG pipeline
```bash
source ~/.venv-rag/bin/activate
python scripts/test_rag.py --health-check
python scripts/test_rag.py --type lab_analysis --compare
```

### Full orchestration pipeline (CLI)
```bash
source ~/.venv-rag/bin/activate
python scripts/test_orchestration.py
python scripts/test_orchestration.py --minimal   # No blood markers
python scripts/test_orchestration.py --output result.json
```

### Orchestration API
```bash
# Start a job
JOB=$(curl -s -X POST http://localhost:8002/api/orchestrate \
  -H "Content-Type: application/json" \
  -d "$(curl -s http://localhost:8002/api/orchestrate/sample-request)" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# Poll status
curl -s http://localhost:8002/api/orchestrate/status/$JOB | python3 -m json.tool

# Fetch result when complete
curl -s http://localhost:8002/api/orchestrate/result/$JOB | python3 -m json.tool | head -40
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make start` | Build and start all 4 services |
| `make stop` | Stop and remove all containers |
| `make logs` | Follow live logs |
| `make download-model` | Download Llama 3.1 8B |
| `make test-llm` | Send test prompt to vLLM |
| `make build-frontend` | Rebuild api and frontend images |

---

## Migrating to a New Server

| Data | In Git? | Migrate how? |
|------|---------|--------------|
| Code & config | ✅ Yes | `git clone` |
| Llama 3.1 model | ❌ No | `bash scripts/download_model.sh` |
| LoRA adapter | ❌ No | Re-train or `scp` |
| Vector DB data | ❌ No | `scp ./data/chroma/` or re-run `ingest.py` |
| Corpus files | ❌ No | `scp ./data/corpus/` |

---

## Troubleshooting

### vLLM OOM on startup
Ensure `--enforce-eager` is in `docker-compose.yml` command. T4 cannot run CUDA graph warmup.

### ChromaDB collections empty after restart
Volume must mount `./data/chroma:/data`. Re-run `ingest.py` if data is lost.

### API 502 on orchestrate endpoint
Check `docker compose logs api` — likely an import error. Rebuild: `docker compose build api && docker compose up -d api`.

### Orchestration pipeline times out in browser
The async job pattern (POST → poll status → fetch result) avoids proxy timeouts. All requests are short. If polling stops, check `docker compose logs api` for the job status.

### Docker build fails — no space left on device
```bash
docker builder prune -af
rm -rf ~/.cache/pip ~/.cache/JetBrains
```

### vLLM uses FlashInfer instead of FlashAttention2
Not an error. T4 (compute 7.5) does not support FlashAttention2 (requires 8.0+).

### ChromaDB v2 UUID error
Already handled — `comparator.py` caches name→UUID mapping. Restart api container to clear cache.

---

## Module Progress

### ✅ Module 1 — Local LLM Deployment

- vLLM serving Llama 3.1 8B via OpenAI-compatible API on port 8000
- `--enforce-eager` resolves T4 CUDA graph OOM

**Key learnings:** 1B params ≈ 2 GB VRAM · T4 uses FlashInfer · CUDA graph warmup is the OOM culprit

---

### ✅ Module 2 — Parameter-Efficient Fine-Tuning / LoRA

- QLoRA: Llama 3.1 8B in 4-bit via BitsAndBytes, LoRA rank=16 alpha=32
- 21-example health domain dataset, LoRA adapter: `./models/adapters/health-v1/` (~161 MB)

**Key learnings:** AWQ is inference-only · BitsAndBytes NF4 is the correct QLoRA path

---

### ✅ Module 3 — Vector Database & RAG Corpus Ingestion

- 4 ChromaDB collections, 13,349 documents
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` on CPU

**Key learnings:** ChromaDB 1.5.8 uses v2 API · volume must mount to `/data`

---

### ✅ Module 4 — Retrieval-Augmented Generation

- End-to-end RAG: embed → retrieve → augmented prompt → structured JSON
- Key finding: RAG returns 4-tier LDL classification from corpus; base model returns simplified flat range

| Metric | With RAG | No RAG |
|--------|----------|--------|
| Latency | 58.8s | 37.1s |
| Prompt tokens | 2,289 | 417 |
| Chunks retrieved | 6 | 0 |

---

### ✅ Module 5 — Agentic AI / Multi-Agent Orchestration

**Completed:**

- `agents/schemas.py` — Pydantic I/O contracts for all agents (HealthProfile, UserGoals, LabAnalysisOutput, NutritionOutput, TrainingOutput, GroceryOutput, OrchestrationResult)
- `agents/base.py` — abstract BaseAgent with shared RAG retrieval, LLM calling, JSON extraction, retry logic
- `agents/lab_analysis_agent.py` — interprets blood markers against clinical guidelines from ChromaDB; calculates TDEE via Mifflin-St Jeor; outputs dietary and training constraints
- `agents/nutrition_agent.py` — 7-day meal plan generated in two half-week LLM calls (Mon–Thu, Fri–Sun) to fit within 2048-token output budget; merged and validated as NutritionOutput
- `agents/training_agent.py` — weekly gym program with exercises, sets/reps, rest periods, and 4-week progression scheme grounded in ACSM/NSCA guidelines
- `agents/grocery_agent.py` — 85 raw ingredients pre-bucketed by keyword into 8 categories; one LLM call per category for fuzzy deduplication and quantity consolidation; final shopping notes via one small LLM call
- `agents/orchestrator.py` — sequential pipeline coordinator; agents share one ChromaDB connection; each agent's output passed as context to downstream agents; never raises on agent failure
- `scripts/test_orchestration.py` — CLI test tool with full and minimal profiles, `--output` flag for JSON export
- `services/api/routes/orchestrate.py` — async job system: `POST /api/orchestrate` returns `job_id` immediately; pipeline runs in `ThreadPoolExecutor`; browser polls `/status/{job_id}` every 10s; result fetched from `/result/{job_id}`
- `services/frontend/src/App.tsx` — Orchestrate tab with 4-agent progress cards, collapsible result panels per agent, poll-based UI that avoids proxy timeout issues

**Pipeline performance (full profile, 7 blood markers, fat_loss goal):**

| Agent | Collections | Duration |
|-------|-------------|----------|
| LabAnalysisAgent | public_health_recommendations | ~62s |
| NutritionAgent | nutrition_guidelines + food_and_recipes | ~200s (2 LLM calls) |
| TrainingAgent | gym_programming | ~77s |
| GroceryAgent | food_and_recipes (per-category) | ~90s (8 LLM calls) |
| **Total** | | **~430s** |

**Key design decisions:**

- **Sequential with data passing:** Lab Analysis output (dietary/training constraints, caloric target) flows into Nutrition and Training as hard constraints — not parallel, because downstream agents depend on upstream outputs
- **Split generation for Nutrition:** 7-day JSON exceeds 2048-token output limit; split into Mon–Thu and Fri–Sun calls, merged in Python
- **Per-category consolidation for Grocery:** One LLM call per category (Proteins, Vegetables, etc.) with fuzzy deduplication; Python pre-bucketing reduces each call to 10–20 items, avoiding truncation
- **Async job pattern for API:** Browser POST returns `job_id` in <1s; pipeline runs server-side; browser polls every 10s — eliminates Vite proxy connection timeout problem for 7-minute requests
- **Shared ChromaDB connection:** Orchestrator initialises one `HealthRetriever` shared across all agents — avoids 4× embedding model reload overhead

**Key learnings:**
- LLM enums must match exactly — Llama returns `"borderline-high"` not `"borderline"`; MarkerStatus enum extended to cover all observed values
- Nutrition JSON (~13,000 chars) reliably truncates at 2048 tokens — split generation is the correct fix
- Grocery: keyword pre-bucketing + per-category LLM consolidation handles fuzzy ingredient matching that pure Python string matching cannot
- Vite proxy drops connections after ~2 min regardless of `timeout: 0` — async job polling is the production-correct pattern
- Docker build context must be project root (not `services/api/`) to COPY `agents/` and `rag/` into the API container

---

### 🔜 Module 6 — Model Context Protocol (Planned)

Expose health data, workout history, and recipe database via MCP servers.

---

### ✅ Frontend Dashboard

**6 views:**
- **Chat & Compare** — query input, side-by-side base vs RAG, sources panel, grounding metrics
- **Orchestrate** — full 4-agent pipeline with async polling, per-agent progress cards, collapsible result panels
- **Analytics** — session latency trend and grounding score history
- **Knowledge Base** — direct ChromaDB vector search UI
- **Health Profile** — body metrics, blood markers, goal, notes — used by both Chat and Orchestrate
- **Config** — top-k parameter, system info

**Remaining:**
- SSE token streaming (future enhancement)
- EC2 Security Group lockdown before demo

---

## Design Decisions

### Why sequential agents over parallel?

Lab Analysis outputs `dietary_constraints` and `training_constraints` that Nutrition and Training must respect. Nutrition outputs the `weekly_plan` that Grocery aggregates. The dependency graph is linear — parallelism would require a merge/reconcile step that adds complexity without benefit on a single T4.

### Why split Nutrition generation into two calls?

A 7-day × 3-meal plan with ingredients, instructions, and macros consistently exceeds 2048 tokens. Splitting into Mon–Thu and Fri–Sun generates two valid JSON fragments that Python merges — simpler and more reliable than raising the token limit (which would slow generation significantly on T4).

### Why per-category LLM consolidation for Grocery?

Simple string matching fails on `"150g chicken breast"` vs `"chicken breast, diced, 200g"`. The LLM handles fuzzy normalisation naturally. Pre-bucketing by keyword (Python, instant) limits each LLM call to 10–20 items — well within 2048 tokens and never truncates.

### Why async job polling instead of a long HTTP request?

Vite's dev server proxy drops connections after ~2 minutes regardless of timeout configuration. A 7-minute pipeline request will always fail through the proxy. The async pattern (POST → job_id → poll status → fetch result) makes every individual HTTP call short, avoiding the proxy entirely as a bottleneck.

### Why Llama 3.1 8B over Mistral 7B?

128K context window vs 4K — essential for RAG with multiple retrieved chunks. Single model for both training (BitsAndBytes QLoRA) and inference (vLLM). AWQ-quantized models like Mistral 7B AWQ cannot be used for LoRA training.

### Why four separate ChromaDB collections?

Collection-level isolation enforces domain separation cleanly — a Lab Analysis Agent queries only clinical guidelines, never recipes. This removes the need for metadata filtering and keeps retrieval fast and precise.

### Why CPU for embeddings?

Preserves the full 16 GB T4 VRAM for Llama 3.1 inference. `all-MiniLM-L6-v2` on CPU runs at ~0.3s per query — negligible for this workload.

---

## Links

- **GitHub Repository:** https://github.com/gharaehs/ai-health-orchestration
- **Mentorship Program:** AI Technical Deep Dive
- **Model:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **vLLM Documentation:** https://docs.vllm.ai
- **ChromaDB Documentation:** https://docs.trychroma.com

---

*Last updated: Modules 1–5 complete · Frontend dashboard complete · June 2026*