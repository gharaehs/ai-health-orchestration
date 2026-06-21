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
| Module 6 | Model Context Protocol (MCP) | 🔄 In Progress |
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
│             ▼                └──────────────┬───────────────┘   │
│  ┌──────────────────────┐                   │ (direct, 3 agents)│
│  │   FastAPI Container  │                   │                   │
│  │   Port: 8002         │                   ▼                   │
│  │                      │    ┌──────────────────────────────┐   │
│  │  RAG pipeline        │    │   MCP RAG Server (Module 6)   │   │
│  │  Multi-agent orch.   │◄──►│   Port: 8004                  │   │
│  │  Async job system    │MCP │   Wraps HealthRetriever        │   │
│  │  ChromaDB (3 agents) │    │   4 named tools + retrieve_    │   │
│  │  MCP client (Nutri.) │    │   context (collection-routed)  │   │
│  └──────────────────────┘    └──────────────────────────────┘   │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │   React/Vite Container│                                      │
│  │   Port: 3000          │                                      │
│  └──────────────────────┘                                       │
│                                                                  │
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
    ├── 2. NutritionAgent    → nutrition_guidelines + food_and_recipes (via MCP server, Module 6)
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

**MCP retrieval path (Module 6, NutritionAgent only):**

```
NutritionAgent._retrieve()  (override of BaseAgent._retrieve)
    │
    ▼
MCP Client (streamable HTTP) ──► MCP RAG Server (port 8004)
    │                                  │
    │                                  ▼
    │                          HealthRetriever.retrieve()
    │                                  │
    │                                  ▼
    │                            ChromaDB (port 8001)
    ◄──────────────────────────────────┘
result.structuredContent["result"]  →  formatted context block
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
├── docker-compose.yml              # Orchestrates all 5 services
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
│   ├── nutrition_agent.py          # Agent 2: 7-day meal plan — RAG via MCP (Module 6)
│   ├── training_agent.py           # Agent 3: weekly gym program
│   └── grocery_agent.py            # Agent 4: per-category shopping list
│
├── services/
│   ├── llm/
│   │   └── Dockerfile              # vLLM container configuration
│   ├── mcp-rag/                    # ✅ Module 6: MCP server wrapping RAG corpus
│   │   ├── Dockerfile              # CPU-only torch — no GPU needed for embeddings
│   │   └── mcp_server.py           # FastMCP server: 4 named tools + retrieve_context
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

### MCP Server — RAG Corpus (Module 6)

- **Image:** custom, built from `services/mcp-rag/Dockerfile`
- **Port:** `8004`
- **Transport:** Streamable HTTP (`http://mcp-rag:8004/mcp` on the internal Docker network)
- **Wraps:** `rag/retriever.py`'s `HealthRetriever` — no duplicated retrieval logic, same embedding model and ChromaDB connection pattern as the direct path
- **Tools exposed:**
  - `retrieve_context(query_text, query_type, n_results_per_collection, max_distance)` — mirrors `HealthRetriever.retrieve()`'s collection routing exactly; this is what `NutritionAgent` calls
  - `search_nutrition_guidelines(query, top_k)`
  - `search_food_and_recipes(query, top_k)`
  - `search_gym_programming(query, top_k)`
  - `search_public_health_recommendations(query, top_k)`
  - `corpus_health_check()` — returns document counts per collection, for debugging
- **Consumer:** `NutritionAgent` only (scoped deliberately — see [Design Decisions](#design-decisions)). `LabAnalysisAgent`, `TrainingAgent`, and `GroceryAgent` still call `HealthRetriever` directly via `BaseAgent._retrieve()`.
- **Dependency note:** uses CPU-only torch (`--index-url https://download.pytorch.org/whl/cpu`) since this container has no GPU reservation and only runs `sentence-transformers` embeddings on CPU — avoids ~2.5 GB of unused CUDA libraries that would otherwise be pulled in by the default GPU torch wheel.

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
8. **Node.js + npx** (for MCP Inspector, used to test the Module 6 MCP server standalone)

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
make start    # Build and start all services
make logs     # Watch startup logs
make stop     # Stop all services
```

Access: `http://<your-ec2-public-ip>:3000`

**MCP server standalone test** (before wiring into agents, or for debugging):
```bash
docker compose up -d mcp-rag
npx @modelcontextprotocol/inspector
# Connect: Transport Type = Streamable HTTP, URL = http://<ec2-ip>:8004/mcp
# Test corpus_health_check first, then retrieve_context with a real query_type
```

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

### Verifying MCP retrieval is actually being used (not silently falling back)

Compare `prompt_len` in the api logs for NutritionAgent's first LLM call:
- **~19,000+ chars** → MCP retrieval succeeded and context was injected
- **~1,700–1,800 chars** → MCP retrieval failed and `_retrieve()` silently fell back to `""` (check for `[NutritionAgent] MCP retrieval error` in logs)

```bash
docker compose logs api | grep -i "nutritionagent\|mcp"
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make start` | Build and start all services |
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

**Note:** the EC2 instance's public IP changes on stop/start unless an Elastic IP is attached. If services become unreachable after a restart, check the current IP first (`curl -s http://169.254.169.254/latest/meta-data/public-ipv4`) before debugging anything else.

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
If the failing step is installing `torch`, check whether the container actually needs GPU support. CPU-only containers (e.g. `mcp-rag`) should install torch via `--index-url https://download.pytorch.org/whl/cpu` — the default GPU wheel pulls in ~2.5 GB of unused `nvidia-cuda-*` packages.

### vLLM uses FlashInfer instead of FlashAttention2
Not an error. T4 (compute 7.5) does not support FlashAttention2 (requires 8.0+).

### ChromaDB v2 UUID error
Already handled — `comparator.py` caches name→UUID mapping. Restart api container to clear cache.

### Module 6: agent's `_retrieve()` override not taking effect
If logs show `agents.base` logging the RAG query instead of `agents.nutrition_agent`, the override method isn't actually present in the subclass (commonly: it accidentally ended up in `agents/base.py` instead of `agents/nutrition_agent.py`, or was added at module level instead of inside the class body). Python resolves `self._retrieve()` via the instance's actual class — if the subclass doesn't define it, the inherited `BaseAgent` version silently runs instead with no error.

### Module 6: MCP tool result parsing — `KeyError` or `TypeError: string indices must be integers`
A tool returning `list[dict]` is exposed via `result.structuredContent`, wrapped as `{"result": [...]}` — **not** via `result.content[0].text`, which behaves differently depending on SDK version. Use:
```python
result = await session.call_tool("retrieve_context", {...})
return result.structuredContent["result"]
```

### Module 6: `asyncio.run()` inside a sync agent method
Safe in this architecture specifically because `Orchestrator.run()` and `_run_agent()` are fully synchronous with no `async`/`await` anywhere — the background thread running the pipeline never has its own event loop already running. If you ever make the Orchestrator itself async, this would need to change to `await` directly instead.

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

### 🔄 Module 6 — Model Context Protocol (In Progress)

**Goal:** expose health data, workout history, and the recipe database via MCP servers, and convert at least one agent's retrieval path to use the protocol instead of direct Python calls — demonstrating MCP's interoperability story without over-engineering a system that already works end-to-end.

**Completed: RAG corpus exposed as an MCP server, consumed by NutritionAgent**

- `services/mcp-rag/mcp_server.py` — `FastMCP` server wrapping the existing `rag/retriever.py` `HealthRetriever` with zero duplicated logic. Exposes 5 tools: `retrieve_context` (mirrors `HealthRetriever.retrieve()`'s collection routing exactly — the one agents actually use) plus 4 collection-specific `search_*` tools for finer-grained/external use, and `corpus_health_check` for debugging
- Runs as its own container (`mcp-rag`, port 8004) on Streamable HTTP transport, verified standalone via the MCP Inspector before any agent code was touched
- `agents/nutrition_agent.py` — `_retrieve()` overridden (subclass-only, not touching `BaseAgent`) to call the MCP server via `mcp.client.session.ClientSession` + `streamablehttp_client` instead of calling `HealthRetriever` directly; output format matches the original exactly, so all downstream prompt-building code is unchanged
- Verified end-to-end in production: full pipeline run confirmed via log evidence (`prompt_len` for NutritionAgent's first call back to ~19,100 chars, matching the pre-MCP grounded baseline, vs. ~1,800 chars during the silent-fallback failure runs) plus direct MCP session logs showing the full JSON-RPC lifecycle (initialize → negotiate protocol `2025-06-18` → tool call → session delete)

**Scoping decision:** only `NutritionAgent`'s retrieval was converted. `LabAnalysisAgent`, `TrainingAgent`, and `GroceryAgent` still call `HealthRetriever` directly via `BaseAgent._retrieve()`. This is deliberate — see [Design Decisions](#design-decisions).

**Still open / planned:**
- Calendar integration MCP server (push generated meal plans and gym programs to Google Calendar) — highest priority next piece, since it's a genuine third-party boundary where MCP's value is clearest
- EC2 Security Group lockdown before any demo (currently several ports including 8004 are open to `0.0.0.0/0`)
- Optional: exposing the `mcp-rag` server to external MCP clients (e.g. Claude Desktop via a custom connector) — would require a public HTTPS endpoint with a real domain/TLS cert, since custom connectors are reached from the client's cloud infrastructure rather than directly from the local network; not pursued since it's orthogonal to the module's core grading criteria

**Key learnings:**
- **MCP earns its place only at genuine external boundaries.** Wrapping agent-to-orchestrator calls in MCP (same codebase, same trust boundary) would add network overhead and operational complexity for zero new flexibility — that decoupling already exists via the `BaseAgent` abstract-class pattern. MCP's value is specifically standardizing the wire format between parties that *don't* share a codebase (a RAG corpus as a reusable service, a third-party calendar API, an external AI client) — not a substitute for normal OOP polymorphism within one's own app.
- **Build context vs. Dockerfile location mismatch.** With `docker-compose.yml`'s build `context: .` (repo root), `COPY` paths in a Dockerfile under `services/mcp-rag/Dockerfile` must still be written relative to the repo root, not relative to the Dockerfile's own directory — caused an early build failure (`requirements.txt: not found`) until corrected.
- **CPU-only torch avoids multi-GB of wasted disk.** `torch==2.5.1` with no index qualifier installs the full CUDA build (~2.5 GB of `nvidia-cuda-*` wheels) even in containers with no GPU reservation. Installing via `--index-url https://download.pytorch.org/whl/cpu` first, then the rest of `requirements.txt` with torch filtered out, fixed repeated `No space left on device` build failures — with zero effect on embedding model accuracy, since the computation was always CPU-bound regardless of which torch build was installed.
- **MCP tool results: `structuredContent`, not `content[0].text`.** A tool returning `list[dict]` is exposed via `result.structuredContent`, wrapped as `{"result": [...]}`. Assuming the unstructured `content[0].text` field held the same shape produced two different wrong guesses (a `TypeError` from iterating dict keys, then a `KeyError` from the wrong field) before checking the actual attribute.
- **A broken `BaseAgent` takes down the entire app, not just one agent.** Pasting new code into the wrong file (`agents/base.py` instead of `agents/nutrition_agent.py`) caused an `IndentationError` that every agent's import chain depends on — manifesting as a full container crash and HTTP 502 on the frontend, several layers away from the actual mistake.
- **Silent fallback paths hide failures convincingly.** `_retrieve()`'s `try/except` returning `""` on any error meant the pipeline kept reporting `4/4 agents succeeded` even on a run where MCP retrieval silently failed — the only tell was a sharp drop in `prompt_len` for that agent's LLM call. Worth comparing against a known-good baseline number whenever a fallback path exists.
- **Default Python logging silently drops `INFO` messages with no handler configured.** None of `BaseAgent`'s, `Orchestrator`'s, or any agent's `logger.info()` calls were visible in `docker compose logs api` until `logging.basicConfig(level=logging.INFO)` was added to the app entrypoint — a pre-existing gap, unrelated to Module 6, that had been hiding all agent-level logging the whole time.

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

### Why expose the RAG corpus as an MCP server, but not the agents themselves?

The RAG corpus is conceptually a reusable knowledge service — exposing it via MCP means any future MCP-compliant consumer (not just this Orchestrator) could query it without custom integration code, and it cleanly demonstrates the protocol's interoperability promise. Agents calling each other within the same codebase don't have that problem: `BaseAgent`'s abstract-class pattern already gives full polymorphism and swappability in-process, for free, with no network hop. Converting agent-to-agent or agent-to-orchestrator calls to MCP would add latency and operational surface (separate containers, network failure handling) without buying any new flexibility — exactly the kind of over-engineering this project avoids elsewhere.

### Why scope the MCP conversion to NutritionAgent only?

`BaseAgent._retrieve()` is shared by all four agents. Editing it directly would convert every agent to MCP at once — more than needed to demonstrate the architecture, and a bigger blast radius for Module 6's actual grading criteria. Overriding `_retrieve()` specifically inside `NutritionAgent` (rather than modifying the shared base class) proves the pattern works end-to-end while leaving `LabAnalysisAgent`, `TrainingAgent`, and `GroceryAgent` — and the well-tested `BaseAgent` they share — completely untouched.

---

## Links

- **GitHub Repository:** https://github.com/gharaehs/ai-health-orchestration
- **Mentorship Program:** AI Technical Deep Dive
- **Model:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **vLLM Documentation:** https://docs.vllm.ai
- **ChromaDB Documentation:** https://docs.trychroma.com
- **Model Context Protocol:** https://modelcontextprotocol.io

---

*Last updated: Modules 1–5 complete · Module 6 in progress (RAG-via-MCP for NutritionAgent verified end-to-end; calendar integration planned) · Frontend dashboard complete · June 2026*