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
| Module 5 | Agentic AI / Multi-Agent Orchestration | 🔜 Next |
| Module 6 | Model Context Protocol (MCP) | 🔜 Planned |
| Frontend | Analytics Dashboard & Chat Interface | ✅ Complete (Steps 1–9 of 10) |

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
│  │  Parallel RAG+base   │    │  Chat & Compare               │   │
│  │  ChromaDB retrieval  │    │  Analytics charts             │   │
│  │  Grounding metrics   │    │  Knowledge Base search        │   │
│  └──────────────────────┘    │  Health Profile form          │   │
│                              └──────────────────────────────┘   │
│    ./models/llama/       (mounted volume, not in git)           │
│    ./models/adapters/    (LoRA adapters, not in git)            │
│    ./data/chroma/        (mounted volume, persisted to disk)    │
│    ./data/corpus/        (corpus files, not in git)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow (Module 4 RAG pipeline):**

```
User Query (health profile + goal)
    │
    ▼
HealthRAGPipeline (rag/pipeline.py)
    │
    ├── 1. Embed query → sentence-transformers/all-MiniLM-L6-v2 (CPU)
    │
    ├── 2. Retrieve → ChromaDB (port 8001)
    │       ├── nutrition_guidelines      ← Macros, DRIs, dietary patterns
    │       ├── food_and_recipes          ← Recipes + USDA nutrient data
    │       ├── gym_programming           ← Rep ranges, periodization
    │       └── public_health_recs        ← Lab markers, clinical thresholds
    │
    ├── 3. Build augmented prompt → HealthPromptBuilder (STRUCTURED strategy)
    │       └── [CONTEXT: <collection label>] + retrieved chunks + user request
    │
    └── 4. Generate → vLLM API (port 8000)
            └── Structured JSON response (meal plan / gym program / lab analysis)
```

**Data flow (Frontend API middleware):**

```
Browser (port 3000)
    │  POST /api/chat
    ▼
FastAPI Middleware (port 8002)
    │
    ├── asyncio.gather()
    │       ├── Base call → vLLM (no context)
    │       └── Embed query → ChromaDB retrieval → augmented prompt → vLLM
    │
    └── Return: base_response + rag_response + sources + grounding metrics
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
├── services/
│   ├── llm/
│   │   └── Dockerfile              # vLLM container configuration
│   ├── api/                        # ✅ FastAPI middleware
│   │   ├── Dockerfile
│   │   ├── main.py                 # FastAPI app + /api/health + /api/search
│   │   ├── routes/
│   │   │   ├── chat.py             # POST /api/chat (parallel RAG + base calls)
│   │   │   └── search.py           # POST /api/search (Knowledge Base explorer)
│   │   ├── core/
│   │   │   ├── comparator.py       # asyncio.gather() RAG vs base runner
│   │   │   ├── metrics.py          # Grounding score, token count helpers
│   │   │   └── config.py           # Service URLs, timeouts
│   │   └── requirements.txt
│   └── frontend/                   # ✅ React/Vite dashboard
│       ├── Dockerfile
│       ├── package.json
│       ├── vite.config.ts          # Vite config + proxy to api:8002
│       └── src/
│           ├── App.tsx             # Full dashboard — all 5 views in one file
│           ├── useChat.ts          # Fetch hook with auto query-type detection
│           ├── types.ts            # TypeScript interfaces
│           ├── main.tsx
│           └── index.css
│
├── scripts/
│   ├── download_model.sh           # Downloads Llama 3.1 8B from HuggingFace
│   ├── ingest.py                   # Module 3: RAG corpus ingestion pipeline
│   └── test_rag.py                 # Module 4: RAG pipeline CLI test tool
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
│   ├── evaluation_results.json     # Module 2 evaluation output
│   └── dataset/
│       └── processed/
│           └── health_training_data.jsonl   # 21 training examples
│
├── models/                         # ⚠️ NOT IN GIT — downloaded locally
│   ├── llama/                      # Llama 3.1 8B Instruct fp16 (~15 GB)
│   └── adapters/
│       └── health-v1/              # LoRA adapter from Module 2 (~161 MB)
│
└── data/
    ├── chroma/                     # ⚠️ NOT IN GIT — vector DB persistent storage
    └── corpus/                     # ⚠️ NOT IN GIT — raw corpus files for ingestion
        ├── public_health_recommendations/
        ├── nutrition_guidelines/
        ├── gym_programming/
        └── food_and_recipes/
```

---

## Services

### LLM Service — vLLM + Llama 3.1 8B Instruct

- **Image:** `vllm/vllm-openai:latest`
- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (fp16, on-the-fly 4-bit quantization via BitsAndBytes)
- **Port:** `8000`
- **API:** OpenAI-compatible (`/v1/chat/completions`, `/v1/models`, etc.)
- **Runtime flags** (set via `docker-compose.yml` command override):
  - `--dtype float16` — serves fp16 model efficiently on T4
  - `--max-model-len 16384` — 16K context window for RAG use
  - `--gpu-memory-utilization 0.85` — reserves 15% VRAM as safety buffer
  - `--max-num-seqs 8` — limits concurrent sequences
  - `--enforce-eager` — disables CUDA graph capture (required for T4, prevents OOM)
  - `--quantization bitsandbytes` + `--load-format bitsandbytes` — on-the-fly 4-bit quantization

> **Why `--enforce-eager`?** The T4 GPU (compute capability 7.5) does not support vLLM's default CUDA graph warmup which causes OOM. Slightly slower for high-throughput but correct for single-user inference.

> **Why runtime flags in `docker-compose.yml`?** The vLLM image (~15 GB) cannot be rebuilt on the T4 without exhausting the 80 GB disk. The `command` override bypasses the Dockerfile `CMD` at runtime — no rebuild needed.

### Vector Database — ChromaDB

- **Image:** `chromadb/chroma:latest`
- **Version:** 1.5.8 (v2 API)
- **Port:** `8001` (host) → `8000` (container internal)
- **Storage:** `./data/chroma/` mounted to `/data` inside the container
- **Collections:** 4 domain-specific collections (13,349 documents total)

> **Critical — internal port:** ChromaDB's internal port is `8000`, not `8001`. Inside the Docker network always use `http://vector-db:8000`.

> **Critical — volume mount:** ChromaDB writes to `/data` inside the container. The correct bind is `./data/chroma:/data`. Using `/chroma/chroma` causes silent data loss on restart.

### API Middleware — FastAPI ✅

- **Framework:** FastAPI + Uvicorn
- **Port:** `8002`
- **Key endpoints:**
  - `POST /api/chat` — parallel RAG + base calls via `asyncio.gather()`; returns both responses + retrieved sources + metrics
  - `POST /api/search` — direct ChromaDB vector search for the Knowledge Base view
  - `GET /api/health` — liveness check for all downstream services (vLLM, ChromaDB)

> **ChromaDB v2 UUID routing:** Query endpoints require collection UUIDs, not names. `comparator.py` fetches and caches the name→UUID mapping at first request via `GET /collections`.

### Frontend Dashboard — React + Vite ✅

- **Framework:** React 18 + TypeScript + Vite
- **Styling:** Tailwind CSS
- **Charts:** Recharts
- **Port:** `3000`
- **Proxy:** `/api/*` proxied to `http://api:8002` inside the Docker network
- **Views:**
  - **Chat & Compare** — query input, side-by-side base vs RAG responses, scrollable sources panel with similarity scores and document metadata, quality metrics row
  - **Analytics** — session latency trend (line chart) and grounding score history (bar chart)
  - **Knowledge Base** — direct ChromaDB vector search with similarity scores and document metadata
  - **Health Profile** — body metrics, blood markers, goal selector, medical notes — persisted in state and included in every query
  - **Config** — top-k parameter control, system info panel

---

## Frontend Dashboard

### Architecture

```
Browser (port 3000)
    │
    ▼
React/Vite (services/frontend)
    │  /api/* proxied to api:8002
    ▼
FastAPI Middleware (services/api, port 8002)
    │
    ├── asyncio.gather()
    │       ├── vLLM base call (no context)             → llm:8000
    │       └── ChromaDB embed+retrieve → vLLM RAG call → llm:8000
    │
    └── JSON: base_response + rag_response + sources + metrics
```

### API Contract

#### `POST /api/chat`

Request:
```json
{
  "query": "How much protein should I eat for muscle gain?",
  "health_profile": { "weight_kg": 80, "lbm_kg": 66, "goal": "muscle_gain" },
  "query_type": "meal_plan",
  "top_k": 6
}
```

`query_type` is auto-detected from keywords when not supplied:
- `meal_plan` — food, eat, protein, calorie, recipe, diet
- `gym_program` — workout, gym, exercise, lift, train, rep, set
- `lab_analysis` — blood, lab, ldl, hdl, glucose, creatinine, marker
- `general` — fallback, queries all 4 collections

Response:
```json
{
  "base_response":  { "content": "...", "latency_s": 14.6, "prompt_tokens": 74,  "completion_tokens": 247, "error": null },
  "rag_response":   { "content": "...", "latency_s": 26.4, "prompt_tokens": 737, "completion_tokens": 392, "error": null },
  "retrieval": {
    "chunks_retrieved": 8,
    "collections_queried": ["nutrition_guidelines", "food_and_recipes"],
    "retrieval_latency_s": 0.31,
    "sources": [
      {
        "score": 0.68,
        "collection": "nutrition_guidelines",
        "excerpt": "1.4 to 1.8 g·kg⁻¹·d⁻¹ for strength-trained athletes...",
        "metadata": { "source": "JISSN Position Stand on Protein and Exercise.pdf" }
      }
    ]
  },
  "metrics": {
    "grounding_score": 0.71,
    "base_score": 0.0,
    "rag_improvement": 0.71,
    "latency_delta_s": 11.8
  }
}
```

#### `POST /api/search`

```json
{ "query": "protein synthesis resistance training", "top_k": 8 }
```

Returns `{ "sources": [...], "total": N }` — same source format as above.

#### `GET /api/health`

```json
{
  "status": "ok",
  "services": {
    "vllm":     { "status": "ok", "model": "/models/llama" },
    "chromadb": { "status": "ok", "collections": 4 }
  }
}
```

### Implementation Steps

| Step | Description | Status |
|------|-------------|--------|
| 1 | FastAPI skeleton + `/api/health` | ✅ Complete |
| 2 | Parallel RAG + base comparator, ChromaDB v2 UUID querying | ✅ Complete |
| 3 | Grounding score + token count metrics | ✅ Complete |
| 4 | SSE token streaming | 🔜 Future enhancement |
| 5 | React + Vite + Tailwind scaffold | ✅ Complete |
| 6 | Chat & Compare view | ✅ Complete |
| 7 | Analytics view with session charts | ✅ Complete |
| 8 | Knowledge Base search explorer | ✅ Complete |
| 9 | Docker containerisation + compose integration | ✅ Complete |
| 10 | EC2 Security Group lockdown | 🔜 Before demo |

### Key learnings from frontend build

- ChromaDB's internal Docker port is `8000`, not `8001` — host mapping `8001→8000` only applies outside the Docker network
- ChromaDB v2 requires collection UUIDs in query paths — names work for GET metadata but not `/query`
- `verbatimModuleSyntax: true` in `tsconfig.app.json` breaks TypeScript interface imports in Vite — remove it
- Vite proxy `target` must use Docker service name (`http://api:8002`) inside container, not `localhost`
- `--no-deps` on `docker compose up --build` prevents unnecessary re-pull of the 15 GB vLLM image
- Real disk space culprits: pip cache, JetBrains IDE cache, unused Python venvs — not Docker build cache

---

## Prerequisites

Before running anything, the host server needs:

1. **Ubuntu 22.04 LTS** (tested, recommended)
2. **NVIDIA GPU drivers** (535+) and **CUDA 12.2**
3. **Docker** + **Docker Compose plugin**
4. **NVIDIA Container Toolkit** (GPU access inside Docker)
5. **Git**
6. **Python 3.12 + python3.12-venv** (for model download, training, and ingestion scripts)
7. **HuggingFace account** with Meta Llama 3.1 license accepted at `huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct`

---

## Fresh Server Setup Guide

Follow these steps in order on a clean Ubuntu 22.04 server.

### Step 1 — Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2 — Verify GPU

```bash
nvidia-smi
```

Expected output:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01   Driver Version: 535.288.01   CUDA Version: 12.2              |
|   0  Tesla T4         Off      |  16160MiB VRAM                                      |
+---------------------------------------------------------------------------------------+
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
sudo usermod -aG docker $USER
newgrp docker
```

### Step 4 — Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Step 5 — Install Git and Configure GitHub

```bash
sudo apt install -y git
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
ssh-keygen -t ed25519 -C "your@email.com"
cat ~/.ssh/id_ed25519.pub   # Add to GitHub → Settings → SSH keys
```

### Step 6 — Install Python venv

```bash
sudo apt install -y python3.12-venv python-is-python3
```

### Step 7 — Clone the Repository

```bash
git clone git@github.com:gharaehs/ai-health-orchestration.git
cd ai-health-orchestration
```

---

## Model Download

The model is **not stored in git**. Run this once on every new server:

```bash
export HF_TOKEN="hf_your_token_here"
bash scripts/download_model.sh
```

Downloads `meta-llama/Meta-Llama-3.1-8B-Instruct` (~15 GB) to `./models/llama/`. Takes 20–40 minutes.

---

## RAG Corpus Ingestion

### Corpus Structure

```
data/corpus/
├── public_health_recommendations/   # Lab reference PDFs, clinical guideline PDFs
├── nutrition_guidelines/            # Research PDFs, DRI DOCX tables, WHO guidelines
├── gym_programming/                 # ACSM/NSCA position stand PDFs, meta-analysis PDFs
└── food_and_recipes/
    ├── All_Diets.csv
    ├── usda_foundation/             # Unzipped USDA Foundation Foods CSVs
    └── usda_sr_legacy/              # Unzipped USDA SR Legacy CSVs
```

### Setup and Run

```bash
python3 -m venv ~/.venv-rag
source ~/.venv-rag/bin/activate
pip install chromadb sentence-transformers pymupdf pandas tiktoken tqdm python-docx

make start   # ChromaDB must be running first
python3 scripts/ingest.py
```

### Collections Created

| Collection | Documents | Content |
|------------|-----------|---------|
| `public_health_recommendations` | 4,246 | Lab reference ranges, clinical guidelines |
| `nutrition_guidelines` | 881 | Research papers, DRI tables |
| `gym_programming` | 416 | ACSM/NSCA position stands, meta-analyses |
| `food_and_recipes` | 7,806 | Recipes with macros + USDA nutrient data |
| **Total** | **13,349** | |

---

## Running the System

```bash
make start    # Build and start all 4 services: llm, vector-db, api, frontend
make logs     # Watch startup logs
make stop     # Stop all services
```

Access the dashboard at `http://<your-ec2-public-ip>:3000`

---

## Testing & Verification

### Full stack health check

```bash
curl http://localhost:8002/api/health | python3 -m json.tool
```

### LLM direct test

```bash
make test-llm
```

### RAG pipeline CLI tests

```bash
source ~/.venv-rag/bin/activate
python scripts/test_rag.py --health-check
python scripts/test_rag.py --type meal_plan --compare
python scripts/test_rag.py --type lab_analysis --compare
python scripts/test_rag.py --evaluate
```

### API chat endpoint test

```bash
curl -s http://localhost:8002/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How much protein should I eat for muscle gain?",
    "health_profile": { "weight_kg": 80, "goal": "muscle_gain" },
    "query_type": "meal_plan",
    "top_k": 4
  }' | python3 -m json.tool
```

### Check ChromaDB

```bash
curl http://localhost:8001/api/v2/heartbeat
```

### Check GPU memory

```bash
nvidia-smi
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make start` | Build and start all 4 services in detached mode |
| `make stop` | Stop and remove all containers |
| `make logs` | Follow live logs from all containers |
| `make download-model` | Download Llama 3.1 8B to `./models/llama/` |
| `make test-llm` | Send a test prompt and display JSON response |
| `make test-api` | Check FastAPI middleware health endpoint |
| `make start-frontend` | Start only api and frontend containers |
| `make logs-frontend` | Follow logs for api and frontend containers |
| `make build-frontend` | Rebuild api and frontend Docker images |

---

## Migrating to a New Server

| Data | Location | In Git? | Migrate how? |
|------|----------|---------|--------------|
| Code & config | GitHub | ✅ Yes | `git clone` |
| Docker setup | GitHub | ✅ Yes | `git clone` |
| RAG pipeline | GitHub | ✅ Yes | `git clone` |
| Training scripts & dataset | GitHub | ✅ Yes | `git clone` |
| Frontend & API source | GitHub | ✅ Yes | `git clone` |
| Llama 3.1 model | `./models/llama/` | ❌ No | `bash scripts/download_model.sh` |
| LoRA adapter | `./models/adapters/` | ❌ No | Re-train or `scp` |
| Vector DB data | `./data/chroma/` | ❌ No | `scp` or re-run `ingest.py` |
| Corpus files | `./data/corpus/` | ❌ No | `scp` from original machine |

```bash
git clone git@github.com:gharaehs/ai-health-orchestration.git
cd ai-health-orchestration
export HF_TOKEN="hf_your_token_here"
bash scripts/download_model.sh
make start
scp -r ./data/corpus/ user@new-server:~/ai-health-orchestration/data/
source ~/.venv-rag/bin/activate && python3 scripts/ingest.py
curl http://localhost:8002/api/health
# Dashboard: http://<ip>:3000
```

---

## Troubleshooting

### vLLM crashes with CUDA Out of Memory

Ensure `--enforce-eager` is in the `docker-compose.yml` command section.

### ChromaDB collections empty after restart

Ensure `docker-compose.yml` mounts `./data/chroma:/data`. Re-run `python3 scripts/ingest.py`.

### ChromaDB v2 — collection ID error

**Symptom:** `"Collection ID is not a valid UUIDv4"`

Already handled in `comparator.py` — name→UUID map cached at first request. If it recurs, restart the api container to clear the cache.

### FastAPI cannot reach vLLM or ChromaDB

Use Compose service names, not `localhost`: `http://llm:8000` and `http://vector-db:8000`.

ChromaDB's **internal** port is `8000` (host maps `8001→8000`). Always use `http://vector-db:8000` inside Docker.

### Frontend shows API errors in browser console

Check the Vite proxy target in `vite.config.ts` — must be `http://api:8002`, not `localhost:8002`. Check API logs: `docker compose logs api`.

### tsconfig TypeScript import error

**Symptom:** `The requested module does not provide an export named 'X'`

Remove `"verbatimModuleSyntax": true`, `"noUnusedLocals": true`, and `"noUnusedParameters": true` from `tsconfig.app.json`. Already fixed in this repo.

### ChromaDB returns v1 API deprecated error

Use `curl http://localhost:8001/api/v2/heartbeat` — v1 is deprecated in ChromaDB 1.5.8.

### RAG query times out

Timeout is set to 600s in `rag/pipeline.py` and `comparator.py`. Normal generation on T4 is 15–60s depending on output length.

### Docker build fails with "no space left on device"

Free space first: `rm -rf ~/.cache/pip ~/.cache/JetBrains ~/.cache/huggingface`. Do not rebuild the vLLM image — use `docker-compose.yml` command overrides only.

### vLLM uses FlashInfer instead of FlashAttention2

Not an error. T4 (compute capability 7.5) does not support FlashAttention2 (requires 8.0+).

---

## Module Progress

### ✅ Module 1 — Local LLM Deployment (Complete)

- EC2 g4dn.xlarge + Ubuntu 22.04 + CUDA 12.2
- vLLM serving Llama 3.1 8B via OpenAI-compatible API on port 8000
- ChromaDB on port 8001, CUDA OOM resolved with `--enforce-eager`

**Key learnings:** 1B params ≈ 2 GB VRAM in FP16 · T4 uses FlashInfer not FlashAttention2 · CUDA graph warmup is the OOM culprit

---

### ✅ Module 2 — Parameter-Efficient Fine-Tuning / LoRA (Complete)

- QLoRA: Llama 3.1 8B in 4-bit via BitsAndBytes, LoRA rank=16 alpha=32
- 21-example health domain training dataset across 6 categories
- LoRA adapter: `./models/adapters/health-v1/` (~161 MB)

**Key learnings:** AWQ is inference-only · BitsAndBytes NF4 is correct for QLoRA · single fp16 model serves both training and inference

---

### ✅ Module 3 — Vector Database & RAG Corpus Ingestion (Complete)

- 4 ChromaDB collections, 13,349 documents
- Sources: JISSN position stands, ACSM guidelines, USDA food data, clinical lab references
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` on CPU

**Key learnings:** Chunking varies by document type · ChromaDB 1.5.8 uses v2 API · volume must mount to `/data`

---

### ✅ Module 4 — Retrieval-Augmented Generation (Complete)

- End-to-end RAG: embed → retrieve → augmented prompt → structured JSON
- STRUCTURED prompt strategy with labelled context blocks per collection
- Key finding: RAG returns 4-tier LDL clinical classification from ingested corpus; base model returns simplified flat range

| Metric | With RAG | No RAG |
|--------|----------|--------|
| Latency | 58.8s | 37.1s |
| Prompt tokens | 2,289 | 417 |
| Chunks retrieved | 6 | 0 |

**Key learnings:** Retrieval ~0.3s — all overhead is generation · STRUCTURED outperforms naive concatenation

---

### 🔜 Module 5 — Agentic AI / Multi-Agent Orchestration (Next)

Planned: Lab Analysis Agent, Nutrition Agent, Training Agent, Grocery Agent — each scoped to a dedicated ChromaDB collection, coordinating via structured intermediate outputs through an orchestration loop.

---

### 🔜 Module 6 — Model Context Protocol / MCP (Planned)

Planned: Expose health data, workout history, and recipe database via MCP servers.

---

### ✅ Frontend Dashboard (Complete — Steps 1–9)

**Built:**

- `services/api/` — FastAPI middleware on port 8002
  - `POST /api/chat` — parallel RAG + base execution via `asyncio.gather()`
  - `POST /api/search` — direct ChromaDB vector search for Knowledge Base view
  - `GET /api/health` — downstream service liveness
  - Grounding score: keyword overlap between RAG response and retrieved chunks
  - ChromaDB v2 UUID caching for all collection queries

- `services/frontend/` — React 18 + TypeScript + Vite + Tailwind on port 3000
  - **Chat & Compare** — query input, side-by-side base vs RAG, scrollable sources with scores and metadata, metrics row
  - **Analytics** — session latency trend and grounding score history via recharts
  - **Knowledge Base** — direct ChromaDB search UI
  - **Health Profile** — body metrics, blood markers, goal, notes — included in every query
  - **Config** — top-k control, system info

- All 4 services in `docker-compose.yml` — `make start` brings up the full stack
- Vite proxy: `/api/*` → `http://api:8002` inside Docker

**Remaining:**

| Step | Description |
|------|-------------|
| 4 | SSE token streaming (future enhancement) |
| 10 | EC2 Security Group lockdown (before demo) |

---

## Design Decisions

### Why Llama 3.1 8B over Mistral 7B?

| Factor | Llama 3.1 8B | Mistral 7B AWQ |
|--------|-------------|----------------|
| Context window | 128K tokens | 4K tokens |
| RAG suitability | ✅ Excellent | ❌ Too short for retrieved docs |
| LoRA training | ✅ fp16 + BitsAndBytes | ❌ AWQ is inference-only |
| Single model train + serve | ✅ Yes | ❌ Requires separate versions |

### Why FastAPI + React over Streamlit or Gradio?

| Factor | FastAPI + React | Streamlit / Gradio |
|--------|----------------|-------------------|
| Parallel RAG + base calls | ✅ Native async | ❌ Requires workarounds |
| SSE token streaming | ✅ First-class | ⚠️ Limited |
| Custom analytics UI | ✅ Full control | ❌ Constrained widgets |
| Production extensibility | ✅ Yes (Module 5+) | ❌ Prototype-only |

### Why a FastAPI middleware instead of the frontend calling vLLM directly?

Direct browser-to-vLLM calls require exposing port 8000 publicly. The middleware enables logic that cannot run in the browser: `asyncio.gather()` parallel execution, grounding score computation, and ChromaDB UUID resolution.

### Why STRUCTURED over NAIVE prompt strategy?

Labelled context blocks (`[CONTEXT: Clinical Lab Reference Ranges]`) tell the model the source and authority of each retrieved chunk, producing stronger grounding than flat concatenation.

### Why four separate ChromaDB collections?

Each agent in Module 5 queries only its relevant domain. A Lab Analysis Agent should never retrieve a recipe when looking up an LDL threshold. Collection-level isolation enforces this cleanly without metadata filtering complexity.

### Why CPU for embeddings?

Preserves the T4's full 16 GB VRAM for Llama 3.1 inference. `all-MiniLM-L6-v2` on CPU runs at ~0.3s per query — fast enough for this workload.

### Why vLLM over llama.cpp?

vLLM is purpose-built for GPU servers: PagedAttention, built-in OpenAI-compatible API, first-class Docker support. llama.cpp is best for CPU or laptop environments.

### Why ChromaDB over FAISS?

ChromaDB runs as a persistent server with a REST API accessible from all containers. FAISS is a library embedded in application code — not suitable for a multi-service Docker architecture.

### Why runtime flags in docker-compose.yml instead of the Dockerfile?

The vLLM image is ~15 GB. Rebuilding on the T4's 80 GB disk exhausts space. The `command` override in `docker-compose.yml` takes effect at container start with no rebuild.

---

## Links

- **GitHub Repository:** https://github.com/gharaehs/ai-health-orchestration
- **Mentorship Program:** AI Technical Deep Dive
- **Model:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **vLLM Documentation:** https://docs.vllm.ai
- **ChromaDB Documentation:** https://docs.trychroma.com

---

*Last updated: Modules 1–4 complete · Frontend dashboard complete (Steps 1–9) · May 2026*