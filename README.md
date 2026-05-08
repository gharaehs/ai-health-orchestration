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
| Frontend | Analytics Dashboard & Chat Interface | 🔄 In Progress (Steps 1–2 of 10 complete) |

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
│  │   Port: 8000         │    │    Port: 8001                 │   │
│  │                      │    │                               │   │
│  │  Llama 3.1 8B        │    │  4 Collections (13,349 docs)  │   │
│  │  Instruct (fp16)     │    │  - public_health_recs         │   │
│  │  GPU: Tesla T4       │    │  - nutrition_guidelines       │   │
│  └──────────┬───────────┘    │  - gym_programming            │   │
│             │                │  - food_and_recipes           │   │
│             ▼                └──────────────────────────────┘   │
│  ┌──────────────────────┐                                        │
│  │   FastAPI Container  │                                        │
│  │   Port: 8002         │                                        │
│  │                      │                                        │
│  │  Parallel RAG+base   │                                        │
│  │  comparator          │                                        │
│  │  Metrics & grounding │                                        │
│  └──────────────────────┘                                        │
│                                                                  │
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
POST /api/chat
    │
    ├── asyncio.gather()
    │       ├── Base call → vLLM (no context)
    │       └── ChromaDB query → embed → retrieve chunks → augmented call → vLLM
    │
    └── Return: base_response + rag_response + retrieval sources + metrics
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
├── docker-compose.yml              # Orchestrates all services
├── Makefile                        # Command interface (start, stop, test, etc.)
├── .gitignore
├── README.md
│
├── services/
│   ├── llm/
│   │   └── Dockerfile              # vLLM container configuration
│   ├── api/                        # ✅ Frontend Step 1–2: FastAPI middleware
│   │   ├── Dockerfile
│   │   ├── main.py                 # FastAPI app + /api/health endpoint
│   │   ├── routes/
│   │   │   └── chat.py             # POST /api/chat (parallel RAG + base calls)
│   │   ├── core/
│   │   │   ├── comparator.py       # asyncio.gather() RAG vs base runner
│   │   │   ├── metrics.py          # Grounding score, token count helpers
│   │   │   └── config.py           # Service URLs, timeouts
│   │   └── requirements.txt
│   └── frontend/                   # 🔜 Frontend Steps 5–9: React/Vite dashboard
│       ├── Dockerfile
│       ├── package.json
│       ├── vite.config.ts
│       └── src/
│           ├── App.tsx
│           ├── components/
│           │   ├── ChatPanel.tsx
│           │   ├── ComparePanel.tsx
│           │   ├── SourcesPanel.tsx
│           │   ├── MetricsBar.tsx
│           │   └── StatusBar.tsx
│           └── hooks/
│               └── useChat.ts
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
- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (fp16, loaded with on-the-fly 4-bit quantization via BitsAndBytes)
- **Port:** `8000`
- **API:** OpenAI-compatible (`/v1/chat/completions`, `/v1/models`, etc.)
- **Runtime flags** (set via `docker-compose.yml` command override):
  - `--dtype float16` — serves fp16 model efficiently on T4
  - `--max-model-len 16384` — 16K context window (increased from 8K for RAG use)
  - `--gpu-memory-utilization 0.85` — reserves 15% VRAM as safety buffer
  - `--max-num-seqs 8` — limits concurrent sequences
  - `--enforce-eager` — disables CUDA graph capture (required for T4, prevents OOM)
  - `--quantization bitsandbytes` + `--load-format bitsandbytes` — on-the-fly 4-bit quantization

> **Why `--max-model-len 16384`?**
> The RAG pipeline injects up to 12,000 chars (~3,000 tokens) of retrieved context into each prompt. Combined with the health profile, output schema, and 2,048 token generation budget, the original 8,192 limit was too tight. 16,384 provides comfortable headroom without exceeding T4 KV cache capacity.

> **Why `--enforce-eager`?**
> The T4 GPU (CUDA compute capability 7.5) does not support vLLM's default CUDA graph warmup (512 simultaneous graph captures), which causes OOM. `--enforce-eager` disables this. Slightly slower for high-throughput scenarios but correct for single-user inference.

> **Why runtime flags in `docker-compose.yml` instead of the Dockerfile?**
> The vLLM image (~15 GB) cannot be rebuilt on the T4 instance without exhausting the 80 GB disk (Docker's build cache requires temporary space equal to the image size). The `command` override in `docker-compose.yml` bypasses the Dockerfile `CMD` at runtime — no rebuild needed.

### Vector Database — ChromaDB

- **Image:** `chromadb/chroma:latest`
- **Version:** 1.5.8 (v2 API)
- **Port:** `8001`
- **Storage:** `./data/chroma/` mounted to `/data` inside the container
- **Collections:** 4 domain-specific collections (13,349 documents total)
- **Used in:** Modules 3, 4, and 5+

> **Critical volume mount:** ChromaDB writes its data to `/data` inside the container. The correct bind mount is `./data/chroma:/data`. Using `/chroma/chroma` as the destination causes data to be written to an ephemeral container layer and lost on restart.

### API Middleware — FastAPI ✅

- **Framework:** FastAPI + Uvicorn
- **Port:** `8002`
- **Purpose:** Sits between the frontend and existing services; runs parallel RAG and base-model calls, collects analytics, and returns structured comparison payloads
- **Key endpoints:**
  - `POST /api/chat` — accepts health profile + query; runs both RAG and base paths in parallel via `asyncio.gather()`; returns both responses + retrieval sources + metrics
  - `GET /api/health` — liveness check for all downstream services (vLLM, ChromaDB)
- **Internal Docker network:** communicates with vLLM at `http://llm:8000` and ChromaDB at `http://vector-db:8000` using Docker Compose service names

> **ChromaDB internal port:** The host maps port 8001 → container port 8000. Inside the Docker network, the API container reaches ChromaDB at `http://vector-db:8000`, not `:8001`.

> **ChromaDB v2 UUID routing:** ChromaDB 1.5.8 requires collection UUIDs (not names) in query endpoint paths. The comparator fetches and caches the name→UUID mapping at first request via `GET /collections`.

### Frontend Dashboard — React + Vite

- **Framework:** React 18 + TypeScript + Vite
- **Port:** `3000`
- **Styling:** Tailwind CSS
- **Status:** 🔜 Planned — React scaffold not yet built

---

## Frontend Dashboard — Architecture & Plan

### Overview

The dashboard is a professional analytics interface that exposes the full intelligence of the AI pipeline — not just the final answer but every intermediate step. It is added as two new Docker containers (`api` and `frontend`) in the existing `docker-compose.yml` stack. The existing vLLM and ChromaDB containers are unchanged.

### Full Stack Architecture

```
Browser (port 3000)
    │
    │  HTTP / SSE streaming
    ▼
FastAPI Middleware (port 8002)        ← ✅ Built: services/api/
    │
    ├──────────────────────────────────────────────┐
    │   parallel calls (asyncio.gather)            │
    ▼                                              ▼
vLLM API (port 8000)             ChromaDB (port 8001)
[base call: no context]          [retrieve top-k chunks]
    │                                              │
    │                             augmented prompt │
    └──────────────────────────────────────────────┘
                          │
                          ▼
                   vLLM API (port 8000)
                   [RAG call: with context]
```

For each user request, the FastAPI middleware:

1. Sends the query to vLLM **without** context (baseline response)
2. Simultaneously queries ChromaDB for the top-k relevant chunks
3. Builds an augmented prompt and sends to vLLM **with** context (RAG response)
4. Measures latency and token counts for both paths
5. Computes a grounding score (keyword overlap between RAG response and retrieved chunks)
6. Returns both responses + all metadata as a single JSON payload

### Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Frontend framework | React 18 + TypeScript | Type safety, component reuse, large ecosystem |
| Build tool | Vite | Fast HMR, simple config, production-optimised bundles |
| Styling | Tailwind CSS | Utility-first, no CSS files to manage, dark mode built-in |
| API middleware | FastAPI + Uvicorn | Native async, OpenAPI docs auto-generated, minimal overhead |
| Streaming | Server-Sent Events (SSE) | Streams LLM tokens to browser without WebSocket complexity |
| HTTP client | `httpx` (async) | Async-native, supports streaming from vLLM |
| Containerisation | Docker + Docker Compose | Consistent with existing stack |

### Dashboard Views

#### 1. Chat & Compare (primary view)

The main interface. Split into four panels:

- **Chat panel** — health profile selector, free-text query input, streamed response output. Toggle switches for RAG on/off and LoRA adapter on/off
- **Comparison panel** — side-by-side view of base model response (no RAG) vs RAG-augmented response for the same query
- **Sources panel** — list of retrieved ChromaDB documents with cosine similarity scores, collection names, and excerpt previews
- **Metrics panel** — per-request quality scores: latency (base vs RAG), token counts, chunks retrieved, grounding score, RAG improvement delta

#### 2. Analytics (session view)

Session-level aggregates: latency trend, similarity score distribution, collection breakdown, grounding score over time.

#### 3. Knowledge Base (explorer view)

Browse ChromaDB collections, search directly, preview chunks with similarity scores and document metadata.

#### 4. Health Profile (input view)

Structured form: scale metrics, blood test results, goals, medical history. Included in every query.

#### 5. Model Config (settings view)

Model selector, RAG parameters (top-k, context chars, collection routing), generation parameters.

### API Contract

#### `POST /api/chat`

Request:
```json
{
  "query": "Based on my blood work, what protein target should I set?",
  "health_profile": { "weight_kg": 82, "lbm_kg": 67, "ldl_mmol": 3.1, "goal": "muscle_gain" },
  "query_type": "lab_analysis",
  "top_k": 6
}
```

Response:
```json
{
  "base_response":  { "content": "...", "latency_s": 14.6, "prompt_tokens": 74,  "completion_tokens": 247 },
  "rag_response":   { "content": "...", "latency_s": 26.4, "prompt_tokens": 737, "completion_tokens": 392 },
  "retrieval": {
    "chunks_retrieved": 8,
    "collections_queried": ["nutrition_guidelines", "food_and_recipes"],
    "retrieval_latency_s": 0.31,
    "sources": [
      { "score": 0.68, "collection": "nutrition_guidelines",
        "excerpt": "1.4 to 1.8 g·kg⁻¹·d⁻¹ for strength-trained athletes...",
        "metadata": { "source": "JISSN Position Stand on Protein and Exercise.pdf" } }
    ]
  },
  "metrics": { "grounding_score": 0.71, "base_score": 0.0, "rag_improvement": 0.71, "latency_delta_s": 11.8 }
}
```

#### `GET /api/health`

```json
{
  "status": "ok",
  "services": {
    "vllm":    { "status": "ok", "model": "/models/llama" },
    "chromadb": { "status": "ok", "collections": 4 }
  }
}
```

### Docker Compose Integration

```yaml
  api:
    build: ./services/api
    ports:
      - "8002:8002"
    environment:
      - VLLM_URL=http://llm:8000
      - CHROMA_URL=http://vector-db:8000
    depends_on:
      - llm
      - vector-db

  frontend:
    build: ./services/frontend
    ports:
      - "3000:3000"
    depends_on:
      - api
```

### Network & Security

All AI services (8000, 8001, 8002) are internal to the Docker network. On AWS, restrict EC2 Security Group inbound rules:

| Port | Source | Purpose |
|------|--------|---------|
| 22 | Your IP only | SSH |
| 3000 | Your IP only | Frontend dashboard |
| 8002 | Your IP only | API (or proxy through frontend) |
| 8000 | Docker network only | vLLM — internal |
| 8001 | Docker network only | ChromaDB — internal |

### Implementation Steps

| Step | Description | Status |
|------|-------------|--------|
| 1 | FastAPI skeleton + `/api/health` endpoint | ✅ Complete |
| 2 | Parallel RAG + base comparator with ChromaDB v2 UUID querying | ✅ Complete |
| 3 | Metrics computation (grounding score, token counts) | ✅ Complete (included in Step 2) |
| 4 | SSE streaming for token-by-token frontend delivery | 🔜 Next |
| 5 | React + Vite + Tailwind scaffold with sidebar navigation | 🔜 Planned |
| 6 | Chat & Compare view (ChatPanel, ComparePanel, SourcesPanel, MetricsBar) | 🔜 Planned |
| 7 | Analytics view with recharts session charts | 🔜 Planned |
| 8 | Knowledge Base explorer (ChromaDB search UI) | 🔜 Planned |
| 9 | Docker integration + `make start` full stack test | 🔜 Planned |
| 10 | EC2 Security Group lockdown | 🔜 Planned |

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

This script will:
1. Create a Python virtual environment at `~/.venv-hf`
2. Install `huggingface_hub`
3. Download `meta-llama/Meta-Llama-3.1-8B-Instruct` (~15 GB) to `./models/llama/`

> Download takes approximately 20–40 minutes. Requires a HuggingFace account with Meta's Llama 3.1 license accepted.

---

## RAG Corpus Ingestion

The ChromaDB vector store is populated by running the ingestion pipeline. Required once on every new server after placing corpus files in `data/corpus/`.

### Corpus Structure

```
data/corpus/
├── public_health_recommendations/   # Lab reference PDFs, clinical guideline PDFs
├── nutrition_guidelines/            # Research PDFs, DRI DOCX tables, WHO guidelines
├── gym_programming/                 # ACSM/NSCA position stand PDFs, meta-analysis PDFs
└── food_and_recipes/
    ├── All_Diets.csv                # Recipe dataset with macros
    ├── usda_foundation/             # Unzipped USDA Foundation Foods CSVs
    └── usda_sr_legacy/              # Unzipped USDA SR Legacy CSVs
```

> **USDA data:** Download Foundation Foods and SR Legacy ZIP files from `fdc.nal.usda.gov/download-datasets/` and unzip into the subdirectories above.

### Setting Up the RAG Environment

```bash
python3 -m venv ~/.venv-rag
source ~/.venv-rag/bin/activate
pip install chromadb sentence-transformers pymupdf pandas tiktoken tqdm python-docx
```

### Running Ingestion

```bash
make start   # ChromaDB must be running first
source ~/.venv-rag/bin/activate
python3 scripts/ingest.py
```

### Collections Created

| Collection | Documents | Content |
|------------|-----------|---------|
| `public_health_recommendations` | 4,246 | Lab reference ranges, clinical guideline PDFs |
| `nutrition_guidelines` | 881 | Research papers (PDF), DRI tables (DOCX) |
| `gym_programming` | 416 | ACSM/NSCA position stands, meta-analyses (PDF) |
| `food_and_recipes` | 7,806 | Recipes with macros + USDA nutrient data per 100g |
| **Total** | **13,349** | |

### Verify Ingestion

```bash
python3 - << 'EOF'
import chromadb
client = chromadb.HttpClient(host="localhost", port=8001)
for name in ["public_health_recommendations", "nutrition_guidelines",
             "gym_programming", "food_and_recipes"]:
    col = client.get_collection(name)
    print(f"{name}: {col.count()} documents")
EOF
```

---

## Running the System

```bash
make start    # Start all services (vllm, vector-db, api)
make logs     # Watch startup logs — wait for: INFO: Application startup complete.
make stop     # Stop all services
```

---

## Testing & Verification

### LLM test

```bash
make test-llm
```

### RAG pipeline tests (Module 4)

```bash
source ~/.venv-rag/bin/activate

python scripts/test_rag.py --health-check
python scripts/test_rag.py --type meal_plan --compare
python scripts/test_rag.py --type lab_analysis --compare
python scripts/test_rag.py --evaluate
```

### API middleware test

```bash
# Health check
curl http://localhost:8002/api/health | python3 -m json.tool

# Full comparison request
curl -s http://localhost:8002/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How much protein should I eat for muscle gain?",
    "health_profile": { "weight_kg": 80, "goal": "muscle_gain" },
    "query_type": "meal_plan",
    "top_k": 4
  }' | python3 -m json.tool
```

### Manual vLLM test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/llama",
    "messages": [{"role": "user", "content": "Say hello!"}],
    "max_tokens": 50
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
| `make start` | Build and start all services in detached mode |
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

### What lives where

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

### Migration steps

```bash
git clone git@github.com:gharaehs/ai-health-orchestration.git
cd ai-health-orchestration
export HF_TOKEN="hf_your_token_here"
bash scripts/download_model.sh
make start
scp -r ./data/corpus/ user@new-server:~/ai-health-orchestration/data/
source ~/.venv-rag/bin/activate && python3 scripts/ingest.py
make test-llm
curl http://localhost:8002/api/health
```

---

## Troubleshooting

### vLLM crashes with CUDA Out of Memory

**Fix:** Ensure `--enforce-eager` is present in the `docker-compose.yml` command section.

### ChromaDB collections empty after restart

**Fix:** Ensure `docker-compose.yml` mounts `./data/chroma:/data`. Re-run `python3 scripts/ingest.py` after fixing.

### ChromaDB v2 API — collection ID error

**Symptom:** `"Collection ID is not a valid UUIDv4"`

**Cause:** ChromaDB 1.5.8 requires UUID in query endpoint paths, not collection names.

**Fix:** Already handled in `comparator.py` — it fetches and caches the name→UUID mapping via `GET /collections` at first request.

### FastAPI cannot reach vLLM or ChromaDB

**Cause:** Using `localhost` inside a Docker container resolves to that container only.

**Fix:** Use Compose service names: `http://llm:8000` and `http://vector-db:8000`.

> Note: ChromaDB's **internal** port is 8000 (mapped to host port 8001). Inside the Docker network always use `http://vector-db:8000`.

### ChromaDB returns v1 API deprecated error

Use `curl http://localhost:8001/api/v2/heartbeat` — v1 is deprecated in 1.5.8.

### RAG query times out

Timeout is set to 600s in `rag/pipeline.py` and `comparator.py`. T4 generation at 2,048 tokens takes 3–4 minutes normally.

### Docker build fails with "no space left on device"

Free space before building: `rm -rf ~/.cache/pip ~/.cache/JetBrains ~/.cache/huggingface`. Do not rebuild the vLLM image — use `docker-compose.yml` `command` overrides instead.

### vLLM uses FlashInfer instead of FlashAttention2

Not an error. T4 (compute capability 7.5) does not support FlashAttention2 (requires 8.0+).

---

## Module Progress

### ✅ Module 1 — Local LLM Deployment (Complete)

- EC2 g4dn.xlarge + Ubuntu 22.04 + CUDA 12.2
- vLLM serving Llama 3.1 8B via OpenAI-compatible API on port 8000
- ChromaDB running on port 8001
- CUDA OOM resolved with `--enforce-eager`

---

### ✅ Module 2 — Parameter-Efficient Fine-Tuning / LoRA (Complete)

- QLoRA fine-tuning: Llama 3.1 8B in 4-bit via BitsAndBytes, LoRA rank=16 alpha=32
- 21-example health domain training dataset
- LoRA adapter saved to `./models/adapters/health-v1/` (~161 MB)

---

### ✅ Module 3 — Vector Database & RAG Corpus Ingestion (Complete)

- 4 ChromaDB collections, 13,349 documents ingested
- Sources: JISSN position stands, ACSM guidelines, USDA food data, clinical lab references
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` on CPU

---

### ✅ Module 4 — Retrieval-Augmented Generation (Complete)

- End-to-end RAG pipeline: embed → retrieve → augmented prompt → structured JSON
- STRUCTURED prompt strategy with labelled context blocks per collection
- Key finding: RAG returns 4-tier LDL clinical classification from ingested corpus vs. simplified flat range from base model

**RAG pipeline metrics (lab analysis):**

| Metric | With RAG | No RAG |
|--------|----------|--------|
| Latency | 58.8s | 37.1s |
| Prompt tokens | 2,289 | 417 |
| Chunks retrieved | 6 | 0 |

---

### 🔜 Module 5 — Agentic AI / Multi-Agent Orchestration (Next)

Planned: Lab Analysis Agent, Nutrition Agent, Training Agent, Grocery Agent coordinating via structured intermediate outputs.

---

### 🔜 Module 6 — Model Context Protocol / MCP (Planned)

Planned: Expose health data, workout history, and recipe database via MCP servers.

---

### 🔄 Frontend Dashboard — In Progress

**Steps complete:**

**✅ Step 1 — FastAPI skeleton**
- `services/api/` with FastAPI + Uvicorn
- `GET /api/health` verifying vLLM and ChromaDB connectivity
- Added to `docker-compose.yml` on port 8002

**✅ Step 2 — Parallel RAG + base comparator**
- `core/comparator.py`: `asyncio.gather()` runs base call and ChromaDB retrieval concurrently
- `core/metrics.py`: grounding score (keyword overlap), approximate token counting
- ChromaDB v2 fix: fetch collection UUIDs via `GET /collections`, cache name→UUID map, use UUIDs for all query requests
- `sentence-transformers/all-MiniLM-L6-v2` embedded in the API container for query embedding
- `POST /api/chat` verified end-to-end: 8 chunks retrieved, RAG response grounded in JISSN corpus

**Key learnings from frontend Steps 1–2:**
- ChromaDB's internal Docker port is 8000, not 8001 — the host mapping `8001→8000` only applies outside the Docker network
- ChromaDB v2 API requires collection UUIDs in query paths — collection names work for metadata GET but not for `/query`
- `docker system prune` reclaims almost nothing if the build cache is already clean — the real space hogs are pip cache, JetBrains IDE cache, and unused Python venvs
- The `--no-deps` flag on `docker compose up --build` prevents Docker from re-pulling the 15GB vLLM image unnecessarily

**Steps remaining:**

| Step | Description |
|------|-------------|
| 4 | SSE streaming for token-by-token delivery to browser |
| 5 | React + Vite + Tailwind scaffold with sidebar navigation |
| 6 | Chat & Compare view with all four panels |
| 7 | Analytics view with session charts (recharts) |
| 8 | Knowledge Base explorer |
| 9 | Docker integration + full stack test |
| 10 | EC2 Security Group lockdown |

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

Direct browser-to-vLLM calls require exposing port 8000 publicly. The middleware also enables logic that cannot run in the browser: parallel `asyncio.gather()` execution, grounding score computation, and ChromaDB UUID resolution.

### Why STRUCTURED over NAIVE prompt strategy?

Labelled context blocks (`[CONTEXT: Clinical Lab Reference Ranges]`) tell the model the source and authority of each chunk, producing stronger grounding than flat concatenation.

### Why four separate ChromaDB collections?

Each agent in Module 5 queries only its relevant domain. A Lab Analysis Agent should never retrieve a recipe when looking up an LDL threshold. Collection-level isolation enforces this without metadata filtering complexity.

### Why CPU for embeddings?

Preserves the T4's full 16 GB VRAM for Llama 3.1 inference. `all-MiniLM-L6-v2` on CPU runs in ~0.3s per query — fast enough for this workload.

### Why vLLM over llama.cpp?

vLLM is purpose-built for GPU servers: PagedAttention, built-in OpenAI-compatible API, first-class Docker support. llama.cpp is best for CPU or laptop environments.

### Why ChromaDB over FAISS?

ChromaDB runs as a persistent server with a REST API accessible from all containers. FAISS is a library embedded in application code — unsuitable for a multi-service Docker architecture.

### Why runtime flags in docker-compose.yml instead of the Dockerfile?

The vLLM image is ~15 GB. Rebuilding on the T4's 80 GB disk exhausts available space. The `command` override in `docker-compose.yml` takes effect at container start with no rebuild.

---

## Links

- **GitHub Repository:** https://github.com/gharaehs/ai-health-orchestration
- **Mentorship Program:** AI Technical Deep Dive
- **Model:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **vLLM Documentation:** https://docs.vllm.ai
- **ChromaDB Documentation:** https://docs.trychroma.com

---

*Last updated: Modules 1–4 complete · Frontend Steps 1–2 complete · May 2026*