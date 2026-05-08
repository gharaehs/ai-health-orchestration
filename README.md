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
| Frontend | Analytics Dashboard & Chat Interface | 🔜 Planned |

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
│   ├── api/                        # Module 7: FastAPI middleware service
│   │   ├── Dockerfile
│   │   ├── main.py                 # FastAPI app entrypoint
│   │   ├── routes/
│   │   │   ├── chat.py             # /api/chat endpoint (RAG + base parallel calls)
│   │   │   ├── evaluate.py         # /api/evaluate endpoint (comparison metrics)
│   │   │   └── health.py           # /api/health endpoint (service status)
│   │   ├── core/
│   │   │   ├── comparator.py       # Parallel RAG vs base response runner
│   │   │   ├── metrics.py          # Latency, token count, grounding score
│   │   │   └── config.py           # Service URLs, timeouts, model config
│   │   └── requirements.txt
│   └── frontend/                   # Module 7: React/Vite dashboard
│       ├── Dockerfile
│       ├── package.json
│       ├── vite.config.ts
│       └── src/
│           ├── App.tsx
│           ├── components/
│           │   ├── ChatPanel.tsx       # Chat input + streaming response
│           │   ├── ComparePanel.tsx    # Base vs RAG side-by-side
│           │   ├── SourcesPanel.tsx    # Retrieved docs with similarity scores
│           │   ├── MetricsBar.tsx      # Per-request quality metrics
│           │   └── StatusBar.tsx       # Live service health indicators
│           └── hooks/
│               └── useChat.ts          # SSE streaming + state management
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

### API Middleware — FastAPI (Module 7)

- **Framework:** FastAPI + Uvicorn
- **Port:** `8002`
- **Purpose:** Sits between the frontend and existing services; intercepts all requests, runs parallel RAG and base-model calls, collects analytics, and streams structured responses back to the UI
- **Key endpoints:**
  - `POST /api/chat` — accepts health profile + query; runs both RAG and base paths; returns streaming response + metadata
  - `GET /api/compare` — returns last request's side-by-side comparison payload
  - `GET /api/health` — liveness check for all downstream services (vLLM, ChromaDB)
- **Internal Docker network:** communicates with vLLM at `http://llm:8000` and ChromaDB at `http://vector-db:8001` using Docker Compose service names (not `localhost`)

### Frontend Dashboard — React + Vite (Module 7)

- **Framework:** React 18 + TypeScript + Vite
- **Port:** `3000`
- **Styling:** Tailwind CSS
- **Purpose:** Professional single-page analytics dashboard with chat interface, RAG comparison view, retrieved source explorer, and per-request quality metrics

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
FastAPI Middleware (port 8002)        ← NEW: services/api/
    │
    ├──────────────────────────────────────────────┐
    │   parallel calls                             │
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
5. Computes a grounding score (how much of the RAG answer traces to retrieved sources)
6. Returns both responses + all metadata to the frontend as a single JSON payload (or streamed via SSE)

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
- **Comparison panel** — side-by-side view of base model response (no RAG) vs RAG-augmented response for the same query. Differences are highlighted
- **Sources panel** — list of retrieved ChromaDB documents with cosine similarity scores, collection names, and excerpt previews. Each source is expandable to show full chunk text and document metadata
- **Metrics panel** — per-request quality scores: latency (base vs RAG), token counts, number of chunks retrieved, JSON validity, a computed grounding score (what % of key claims in the RAG answer trace to retrieved text), and a personalisation score

#### 2. Analytics (session view)

Session-level aggregates across all requests:

- Average RAG vs base latency over time (line chart)
- Distribution of similarity scores for retrieved chunks (histogram)
- Collection breakdown: which ChromaDB collections were used and how often (bar chart)
- JSON validity rate (base vs RAG, base vs LoRA)
- Grounding score trend across the session

#### 3. Knowledge Base (explorer view)

Browse the ChromaDB collections:

- Collection selector (nutrition_guidelines, gym_programming, etc.)
- Free-text search that queries ChromaDB directly and returns the top results with distances
- Document preview with chunk boundaries shown
- Collection stats: document count, embedding dimension, last ingestion timestamp

#### 4. Health Profile (input view)

Structured form for entering health data that is included in every query:

- Scale metrics: weight, body fat %, muscle mass, height
- Blood test results: LDL, HDL, total cholesterol, triglycerides, glucose, HbA1c, creatinine, testosterone
- Goals: fat loss / muscle gain / maintenance / performance
- Medical history: free text

#### 5. Model Config (settings view)

- Model selector (base, base + LoRA adapter, future models)
- RAG parameters: top-k, max context chars, collection routing strategy
- Generation parameters: temperature, max tokens, system prompt override

### API Contract

All endpoints served by the FastAPI middleware on port 8002.

#### `POST /api/chat`

Request body:
```json
{
  "query": "Based on my blood work, what protein target should I set?",
  "health_profile": {
    "weight_kg": 82,
    "body_fat_pct": 18,
    "lbm_kg": 67,
    "ldl_mmol": 3.1,
    "creatinine": 0.9,
    "goal": "muscle_gain"
  },
  "options": {
    "use_rag": true,
    "use_lora": false,
    "top_k": 6,
    "collection_strategy": "auto"
  }
}
```

Response (JSON, or SSE stream):
```json
{
  "request_id": "req_abc123",
  "base_response": {
    "content": "General protein recommendations suggest 1.6–2.2 g/kg...",
    "latency_s": 37.1,
    "prompt_tokens": 417,
    "completion_tokens": 312
  },
  "rag_response": {
    "content": "Based on your creatinine (0.9 mg/dL) and LBM of 67 kg...",
    "latency_s": 58.8,
    "prompt_tokens": 2289,
    "completion_tokens": 535,
    "json_valid": true
  },
  "retrieval": {
    "chunks_retrieved": 6,
    "collections_queried": ["nutrition_guidelines", "public_health_recommendations"],
    "retrieval_latency_s": 0.31,
    "sources": [
      {
        "score": 0.94,
        "collection": "nutrition_guidelines",
        "document_id": "issn_protein_2017_chunk_14",
        "excerpt": "1.4–2.0 g/kg is optimal for muscle protein synthesis...",
        "metadata": { "source": "ISSN Position Stand 2017", "page": 4 }
      }
    ]
  },
  "metrics": {
    "grounding_score": 0.91,
    "personalisation_score": 0.87,
    "base_score": 0.62,
    "rag_improvement": 0.29
  }
}
```

#### `GET /api/health`

Returns status of all downstream services:
```json
{
  "vllm": { "status": "ok", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct" },
  "chromadb": { "status": "ok", "collections": 4, "total_docs": 13349 },
  "lora_adapter": { "status": "loaded", "adapter": "health-v1" }
}
```

#### `GET /api/collections`

Returns ChromaDB collection metadata for the Knowledge Base view.

#### `POST /api/search`

Direct ChromaDB search for the Knowledge Base explorer, bypasses LLM generation.

### Docker Compose Integration

The two new services are added to the existing `docker-compose.yml`:

```yaml
services:
  # --- existing ---
  llm:
    # unchanged
  vector-db:
    # unchanged

  # --- new ---
  api:
    build: ./services/api
    ports:
      - "8002:8002"
    environment:
      - VLLM_URL=http://llm:8000
      - CHROMA_URL=http://vector-db:8001
    depends_on:
      - llm
      - vector-db
    volumes:
      - ./models/adapters:/models/adapters:ro
      - ./rag:/app/rag:ro

  frontend:
    build: ./services/frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://localhost:8002
    depends_on:
      - api
```

### Network & Security

All AI services (ports 8000, 8001, 8002) are bound only to the internal Docker network — they are **not exposed to the public internet**. Only port 3000 (frontend) and port 8002 (API) are mapped to `0.0.0.0` on the EC2 host.

On AWS, use EC2 Security Group rules to restrict access:

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP only | SSH |
| 3000 | TCP | Your IP only | Frontend dashboard |
| 8002 | TCP | Your IP only | API (optional: proxied through frontend) |
| 8000 | TCP | 172.x.x.x (Docker network) | vLLM — internal only |
| 8001 | TCP | 172.x.x.x (Docker network) | ChromaDB — internal only |

For production, add an Nginx reverse proxy container that serves the frontend on port 443 (HTTPS) and proxies `/api/*` to the FastAPI service — removing the need to expose port 8002 directly.

### Implementation Steps

The following steps build the frontend stack in order. Each step is independently testable.

**Step 1 — FastAPI skeleton**

Create `services/api/` with a minimal FastAPI app that proxies requests to vLLM and returns a response. Verify with `curl http://localhost:8002/api/health`.

**Step 2 — Parallel RAG + base calls**

Implement `core/comparator.py`: use `asyncio.gather()` to run the base call and the ChromaDB retrieval + RAG call concurrently. Return both responses with timing metadata.

**Step 3 — Metrics computation**

Implement `core/metrics.py`: grounding score (keyword overlap between RAG response and retrieved chunks), personalisation score (ratio of health-profile-specific terms in the response), JSON validity flag.

**Step 4 — SSE streaming**

Wrap the vLLM streaming endpoint so tokens flow through FastAPI to the browser via Server-Sent Events. This allows the UI to stream the RAG response in real time while the base response is shown as a completed block.

**Step 5 — React scaffold**

Initialise the Vite + React + Tailwind project in `services/frontend/`. Create the sidebar navigation and five view shells (Chat & Compare, Analytics, Knowledge Base, Health Profile, Model Config).

**Step 6 — Chat & Compare view**

Implement `ChatPanel`, `ComparePanel`, `SourcesPanel`, and `MetricsBar` components. Wire to `POST /api/chat` with SSE streaming for the RAG response token stream.

**Step 7 — Analytics view**

Use `recharts` for session-level charts (latency trend, similarity distribution, collection breakdown).

**Step 8 — Knowledge Base explorer**

Implement ChromaDB search via `POST /api/search` and display results with similarity scores and document metadata.

**Step 9 — Docker integration**

Add `api` and `frontend` services to `docker-compose.yml`. Test full stack with `make start`.

**Step 10 — EC2 Security Group**

Restrict inbound rules so only SSH (22) and the dashboard port (3000) are accessible from your IP. All AI service ports remain internal.

### Makefile Additions

```makefile
test-api:          # curl /api/health and print service status
start-frontend:    # docker compose up api frontend -d
logs-frontend:     # docker compose logs -f api frontend
build-frontend:    # docker compose build api frontend
```

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
make start    # Start all services (including api and frontend when implemented)
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

# Verify all ChromaDB collections are reachable
python scripts/test_rag.py --health-check

# Single RAG query (meal plan, gym program, grocery list, lab analysis)
python scripts/test_rag.py --type meal_plan
python scripts/test_rag.py --type gym_program
python scripts/test_rag.py --type lab_analysis

# RAG vs no-RAG side-by-side comparison
python scripts/test_rag.py --type meal_plan --compare
python scripts/test_rag.py --type lab_analysis --compare

# Full evaluation suite (4 test cases × 2 modes → rag/evaluation_results.json)
python scripts/test_rag.py --evaluate
```

### API middleware test (Module 7)

```bash
make test-api

# Or manually:
curl http://localhost:8002/api/health
```

### Manual curl test

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
# 1. Complete Fresh Server Setup Guide above

# 2. Clone the repo
git clone git@github.com:gharaehs/ai-health-orchestration.git
cd ai-health-orchestration

# 3. Download the model (~15 GB)
export HF_TOKEN="hf_your_token_here"
bash scripts/download_model.sh

# 4. Start services
make start

# 5. Transfer corpus files (from original machine)
scp -r ./data/corpus/ user@new-server:~/ai-health-orchestration/data/

# 6. Re-run ingestion
source ~/.venv-rag/bin/activate
python3 scripts/ingest.py

# 7. Test
make test-llm
python scripts/test_rag.py --health-check
make test-api
```

---

## Troubleshooting

### vLLM crashes with CUDA Out of Memory

**Fix:** Ensure `--enforce-eager` is present in the `docker-compose.yml` command section. Already applied in this repo.

### ChromaDB collections empty after restart

**Cause:** Incorrect volume mount destination. ChromaDB writes to `/data` inside the container, not `/chroma/chroma`.

**Fix:** Ensure `docker-compose.yml` has:
```yaml
volumes:
  - ./data/chroma:/data
```
Re-run `python3 scripts/ingest.py` after fixing the mount.

### ChromaDB returns v1 API deprecated error

Expected behavior. ChromaDB 1.5.8 uses v2 API. Always use:
```bash
curl http://localhost:8001/api/v2/heartbeat   # ✅ correct
curl http://localhost:8001/api/v1/heartbeat   # ❌ deprecated
```
The Python client (`chromadb.HttpClient`) handles this automatically.

### RAG query times out

**Cause:** vLLM generating 2,048 tokens on a T4 takes ~3–4 minutes. The default requests timeout of 120s is too short.

**Fix:** Already set to 600s in `rag/pipeline.py`. If you see timeout errors, verify:
```bash
grep "timeout=" ~/ai-health-orchestration/rag/pipeline.py
# Should show: timeout=600
```

### FastAPI cannot reach vLLM or ChromaDB

**Cause:** Using `localhost` inside the `api` container — which only resolves to the container itself, not to sibling containers.

**Fix:** Use Docker Compose service names. In `services/api/core/config.py`:
```python
VLLM_URL = "http://llm:8000"        # not http://localhost:8000
CHROMA_URL = "http://vector-db:8001" # not http://localhost:8001
```
Both services must be on the same Docker network (the default `docker-compose.yml` network handles this automatically).

### Docker build fails with "no space left on device"

**Cause:** The vLLM image (~15 GB) requires temporary build cache space. At 80 GB total with model + corpus + existing image, there is insufficient free space for a rebuild.

**Fix:** Do not rebuild the vLLM image. All runtime configuration is managed via the `command` override in `docker-compose.yml`. Simply edit that file and run `make stop && docker compose up -d`.

### vLLM uses FlashInfer instead of FlashAttention2

**Not an error.** T4 GPU (compute capability 7.5) does not support FlashAttention2 (requires 8.0+). vLLM automatically falls back to FlashInfer, which works correctly on T4.

### Training venv issues

```bash
source ~/.venv-training/bin/activate
python training/train.py
```

If missing, recreate:
```bash
python3 -m venv ~/.venv-training
source ~/.venv-training/bin/activate
pip install "transformers==4.45.2" "peft==0.13.0" "trl==0.11.0" \
            "accelerate==0.34.0" "bitsandbytes==0.43.3" "datasets==2.21.0" \
            "torch==2.5.1" "pyyaml" "scipy" "rich"
```

---

## Module Progress

### ✅ Module 1 — Local LLM Deployment (Complete)

**Completed:**
- EC2 g4dn.xlarge provisioned with Ubuntu 22.04
- NVIDIA drivers (535) + CUDA 12.2 installed and verified
- Docker + Docker Compose + NVIDIA Container Toolkit configured
- vLLM serving Llama 3.1 8B via OpenAI-compatible API on port 8000
- ChromaDB running on port 8001
- CUDA OOM resolved with `--enforce-eager`

**Key learnings:**
- 1B parameters ≈ 2 GB VRAM in FP16
- vLLM's PagedAttention enables efficient multi-request serving
- T4 uses FlashInfer (not FlashAttention2) — correct and expected
- CUDA graph warmup is the OOM culprit on T4; `--enforce-eager` solves it

---

### ✅ Module 2 — Parameter-Efficient Fine-Tuning / LoRA (Complete)

**Completed:**
- Switched from Mistral 7B AWQ to `meta-llama/Meta-Llama-3.1-8B-Instruct` fp16
- Built 21-example health domain training dataset
- QLoRA fine-tuning: Llama 3.1 8B in 4-bit via BitsAndBytes, LoRA (rank=16, alpha=32)
- LoRA adapter trained and saved to `./models/adapters/health-v1/` (~161 MB)
- Evaluation completed: base vs fine-tuned comparison across 5 test prompts

**Training dataset categories:**

| Category | Examples |
|----------|----------|
| Meal Plans | 10 |
| Gym Programs | 5 |
| Grocery Lists | 2 |
| Lab / Blood Test Analysis | 2 |
| Body Composition Assessment | 1 |
| Complete Multi-Output Plans | 1 |

**Key learnings:**
- AWQ is inference-only — incompatible with LoRA training
- BitsAndBytes 4-bit (NF4) is the correct format for QLoRA training
- Single fp16 model serves both training and inference — no separate versions needed
- 128K context window (vs Mistral's 4K) is critical for Module 4 RAG

---

### ✅ Module 3 — Vector Database & RAG Corpus Ingestion (Complete)

**Completed:**
- Curated 4-category domain corpus from authoritative sources
- Built `scripts/ingest.py`: full pipeline from raw files → text extraction → chunking → embedding → ChromaDB storage
- Handles PDF (PyMuPDF), DOCX (python-docx), and CSV (pandas) file types
- 4 isolated ChromaDB collections with category metadata for agent-specific retrieval
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` on CPU
- Total: **13,349 documents** ingested

**Corpus sources by category:**

| Category | Key Sources |
|----------|-------------|
| Public health recommendations | ABIM lab reference ranges, NBME lab values, ACSM/AHA cholesterol guidelines, ESC cardiovascular prevention, ADA diabetes standards, Endocrine Society testosterone guidelines |
| Nutrition guidelines | JISSN protein & nutrient timing position stands, WHO carbohydrate/fat guidelines, DRI tables (RDA/AI/UL for vitamins, minerals, macronutrients), Mediterranean diet meta-analyses |
| Gym programming | ACSM 2026 resistance training position stand, ACSM 2009 progression models, IUSCA hypertrophy position stand, NSCA basics manual, periodization meta-analyses |
| Food & recipes | All_Diets.csv (diet-categorized recipes with macros), USDA Foundation Foods (Dec 2025), USDA SR Legacy (2018) |

**Key learnings:**
- Chunking strategy must vary by document type — sliding window for prose, paragraph-boundary for reference docs, atomic chunks for recipes
- Four isolated collections give agents precise retrieval scope without metadata filtering complexity
- ChromaDB 1.5.8 uses v2 API — `api/v1` endpoint is deprecated
- ChromaDB must be mounted to `/data` inside the container — not `/chroma/chroma`

---

### ✅ Module 4 — Retrieval-Augmented Generation (Complete)

**Completed:**
- Built `rag/` module: retriever, prompt builder, pipeline, and evaluation
- `rag/retriever.py`: ChromaDB query logic with collection routing per output type
- `rag/prompt_builder.py`: two prompt strategies — STRUCTURED (labelled context blocks) and NAIVE (flat concatenation baseline)
- `rag/pipeline.py`: end-to-end orchestration — embed → retrieve → build prompt → call vLLM → parse JSON
- `rag/evaluate.py`: RAG vs. no-RAG evaluation suite across 4 test cases
- `scripts/test_rag.py`: CLI test tool with `--health-check`, `--compare`, `--evaluate` modes
- Context window raised from 8,192 to 16,384 tokens via `docker-compose.yml` command override
- ChromaDB volume mount fixed: `./data/chroma:/data` (was incorrectly `/chroma/chroma`)
- Full end-to-end test passed: retrieval → augmented prompt → structured JSON generation

**RAG pipeline metrics (lab analysis comparison):**

| Metric | With RAG | No RAG |
|--------|----------|--------|
| Latency | 58.8s | 37.1s |
| Prompt tokens (est.) | 2,289 | 417 |
| Chunks retrieved | 6 | 0 |
| JSON parse | ✅ | ✅ |

**Key RAG vs. no-RAG finding — LDL reference range:**
- **RAG:** `"Optimal: < 2.6 mmol/L, Near-optimal: 2.6–3.3, Borderline-high: 3.3–4.1, High: > 4.1"` ← from ingested ABIM/NBME clinical guidelines
- **No-RAG:** `"2.5–3.5 mmol/L"` ← model estimate from training data

The RAG system correctly applied 4-tier clinical classification directly from the ingested corpus. The no-RAG system produced a simplified, less accurate flat range — a medically meaningful difference.

**Key learnings:**
- Retrieval is fast (~0.3s for 10 chunks across 2 collections) — the latency overhead is entirely generation time
- STRUCTURED prompt strategy (labelled context blocks per collection) produces better grounding than naive concatenation
- `max_context_chars=12,000` fits comfortably within the 16,384 token window alongside the health profile, schema, and 2,048-token output budget
- ChromaDB volume mount to the wrong container path causes silent data loss on restart — always verify with `ls data/chroma/` after first ingestion

---

### 🔜 Module 5 — Agentic AI / Multi-Agent Orchestration (Next)

Planned: Lab Analysis Agent, Nutrition Agent, Training Agent, Grocery Agent — each with dedicated ChromaDB collection scope, coordinating via structured intermediate outputs through an orchestration loop.

---

### 🔜 Module 6 — Model Context Protocol / MCP (Planned)

Planned: Expose health data, workout history, and recipe database via MCP servers.

---

### 🔜 Frontend Dashboard (Planned)

See the [Frontend Dashboard — Architecture & Plan](#frontend-dashboard--architecture--plan) section above for full details.

**Planned deliverables:**
- `services/api/` — FastAPI middleware with parallel RAG + base execution and metrics computation
- `services/frontend/` — React + Vite + Tailwind dashboard with five views
- Updated `docker-compose.yml` with `api` and `frontend` services
- Updated `Makefile` with frontend management commands
- EC2 Security Group configuration locked to your IP on port 3000

---

## Design Decisions

### Why Llama 3.1 8B over Mistral 7B?

| Factor | Llama 3.1 8B | Mistral 7B AWQ |
|--------|-------------|----------------|
| Context window | 128K tokens | 4K tokens |
| RAG suitability | ✅ Excellent | ❌ Too short for retrieved docs |
| LoRA training | ✅ fp16 + BitsAndBytes | ❌ AWQ is inference-only |
| Single model train + serve | ✅ Yes | ❌ Requires separate versions |

The 128K context window was the decisive factor — essential for Module 4 RAG where retrieved documents, user health profiles, and conversation history must fit in a single context.

### Why FastAPI + React over Streamlit or Gradio?

| Factor | FastAPI + React | Streamlit / Gradio |
|--------|----------------|-------------------|
| Parallel RAG + base calls | ✅ Native async | ❌ Requires workarounds |
| SSE token streaming | ✅ First-class | ⚠️ Limited |
| Custom analytics UI | ✅ Full control | ❌ Constrained widgets |
| Production extensibility | ✅ Yes (Module 5+) | ❌ Prototype-only |
| Existing stack integration | ✅ Clean Docker service | ⚠️ Separate process |

Streamlit and Gradio are excellent for quick prototypes. For this system, where the UI must expose retrieved sources, per-chunk similarity scores, parallel response comparisons, and session-level analytics charts, a React frontend with a FastAPI middleware gives full control without fighting framework limitations.

### Why a FastAPI middleware instead of the frontend calling vLLM directly?

Direct browser-to-vLLM calls would require exposing port 8000 to the public internet, introducing security risk. The middleware also enables logic that cannot live in the browser: parallel execution of the RAG and base paths with `asyncio.gather()`, metrics computation, and streaming response aggregation.

### Why STRUCTURED over NAIVE prompt strategy?

The STRUCTURED strategy groups retrieved chunks by collection and labels each section (e.g. `[CONTEXT: Clinical Lab Reference Ranges & Health Guidelines]`). This tells the model exactly what each block of text represents and strongly encourages it to use and cite the retrieved data. The NAIVE strategy (flat concatenation) produces lower-quality grounding because the model cannot distinguish the source or authority of each chunk.

### Why four separate ChromaDB collections?

Each agent in the multi-agent architecture (Module 5) queries only the domain relevant to its task. A Lab Analysis Agent should never retrieve a recipe when looking up an LDL threshold. Collection-level isolation enforces this separation cleanly without complex metadata filtering logic in every query.

### Why CPU for embeddings?

The ingestion pipeline and runtime retrieval both use `all-MiniLM-L6-v2` on CPU. This preserves the T4's full 16 GB VRAM for Llama 3.1 inference. The embedding model is fast enough on CPU (~0.3s per query) for the query volumes this system handles.

### Why vLLM over llama.cpp?

| Factor | vLLM (chosen) | llama.cpp |
|--------|--------------|-----------|
| T4 GPU utilization | Excellent | Good but suboptimal |
| Throughput | High (PagedAttention) | Lower |
| OpenAI-compatible API | Built-in | Requires wrapper |
| Best use case | GPU server | CPU / laptop |

### Why ChromaDB over FAISS?

ChromaDB runs as a persistent server with a REST API, accessible from any container in the compose stack. FAISS is a library embedded in application code — unsuitable for a multi-service Docker architecture where multiple agents need independent access to vector storage.

### Why runtime flags in docker-compose.yml instead of the Dockerfile?

The vLLM image is ~15 GB. Rebuilding it on the T4 instance requires temporary build cache space equal to the image size, which exhausts the 80 GB disk. All vLLM configuration is therefore managed via the `command` override in `docker-compose.yml`, which takes effect at container start without a rebuild.

---

## Links

- **GitHub Repository:** https://github.com/gharaehs/ai-health-orchestration
- **Mentorship Program:** AI Technical Deep Dive
- **Model:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **vLLM Documentation:** https://docs.vllm.ai
- **ChromaDB Documentation:** https://docs.trychroma.com

---

*Last updated: Modules 1, 2, 3 & 4 complete — Frontend dashboard planned — May 2026*