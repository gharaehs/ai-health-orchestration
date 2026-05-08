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

This system implements a multi-module AI pipeline across 6 engineering modules:

| Module | Topic | Status |
|--------|-------|--------|
| Module 1 | Local LLM Deployment & Quantization | ✅ Complete |
| Module 2 | Parameter-Efficient Fine-Tuning (LoRA) | ✅ Complete |
| Module 3 | Vector Database & RAG Corpus Ingestion | ✅ Complete |
| Module 4 | Retrieval-Augmented Generation (RAG) | ✅ Complete |
| Module 5 | Agentic AI / Multi-Agent Orchestration | 🔜 Next |
| Module 6 | Model Context Protocol (MCP) | 🔜 Planned |

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
│   └── llm/
│       └── Dockerfile              # vLLM container configuration
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
make start    # Start all services
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

---

## Migrating to a New Server

### What lives where

| Data | Location | In Git? | Migrate how? |
|------|----------|---------|--------------|
| Code & config | GitHub | ✅ Yes | `git clone` |
| Docker setup | GitHub | ✅ Yes | `git clone` |
| RAG pipeline | GitHub | ✅ Yes | `git clone` |
| Training scripts & dataset | GitHub | ✅ Yes | `git clone` |
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

## Design Decisions

### Why Llama 3.1 8B over Mistral 7B?

| Factor | Llama 3.1 8B | Mistral 7B AWQ |
|--------|-------------|----------------|
| Context window | 128K tokens | 4K tokens |
| RAG suitability | ✅ Excellent | ❌ Too short for retrieved docs |
| LoRA training | ✅ fp16 + BitsAndBytes | ❌ AWQ is inference-only |
| Single model train + serve | ✅ Yes | ❌ Requires separate versions |

The 128K context window was the decisive factor — essential for Module 4 RAG where retrieved documents, user health profiles, and conversation history must fit in a single context.

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

*Last updated: Modules 1, 2, 3 & 4 complete — May 2026*