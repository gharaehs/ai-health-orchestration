# AI Health Orchestration System

> A modular, production-style AI system that ingests structured health data and user goals to generate grounded, structured outputs including weekly meal plans, grocery lists, and gym programs.

Built as part of the **AI Technical Deep Dive** mentorship program at diconium.

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
| Module 3 | Vector Database (FAISS / Chroma) | 🔜 Next |
| Module 4 | Retrieval-Augmented Generation (RAG) | 🔜 Planned |
| Module 5 | Agentic AI / Multi-Agent Orchestration | 🔜 Planned |
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
│  │  Llama 3.1 8B        │    │  Vector Database              │   │
│  │  Instruct (fp16)     │    │  (Semantic Memory)            │   │
│  │  GPU: Tesla T4       │    │  Persistent Storage           │   │
│  └──────────┬───────────┘    └──────────────────────────────┘   │
│             │                                                    │
│             ▼                                                    │
│    ./models/llama/       (mounted volume, not in git)           │
│    ./models/adapters/    (LoRA adapters, not in git)            │
│    ./data/chroma/        (mounted volume, not in git)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:**

```
User Query
    │
    ▼
vLLM API (port 8000)          ← OpenAI-compatible REST API
    │
    ├── Module 3+: Query ChromaDB (port 8001) for relevant context
    │       └── RAG: Augment prompt with retrieved documents
    │
    └── Generate structured JSON response
```

---

## Infrastructure & Hardware

| Component | Specification |
|-----------|--------------|
| **Instance** | AWS EC2 g4dn.xlarge |
| **GPU** | NVIDIA Tesla T4 — 16 GB VRAM |
| **CPU** | 4 vCPUs |
| **RAM** | 16 GB |
| **Storage** | 48 GB SSD (NVMe) |
| **OS** | Ubuntu 22.04 LTS |
| **CUDA** | 12.2 |
| **Driver** | 535.288.01 |

**GPU Memory Usage (Llama 3.1 8B fp16 via vLLM):**

| Component | VRAM Used    |
|-----------|--------------|
| Llama 3.1 8B (bitsandbytes 4-bit) | ~5.65 GB     |
| KV Cache (PagedAttention) | ~6.8 GB      |
| System overhead | ~0.5 GB      |
| Safety buffer (15%) | ~2.1 GB      |
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
│   └── download_model.sh           # Downloads Llama 3.1 8B from HuggingFace
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
    └── chroma/                     # ⚠️ NOT IN GIT — vector DB persistent storage
```

> **Why are models not in git?**
> Model files are 15+ GB. They are excluded from git intentionally. The `scripts/download_model.sh` script recreates them on any new server.

---

## Services

### LLM Service — vLLM + Llama 3.1 8B Instruct

- **Image:** `vllm/vllm-openai:latest`
- **Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (fp16, loaded with on-the-fly float16 quantization)
- **Port:** `8000`
- **API:** OpenAI-compatible (`/v1/chat/completions`, `/v1/models`, etc.)
- **Inference flags:**
  - `--dtype float16` — serves fp16 model efficiently on T4
  - `--max-model-len 8192` — context window (Llama 3.1 supports up to 128K)
  - `--gpu-memory-utilization 0.85` — reserves 15% VRAM as safety buffer
  - `--max-num-seqs 8` — limits concurrent sequences
  - `--enforce-eager` — disables CUDA graph capture (required for T4, prevents OOM)
  - `--quantization bitsandbytes` + `--load-format bitsandbytes` — loads fp16 weights and quantizes to 4-bit on the fly, bringing VRAM usage from ~16 GB to ~5.65 GB

> **Why `--enforce-eager`?**
> The T4 GPU (CUDA compute capability 7.5) struggles with vLLM's default CUDA graph warmup which tries to capture 512 graph sizes simultaneously, causing an OOM crash. `--enforce-eager` disables this. It's slightly slower for high-throughput scenarios but works perfectly for development and single-user inference.

> **Why not AWQ for serving?**
> We switched to Llama 3.1 8B fp16 as the single model for both training and serving. This eliminates the need for separate train/serve model versions. vLLM serves it efficiently with `--dtype float16` on the T4.

### Vector Database — ChromaDB

- **Image:** `chromadb/chroma:latest`
- **Port:** `8001`
- **Storage:** `./data/chroma/` (persisted on disk)
- **Used in:** Module 3+ for semantic retrieval / RAG

---

## Prerequisites

Before running anything, the host server needs:

1. **Ubuntu 22.04 LTS** (tested, recommended)
2. **NVIDIA GPU drivers** (535+) and **CUDA 12.2**
3. **Docker** + **Docker Compose plugin**
4. **NVIDIA Container Toolkit** (GPU access inside Docker)
5. **Git**
6. **Python 3.12 + python3.12-venv** (for model download and training scripts)
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

You should see the Tesla T4 with CUDA 12.2. If drivers are missing:

```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
# After reboot, reconnect via SSH and verify:
nvidia-smi
```

Expected output:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.288.01   Driver Version: 535.288.01   CUDA Version: 12.2              |
| GPU  Name        Persistence-M | GPU Memory Usage |
|   0  Tesla T4         Off      |  16160MiB VRAM   |
+---------------------------------------------------------------------------------------+
```

### Step 3 — Install Docker

```bash
# Install prerequisites
sudo apt install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Allow running Docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker --version
docker compose version
```

### Step 4 — Install NVIDIA Container Toolkit

This bridges Docker and the GPU — **required for vLLM to access the T4**.

```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is accessible inside Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

You should see the same T4 output from inside the container. ✅

### Step 5 — Install Git and Configure GitHub

```bash
sudo apt install -y git

git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Generate SSH key for GitHub
ssh-keygen -t ed25519 -C "your@email.com"
# Press Enter for all prompts

# Copy this and add to: GitHub → Settings → SSH and GPG keys → New SSH key
cat ~/.ssh/id_ed25519.pub
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

> Download takes approximately 20–40 minutes depending on internet speed.
> Requires a HuggingFace account with Meta's Llama 3.1 license accepted.

**What gets downloaded:**

```
models/llama/
├── config.json
├── generation_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model-00001-of-00004.safetensors   ← model weights split across 4 shards (~15 GB total)
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
└── model.safetensors.index.json
```

---

## Running the System

### Start all services

```bash
make start
```

This runs `docker compose up -d --build` and starts:
- vLLM container (builds from `services/llm/Dockerfile`)
- ChromaDB container (pulls from Docker Hub)

**First start takes longer** (~5–10 min) as Docker pulls and builds images. Subsequent starts are fast.

### Watch startup logs

```bash
make logs
```

Wait for this line — it means vLLM is ready:
```
INFO:     Application startup complete.
```

You will also see informational messages like:
```
INFO: T4 does not support FlashAttention2, using FlashInfer backend
INFO: Available KV cache: X GiB
```
These are expected and not errors.

### Stop all services

```bash
make stop
```

### Restart

```bash
make stop && make start
```

---

## Testing & Verification

### Quick test — send a prompt to the LLM

```bash
make test-llm
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

### Check available models

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

### Check running containers

```bash
docker ps
```

Expected output:
```
CONTAINER ID   IMAGE                         PORTS                    NAMES
xxxxxxxxxxxx   ai-health-orchestration-llm   0.0.0.0:8000->8000/tcp   ai-health-orchestration-llm-1
xxxxxxxxxxxx   chromadb/chroma:latest        0.0.0.0:8001->8000/tcp   ai-health-orchestration-vector-db-1
```

### Check GPU memory usage

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

The entire system is designed for fast, reproducible migration.

### What lives where

| Data | Location | In Git? | Migrate how? |
|------|----------|---------|--------------|
| Code & config | GitHub | ✅ Yes | `git clone` |
| Docker setup | GitHub | ✅ Yes | `git clone` |
| Training scripts & dataset | GitHub | ✅ Yes | `git clone` |
| Llama 3.1 model | `./models/llama/` | ❌ No | `bash scripts/download_model.sh` |
| LoRA adapter | `./models/adapters/` | ❌ No | Re-train or `scp` |
| Vector DB data | `./data/chroma/` | ❌ No | `scp` or re-index |

### Migration steps

```bash
# 1. On the new server — complete the Fresh Server Setup Guide above

# 2. Clone the repo
git clone git@github.com:gharaehs/ai-health-orchestration.git
cd ai-health-orchestration

# 3. Download the model (~15 GB)
export HF_TOKEN="hf_your_token_here"
bash scripts/download_model.sh

# 4. Start everything
make start

# 5. Watch logs until ready
make logs

# 6. Test
make test-llm
```

---

## Troubleshooting

### vLLM crashes with CUDA Out of Memory

**Symptom:**
```
torch.OutOfMemoryError: Tried to allocate X MiB. GPU 0 has Y MiB free
```

**Fix:** Ensure `--enforce-eager` is in `services/llm/Dockerfile`. Already applied in this repo.

---

### Docker: No space left on device

**Symptom:**
```
failed to register layer: write ...: no space left on device
```

**Fix:** Free up disk space:
```bash
docker system prune -af
rm -rf /home/ubuntu/.local/lib/   # removes old system-wide pip packages if present
df -h
```

---

### vLLM uses FlashInfer instead of FlashAttention2

This is **not an error**. The T4 GPU (compute capability 7.5) does not support FlashAttention2 which requires compute capability 8.0+. vLLM automatically falls back to FlashInfer, which works correctly on T4.

---

### Training venv issues

Always activate the training venv before running training scripts:
```bash
source ~/.venv-training/bin/activate
python training/train.py
```

If the venv is missing, recreate it:
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
- Docker + Docker Compose installed
- NVIDIA Container Toolkit configured
- GitHub repository created with portable structure
- vLLM serving via OpenAI-compatible API on port 8000
- ChromaDB running on port 8001 (ready for Module 3)
- CUDA OOM issue resolved with `--enforce-eager` flag
- Full test passed: LLM responding to prompts

**Key learnings:**
- 1B parameters ≈ 2 GB VRAM in FP16
- vLLM's PagedAttention enables efficient multi-request serving
- T4 uses FlashInfer (not FlashAttention2) — correct and expected
- CUDA graph warmup is the key OOM culprit on T4; `--enforce-eager` solves it

---

### ✅ Module 2 — Parameter-Efficient Fine-Tuning / LoRA (Complete)

**Completed:**
- Switched from Mistral 7B AWQ to `meta-llama/Meta-Llama-3.1-8B-Instruct` fp16 — single model for both training and serving
- Built 21-example health domain training dataset covering meal plans, gym programs, grocery lists, blood test analysis, body composition, and complete multi-output plans
- QLoRA fine-tuning pipeline: Llama 3.1 8B loaded in 4-bit via BitsAndBytes, LoRA adapters applied (rank=16, alpha=32)
- LoRA adapter trained and saved to `./models/adapters/health-v1/` (~161 MB)
- Evaluation completed: base vs fine-tuned comparison across 5 test prompts
- JSON consistency improvement confirmed on blood test analysis output

**Key learnings:**
- AWQ quantization is inference-only — incompatible with LoRA training (cannot dequantize for gradient computation)
- BitsAndBytes 4-bit (NF4) is the correct quantization format for QLoRA training
- A single fp16 model can serve both training (via BitsAndBytes) and inference (via vLLM) — no separate model versions needed
- LoRA adapters are model-specific but the training dataset is a permanent, reusable asset
- 21 training examples is sufficient to demonstrate the technique; production fine-tuning benefits from hundreds of examples
- Llama 3.1's 128K context window (vs Mistral's 4K) is critical for Module 4 RAG use cases

**Training dataset categories:**

| Category | Examples |
|----------|----------|
| Meal Plans | 10 |
| Gym Programs | 5 |
| Grocery Lists | 2 |
| Lab / Blood Test Analysis | 2 |
| Body Composition Assessment | 1 |
| Complete Multi-Output Plans | 1 |

---

### 🔜 Module 3 — Vector Database (Next)

Planned: Ingest curated nutrition and gym programming knowledge into ChromaDB, implement semantic chunking and embedding generation.

---

### 🔜 Module 4 — Retrieval-Augmented Generation (Planned)

Planned: Connect ChromaDB retrieval to the LLM pipeline, grounding health plan generation in retrieved domain knowledge.

---

### 🔜 Module 5 — Agentic AI / Multi-Agent Orchestration (Planned)

Planned: Lab Analysis Agent, Nutrition Agent, Training Agent, Grocery Agent — coordinating via structured intermediate outputs.

---

### 🔜 Module 6 — Model Context Protocol / MCP (Planned)

Planned: Expose health data, workout history, and recipe database via MCP servers.

---

## Design Decisions

### Why Llama 3.1 8B Instruct over Mistral 7B?

| Factor | Llama 3.1 8B | Mistral 7B AWQ |
|--------|-------------|----------------|
| Context window | 128K tokens | 4K tokens |
| RAG suitability (Module 4) | ✅ Excellent | ❌ Too short for retrieved docs |
| LoRA training compatible | ✅ fp16 + BitsAndBytes | ❌ AWQ is inference-only |
| Single model for train + serve | ✅ Yes | ❌ Requires separate versions |
| JSON structured output | Excellent | Good |
| Community support | Massive | Good |

The 128K context window was the decisive factor — essential for Module 4 RAG where retrieved documents, user health profiles, and conversation history must fit in a single context.

### Why fp16 instead of AWQ for serving?

AWQ is inference-optimized and would be slightly more VRAM-efficient. However, using the same fp16 model for both training and serving eliminates the need to maintain two separate model versions and simplifies the entire pipeline. vLLM handles fp16 efficiently with `--dtype float16` on the T4.

### Why a single model for training and serving?

One fp16 download serves both purposes: BitsAndBytes loads it in 4-bit for training (freeing VRAM for gradients), and vLLM loads it in float16 for serving. The LoRA adapter produced during training can be loaded on top at inference time. This is cleaner, cheaper to maintain, and simpler to upgrade when a better model is released.

### Why vLLM over llama.cpp?

| Factor | vLLM (chosen) | llama.cpp |
|--------|--------------|-----------|
| T4 GPU utilization | Excellent | Good but suboptimal |
| Throughput | High (PagedAttention) | Lower |
| OpenAI-compatible API | Built-in | Requires wrapper |
| Docker support | First-class | More setup |
| Best use case | GPU server | CPU / laptop |

### Why models outside the Docker image?

Baking the 15 GB model into the Docker image would make it extremely large, slow to build, and impossible to push. Mounting it as a volume keeps the image lightweight and the model reusable across container rebuilds.

### Why ChromaDB over FAISS?

ChromaDB offers a persistent server with a REST API, making it accessible across all services in the compose stack. FAISS is a library (not a server) better suited for embedding into application code directly. ChromaDB is the right choice for the multi-service architecture here.

---

## Links

- **GitHub Repository:** https://github.com/gharaehs/ai-health-orchestration
- **Mentorship Program:** AI Technical Deep Dive — diconium
- **Model:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
- **vLLM Documentation:** https://docs.vllm.ai
- **ChromaDB Documentation:** https://docs.trychroma.com

---

*Last updated: Modules 1 & 2 complete — March 2026*