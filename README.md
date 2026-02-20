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
| Module 2 | Parameter-Efficient Fine-Tuning (LoRA) | 🔜 Next |
| Module 3 | Vector Database (FAISS / Chroma) | 🔜 Planned |
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
│  │                      │    │                              │   │
│  │  Mistral 7B Instruct │    │  Vector Database             │   │
│  │  AWQ (4-bit quant.)  │    │  (Semantic Memory)           │   │
│  │  GPU: Tesla T4       │    │  Persistent Storage          │   │
│  └──────────┬───────────┘    └──────────────────────────────┘   │
│             │                                                    │
│             ▼                                                    │
│    ./models/mistral/     (mounted volume, not in git)           │
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

**GPU Memory Usage:**

| Component | VRAM Used |
|-----------|-----------|
| Mistral 7B AWQ (4-bit) | ~3.88 GB |
| KV Cache (PagedAttention) | ~8.81 GB |
| System overhead | ~0.5 GB |
| Safety buffer (15%) | ~2.31 GB |
| **Total available** | **15.36 GB** |

---

## Repository Structure

```
ai-health-orchestration/
│
├── docker-compose.yml          # Orchestrates all services
├── Makefile                    # Command interface (start, stop, test, etc.)
├── .gitignore
├── README.md
│
├── services/
│   └── llm/
│       └── Dockerfile          # vLLM container configuration
│
├── scripts/
│   └── download_model.sh       # Downloads Mistral 7B from HuggingFace
│
├── models/                     # ⚠️ NOT IN GIT — downloaded locally
│   └── mistral/                # Mistral 7B Instruct AWQ weights (~4.15 GB)
│
└── data/
    └── chroma/                 # ⚠️ NOT IN GIT — vector DB persistent storage
```

> **Why are models not in git?**
> Model files are 4+ GB each. They are excluded from git intentionally. The `scripts/download_model.sh` script recreates them on any new server in ~10 minutes.

---

## Services

### LLM Service — vLLM + Mistral 7B

- **Image:** `vllm/vllm-openai:latest`
- **Model:** `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` (4-bit AWQ quantization)
- **Port:** `8000`
- **API:** OpenAI-compatible (`/v1/chat/completions`, `/v1/models`, etc.)
- **Inference flags:**
  - `--quantization awq` — enables 4-bit AWQ inference
  - `--max-model-len 4096` — maximum context window
  - `--gpu-memory-utilization 0.85` — reserves 15% VRAM as safety buffer
  - `--max-num-seqs 8` — limits concurrent sequences
  - `--enforce-eager` — disables CUDA graph capture (required for T4, prevents OOM)

> **Why `--enforce-eager`?**
> The T4 GPU (CUDA compute capability 7.5) struggles with vLLM's default CUDA graph warmup which tries to capture 512 graph sizes simultaneously, causing an OOM crash. `--enforce-eager` disables this. It's slightly slower for high-throughput scenarios but works perfectly for development and single-user inference.

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
6. **Python 3.12 + python3.12-venv** (for the model download script only)

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

### Step 6 — Install Python venv (for model download script)

```bash
sudo apt install -y python3.12-venv
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
make download-model
```

This script will:
1. Create a Python virtual environment at `~/.venv-hf`
2. Install `huggingface_hub`
3. Download `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` (~4.15 GB) to `./models/mistral/`

> Download takes approximately 10–20 minutes depending on internet speed.
> The model is saved locally so container restarts do not re-download it.

**What gets downloaded:**

```
models/mistral/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── model.safetensors          ← main model weights (4.15 GB, AWQ 4-bit)
└── quant_config.json
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
INFO: Available KV cache: 8.81 GiB / 72,176 tokens
INFO: GPU KV cache blocks: 568
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

This sends a `POST /v1/chat/completions` request to the local vLLM server. Expected response:

```json
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "model": "/models/mistral",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello there! It's nice to meet you. How can I assist you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 19,
        "total_tokens": 31
    }
}
```

### Manual curl test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/mistral",
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

With vLLM running you should see ~13 GB allocated (model + KV cache):
```
|   0  Tesla T4   Off  | ...  | 13000MiB / 15360MiB | GPU-Util % |
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make start` | Build and start all services in detached mode |
| `make stop` | Stop and remove all containers |
| `make logs` | Follow live logs from all containers |
| `make download-model` | Download Mistral 7B AWQ to `./models/mistral/` |
| `make test-llm` | Send a test prompt and display JSON response |

---

## Migrating to a New Server

The entire system is designed for fast, reproducible migration.

### What lives where

| Data | Location | In Git? | Migrate how? |
|------|----------|---------|--------------|
| Code & config | GitHub | ✅ Yes | `git clone` |
| Docker setup | GitHub | ✅ Yes | `git clone` |
| Mistral model | `./models/mistral/` | ❌ No | `make download-model` |
| Vector DB data | `./data/chroma/` | ❌ No | `scp` or re-index |

### Migration steps (10–15 minutes)

```bash
# 1. On the new server — complete the Fresh Server Setup Guide above

# 2. Clone the repo
git clone git@github.com:gharaehs/ai-health-orchestration.git
cd ai-health-orchestration

# 3. Download the model (~4.15 GB)
make download-model

# 4. Start everything
make start

# 5. Watch logs until ready
make logs

# 6. Test
make test-llm
```

That's it. The API will be serving on port 8000 exactly as before.

> **ChromaDB data:** If you have indexed documents in ChromaDB that you want to keep, copy the `./data/chroma/` folder to the new server before starting:
> ```bash
> scp -r ./data/chroma/ user@new-server:~/ai-health-orchestration/data/
> ```
> Otherwise the vector DB starts empty and you re-index your documents.

---

## Troubleshooting

### vLLM crashes with CUDA Out of Memory

**Symptom:**
```
torch.OutOfMemoryError: Tried to allocate 224.00 MiB. GPU 0 has 219.56 MiB free
```

**Cause:** vLLM's default CUDA graph warmup captures 512 graph sizes simultaneously, which exhausts T4 VRAM.

**Fix:** Ensure `--enforce-eager` is in `services/llm/Dockerfile`. This disables CUDA graph capture entirely. Already applied in this repo.

---

### Docker: No space left on device

**Symptom:**
```
failed to register layer: write ...: no space left on device
```

**Fix:** Free up disk space. Remove unused Docker images and containers:

```bash
docker system prune -af
df -h  # verify space is free
```

If Ollama or other large services were previously installed:
```bash
sudo systemctl stop ollama
sudo systemctl disable ollama
sudo rm /usr/local/bin/ollama
sudo rm -rf /usr/share/ollama
sudo systemctl daemon-reload
```

---

### `huggingface-cli: command not found` during model download

**Fix:** The download script uses the `hf` binary (not `huggingface-cli`). The correct script at `scripts/download_model.sh` already uses `~/.venv-hf/bin/hf`. If you see this error, run:

```bash
cat scripts/download_model.sh  # verify it uses ~/.venv-hf/bin/hf
make download-model
```

---

### `python3 -m venv` fails

**Symptom:**
```
The virtual environment was not created successfully because ensurepip is not available.
```

**Fix:**
```bash
sudo apt install -y python3.12-venv
make download-model
```

---

### Docker can't see the GPU

**Symptom:**
```
Error: could not select device driver "" with capabilities: [[gpu]]
```

**Fix:** Reinstall the NVIDIA Container Toolkit and restart Docker:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### vLLM uses FlashInfer instead of FlashAttention2

This is **not an error**. The T4 GPU (compute capability 7.5) does not support FlashAttention2 which requires compute capability 8.0+. vLLM automatically falls back to FlashInfer, which works correctly on T4.

---

### docker-compose.yml `version` warning

**Symptom:**
```
WARN: the attribute `version` is obsolete, it will be ignored
```

**Fix:** This is just a deprecation warning from newer Docker versions and does not affect functionality. Harmless.

---

## Module Progress

### ✅ Module 1 — Local LLM Deployment (Complete)

**Completed:**
- EC2 g4dn.xlarge provisioned with Ubuntu 22.04
- NVIDIA drivers (535) + CUDA 12.2 installed and verified
- Docker + Docker Compose installed
- NVIDIA Container Toolkit configured
- GitHub repository created with portable structure
- Mistral 7B Instruct AWQ downloaded (4.15 GB, 4-bit quantization)
- vLLM serving via OpenAI-compatible API on port 8000
- ChromaDB running on port 8001 (ready for Module 3)
- CUDA OOM issue resolved with `--enforce-eager` flag
- Full test passed: LLM responding to prompts

**Key learnings:**
- 1B parameters ≈ 2 GB VRAM in FP16; AWQ 4-bit reduces this by ~4×
- Mistral 7B in AWQ uses only 3.88 GB, leaving 11+ GB for KV cache
- vLLM's PagedAttention enables efficient multi-request serving
- T4 uses FlashInfer (not FlashAttention2) — this is correct and expected
- CUDA graph warmup is the key OOM culprit on T4; `--enforce-eager` solves it

---

### 🔜 Module 2 — Fine-Tuning with LoRA

Next steps: LoRA / QLoRA adapter training to specialize Mistral for structured health data output (JSON meal plans, gym programs).

---

## Design Decisions

### Why vLLM over llama.cpp?

| Factor | vLLM (chosen) | llama.cpp |
|--------|--------------|-----------|
| T4 GPU utilization | Excellent | Good but suboptimal |
| Throughput | High (PagedAttention) | Lower |
| OpenAI-compatible API | Built-in | Requires wrapper |
| Docker support | First-class | More setup |
| Best use case | GPU server | CPU / laptop |

vLLM is purpose-built for GPU servers like our T4. llama.cpp is best for CPU-only or laptop environments.

### Why AWQ quantization?

AWQ (Activation-Aware Weight Quantization) 4-bit reduces Mistral 7B from ~14 GB (FP16) to ~3.88 GB, fitting easily on the T4's 16 GB VRAM while preserving near-full accuracy. This leaves ~11 GB for the KV cache, enabling long contexts and multiple concurrent requests.

### Why models outside the Docker image?

Baking the 4 GB model into the Docker image would make it ~12 GB, slow to build and push. Mounting it as a volume keeps the image lightweight and the model reusable across container rebuilds.

### Why ChromaDB over FAISS?

ChromaDB offers a persistent server with a REST API, making it accessible across all services in the compose stack. FAISS is a library (not a server) better suited for embedding into application code directly. ChromaDB is the right choice for the multi-service architecture here. FAISS will be explored in Module 3 for direct embedding use cases.

---

## Links

- **GitHub Repository:** https://github.com/gharaehs/ai-health-orchestration
- **Mentorship Program:** AI Technical Deep Dive — diconium
- **Model Source:** https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-AWQ
- **vLLM Documentation:** https://docs.vllm.ai
- **ChromaDB Documentation:** https://docs.trychroma.com

---

*Last updated: Module 1 complete — February 2026*
