#!/bin/bash
# ============================================================
# Download Llama 3.1 8B Instruct (fp16)
# AI Health Orchestration System
# ============================================================
#
# Usage:
#   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
#   bash scripts/download_model.sh
#
# Requirements:
#   - HuggingFace account with Meta Llama 3.1 license accepted
#   - HF_TOKEN environment variable set
#   - ~16GB free disk space

set -e

MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR="./models/llama"

# Check token
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "Run: export HF_TOKEN=\"hf_your_token_here\""
    exit 1
fi

echo "============================================================"
echo "Downloading: $MODEL_ID"
echo "Destination: $MODEL_DIR"
echo "============================================================"

# Create model directory
mkdir -p "$MODEL_DIR"

# Set up Python venv for download
VENV_PATH="$HOME/.venv-hf"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating Python venv at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"
pip install -q --upgrade huggingface_hub

echo "Starting download (~16GB, this will take 10-30 minutes)..."

python3 - <<EOF
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$MODEL_DIR",
    token=os.environ["HF_TOKEN"],
    ignore_patterns=["*.pt", "original/*"],  # Skip redundant formats
)
print("Download complete!")
EOF

echo ""
echo "============================================================"
echo "Model downloaded to: $MODEL_DIR"
echo "============================================================"
ls -lh "$MODEL_DIR"