#!/bin/bash
# Downloads Mistral 7B AWQ quantized model (fits well in T4 16GB)
# AWQ quantized = ~4GB, leaves plenty of VRAM headroom

echo "Installing huggingface_hub..."
pip install -q huggingface_hub

echo "Downloading Mistral 7B Instruct AWQ..."
huggingface-cli download \
  TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
  --local-dir ./models/mistral \
  --local-dir-use-symlinks False

echo "Done. Model saved to ./models/mistral"
