#!/bin/bash
echo "Setting up virtual environment..."
python3 -m venv ~/.venv-hf

echo "Installing huggingface_hub..."
~/.venv-hf/bin/pip install -q huggingface_hub

echo "Downloading Mistral 7B Instruct AWQ..."
~/.venv-hf/bin/hf download \
  TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
  --local-dir ./models/mistral

echo "Done. Model saved to ./models/mistral"
