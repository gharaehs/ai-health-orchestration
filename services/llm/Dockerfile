FROM vllm/vllm-openai:latest

WORKDIR /app

CMD ["--model", "/models/mistral", \
     "--quantization", "awq", \
     "--max-model-len", "4096", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--gpu-memory-utilization", "0.85", \
     "--max-num-seqs", "8", \
     "--enforce-eager"]
