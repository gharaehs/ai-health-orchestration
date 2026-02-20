FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Default command - model name passed via environment variable
CMD ["--model", "/models/mistral", \
     "--quantization", "awq", \
     "--max-model-len", "4096", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
