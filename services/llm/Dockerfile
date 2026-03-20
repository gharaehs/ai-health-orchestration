FROM vllm/vllm-openai:latest

# Mount point for the model (loaded at runtime via volume)
# Model: meta-llama/Meta-Llama-3.1-8B-Instruct (fp16)
# Located at: ./models/llama/ on the host

ENV MODEL_PATH=/models/llama

CMD ["--model", "/models/llama", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.85", \
     "--max-num-seqs", "8", \
     "--enforce-eager", \
     "--dtype", "float16"]