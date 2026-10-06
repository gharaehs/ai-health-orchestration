import os

# LLM settings (LLM_PROVIDER, BEDROCK_MODEL_ID, AWS_REGION, VLLM_URL, VLLM_MODEL,
# LLM_TIMEOUT) are read by agents/llm_client.py.
CHROMA_URL = os.environ.get("CHROMA_URL", "http://vector-db:8000")
