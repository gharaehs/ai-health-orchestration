FROM python:3.12-slim
WORKDIR /app
COPY services/api/requirements.txt .

# CPU-only torch — embeddings run on CPU, avoids ~2.5GB of nvidia-cuda-* wheels
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of requirements.txt, skipping torch (already installed above)
RUN grep -v '^torch' requirements.txt > requirements-no-torch.txt \
    && pip install --no-cache-dir -r requirements-no-torch.txt

COPY services/api/ .
COPY agents/ ./agents/
COPY rag/ ./rag/
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
