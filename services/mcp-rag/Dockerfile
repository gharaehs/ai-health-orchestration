FROM python:3.12-slim

WORKDIR /app

COPY services/api/requirements.txt requirements.txt

# CPU-only torch — this container has no GPU reservation, avoids ~2.5GB of nvidia-cuda-* wheels
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Install the rest of requirements.txt, skipping torch (already installed above)
RUN grep -v '^torch' requirements.txt > requirements-no-torch.txt \
    && pip install --no-cache-dir -r requirements-no-torch.txt

RUN pip install --no-cache-dir "mcp[cli]"

COPY rag/ ./rag/
COPY services/mcp-rag/mcp_server.py .

EXPOSE 8004
CMD ["python", "mcp_server.py"]