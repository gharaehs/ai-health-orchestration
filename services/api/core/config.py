import os

CHROMA_URL = "http://vector-db:8000"
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "3"))
