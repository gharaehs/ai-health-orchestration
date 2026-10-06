from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import httpx
from agents import llm_client
from core.config import CHROMA_URL
from routes.chat import router as chat_router
from routes.search import router as search_router
from routes.orchestrate import router as orchestrate_router

import logging
logging.basicConfig(level=logging.INFO)



app = FastAPI(title="Health AI Orchestration API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(orchestrate_router, prefix="/api")


@app.get("/api/health")
async def health():
    status = {}
    status["llm"] = await asyncio.to_thread(llm_client.health)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{CHROMA_URL}/api/v2/heartbeat")
            r.raise_for_status()
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(
                f"{CHROMA_URL}/api/v2/tenants/default_tenant"
                f"/databases/default_database/collections"
            )
            collections = r.json()
        status["chromadb"] = {"status": "ok", "collections": len(collections)}
    except Exception as e:
        status["chromadb"] = {"status": "error", "detail": str(e)}

    overall = "ok" if all(v["status"] == "ok" for v in status.values()) else "degraded"
    return {"status": overall, "services": status}