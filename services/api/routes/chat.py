from fastapi import APIRouter
from pydantic import BaseModel
from core.comparator import run_comparison

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    health_profile: dict = {}
    query_type: str = "general"
    top_k: int = 6


@router.post("/chat")
async def chat(req: ChatRequest):
    result = await run_comparison(
        query=req.query,
        health_profile=req.health_profile,
        query_type=req.query_type,
        top_k=req.top_k,
    )
    return result
