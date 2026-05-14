from fastapi import APIRouter
from pydantic import BaseModel
from core.comparator import retrieve_chunks

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    collections: list[str] = [
        "nutrition_guidelines", "public_health_recommendations",
        "gym_programming", "food_and_recipes"
    ]

@router.post("/search")
async def search(req: SearchRequest):
    chunks = await retrieve_chunks(req.query, req.collections, req.top_k)
    return {"sources": chunks, "total": len(chunks)}
