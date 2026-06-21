"""
services/mcp-rag/mcp_server.py

MCP server wrapping HealthRetriever so the RAG corpus can be queried
via the Model Context Protocol.
"""

import logging
import os

from mcp.server.fastmcp import FastMCP

from rag.retriever import HealthRetriever, RetrievedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Internal Docker network — NOT the 8001 host-mapped port.
CHROMA_HOST = os.environ.get("CHROMA_HOST", "vector-db")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

mcp = FastMCP("health-rag-corpus", host="0.0.0.0", port=8004)

# Loaded once at startup — this is where the embedding model load cost lives.
retriever = HealthRetriever(chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT)


def _serialize(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "content": c.content,
            "collection": c.collection,
            "source": c.source,
            "category": c.category,
            "distance": c.distance,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]


@mcp.tool()
def retrieve_context(
    query_text: str,
    query_type: str = "full",
    n_results_per_collection: int = 4,
    max_distance: float = 1.0,
) -> list[dict]:
    """
    Retrieve relevant context for a health query, routed to the right
    collection(s). query_type: nutrition, food, lab, gym, meal_plan,
    gym_program, lab_analysis, grocery, full.
    """
    chunks = retriever.retrieve(
        query_text=query_text,
        query_type=query_type,
        n_results_per_collection=n_results_per_collection,
        max_distance=max_distance,
    )
    return _serialize(chunks)


@mcp.tool()
def search_nutrition_guidelines(query: str, top_k: int = 5) -> list[dict]:
    """Search the nutrition_guidelines collection directly."""
    return _serialize(retriever.query_collection("nutrition_guidelines", query, top_k))


@mcp.tool()
def search_food_and_recipes(query: str, top_k: int = 5) -> list[dict]:
    """Search the food_and_recipes collection directly."""
    return _serialize(retriever.query_collection("food_and_recipes", query, top_k))


@mcp.tool()
def search_gym_programming(query: str, top_k: int = 5) -> list[dict]:
    """Search the gym_programming collection directly."""
    return _serialize(retriever.query_collection("gym_programming", query, top_k))


@mcp.tool()
def search_public_health_recommendations(query: str, top_k: int = 5) -> list[dict]:
    """Search the public_health_recommendations collection directly."""
    return _serialize(retriever.query_collection("public_health_recommendations", query, top_k))


@mcp.tool()
def corpus_health_check() -> dict:
    """Return document counts per collection."""
    return retriever.health_check()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")