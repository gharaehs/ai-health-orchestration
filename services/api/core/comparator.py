import asyncio
import time
import httpx
from sentence_transformers import SentenceTransformer
from core.config import VLLM_URL, CHROMA_URL, VLLM_MODEL, REQUEST_TIMEOUT
from core.metrics import compute_grounding_score, count_tokens_approx

_embedder = None
_collection_ids: dict[str, str] = {}  # name -> UUID

CHROMA_BASE = (
    f"{CHROMA_URL}/api/v2/tenants/default_tenant/databases/default_database"
)

COLLECTION_MAP = {
    "meal_plan":    ["nutrition_guidelines", "food_and_recipes"],
    "gym_program":  ["gym_programming"],
    "lab_analysis": ["public_health_recommendations", "nutrition_guidelines"],
    "general":      ["nutrition_guidelines", "public_health_recommendations",
                     "gym_programming", "food_and_recipes"],
}


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


async def get_collection_ids() -> dict[str, str]:
    """Fetch and cache collection name→UUID mapping from ChromaDB."""
    global _collection_ids
    if _collection_ids:
        return _collection_ids
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{CHROMA_BASE}/collections")
        r.raise_for_status()
        for col in r.json():
            _collection_ids[col["name"]] = col["id"]
    print(f"Cached collection IDs: {_collection_ids}")
    return _collection_ids


async def retrieve_chunks(query: str, collections: list[str],
                           top_k: int = 3) -> list[dict]:
    embedder  = get_embedder()
    embedding = embedder.encode(query, normalize_embeddings=True).tolist()
    col_ids   = await get_collection_ids()

    chunks = []
    async with httpx.AsyncClient(timeout=30) as client:
        for name in collections:
            uuid = col_ids.get(name)
            if not uuid:
                print(f"No UUID found for collection: {name}")
                continue
            try:
                r = await client.post(
                    f"{CHROMA_BASE}/collections/{uuid}/query",
                    json={
                        "query_embeddings": [embedding],
                        "n_results":        top_k,
                        "include":          ["documents", "metadatas", "distances"],
                    },
                )
                data  = r.json()
                docs  = data.get("documents",  [[]])[0]
                dists = data.get("distances",  [[]])[0]
                metas = data.get("metadatas",  [[]])[0]
                for doc, dist, meta in zip(docs, dists, metas):
                    chunks.append({
                        "collection": name,
                        "excerpt":    doc[:400],
                        "score":      round(1 - dist, 3),
                        "metadata":   meta or {},
                    })
            except Exception as e:
                print(f"ChromaDB query error on {name}: {e}")

    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks


async def call_vllm(messages: list[dict], label: str) -> dict:
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(f"{VLLM_URL}/v1/chat/completions", json={
                "model":       VLLM_MODEL,
                "messages":    messages,
                "max_tokens":  1024,
                "temperature": 0.3,
            })
            data    = r.json()
            content = data["choices"][0]["message"]["content"]
            usage   = data.get("usage", {})
            return {
                "content":           content,
                "latency_s":         round(time.perf_counter() - start, 2),
                "prompt_tokens":     usage.get("prompt_tokens",
                                               count_tokens_approx(str(messages))),
                "completion_tokens": usage.get("completion_tokens",
                                               count_tokens_approx(content)),
                "error": None,
            }
    except Exception as e:
        return {
            "content": "", "latency_s": round(time.perf_counter() - start, 2),
            "prompt_tokens": 0, "completion_tokens": 0, "error": str(e),
        }


async def run_comparison(query: str, health_profile: dict,
                          query_type: str = "general",
                          top_k: int = 6) -> dict:
    system_prompt = (
        "You are a health and fitness AI assistant. "
        "Provide specific, evidence-based recommendations. "
        "Be concise and structured."
    )
    user_content  = f"Health profile: {health_profile}\n\nRequest: {query}"
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    collections     = COLLECTION_MAP.get(query_type, COLLECTION_MAP["general"])
    retrieval_start = time.perf_counter()

    chunks, base_result = await asyncio.gather(
        retrieve_chunks(query, collections, top_k),
        call_vllm(base_messages, "base"),
    )
    retrieval_latency = round(time.perf_counter() - retrieval_start, 2)

    context_blocks = []
    for col in collections:
        col_chunks = [c for c in chunks if c["collection"] == col]
        if col_chunks:
            label = col.replace("_", " ").title()
            block = f"[CONTEXT: {label}]\n" + "\n---\n".join(
                c["excerpt"] for c in col_chunks
            )
            context_blocks.append(block)

    rag_context  = "\n\n".join(context_blocks)
    rag_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": (
            f"{rag_context}\n\n"
            f"Health profile: {health_profile}\n\n"
            f"Request: {query}"
        )},
    ]

    rag_result = await call_vllm(rag_messages, "rag")
    grounding  = compute_grounding_score(rag_result["content"], chunks)
    base_score = compute_grounding_score(base_result["content"], chunks)

    return {
        "base_response": base_result,
        "rag_response":  rag_result,
        "retrieval": {
            "chunks_retrieved":    len(chunks),
            "collections_queried": collections,
            "retrieval_latency_s": retrieval_latency,
            "sources":             chunks,
        },
        "metrics": {
            "grounding_score":  grounding,
            "base_score":       base_score,
            "rag_improvement":  round(grounding - base_score, 3),
            "latency_delta_s":  round(
                rag_result["latency_s"] - base_result["latency_s"], 2),
        },
    }
