"""
rag/retriever.py

Handles all ChromaDB query logic for the RAG pipeline.
Uses sentence-transformers/all-MiniLM-L6-v2 on CPU — same model used during ingestion.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Collection routing ────────────────────────────────────────────────────────
# Maps query intent to the appropriate ChromaDB collection.
# Each agent in Module 5 will use a specific collection; the RAG pipeline
# can query one or multiple collections depending on the request type.

COLLECTION_MAP = {
    "nutrition":    "nutrition_guidelines",
    "food":         "food_and_recipes",
    "lab":          "public_health_recommendations",
    "gym":          "gym_programming",
    "meal_plan":    ["nutrition_guidelines", "food_and_recipes"],
    "gym_program":  ["gym_programming"],
    "lab_analysis": ["public_health_recommendations"],
    "grocery":      ["food_and_recipes"],
    "full":         ["nutrition_guidelines", "food_and_recipes",
                     "gym_programming", "public_health_recommendations"],
}

ALL_COLLECTIONS = [
    "nutrition_guidelines",
    "food_and_recipes",
    "gym_programming",
    "public_health_recommendations",
]


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its metadata and relevance score."""
    content: str
    collection: str
    source: str
    category: str
    distance: float     # Lower = more similar (cosine distance)
    chunk_index: int


class HealthRetriever:
    """
    Retrieves semantically relevant chunks from ChromaDB for health queries.

    Mirrors the ingestion setup exactly:
    - Same embedding model: all-MiniLM-L6-v2
    - Same distance metric: cosine (ChromaDB default)
    - Same CPU execution (preserves T4 VRAM for vLLM)
    """

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8001,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        logger.info("Initialising HealthRetriever...")

        # ChromaDB client
        self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

        # Embedding model — must match ingestion
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # Cache collection handles to avoid repeated lookups
        self._collections: dict = {}
        self._load_collections()

        logger.info("HealthRetriever ready.")

    def _load_collections(self):
        """Load and cache all collection handles from ChromaDB."""
        for name in ALL_COLLECTIONS:
            try:
                self._collections[name] = self.client.get_collection(name)
                count = self._collections[name].count()
                logger.info(f"  Collection '{name}': {count} documents")
            except Exception as e:
                logger.warning(f"  Collection '{name}' not found: {e}")

    def _embed(self, text: str) -> list[float]:
        """Embed a query string using the same model as ingestion."""
        return self.embedder.encode(text, convert_to_numpy=True).tolist()

    def query_collection(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Query a single ChromaDB collection for relevant chunks.

        Args:
            collection_name: One of the 4 defined collection names.
            query_text:       Natural language query to embed and search.
            n_results:        Number of top chunks to return.

        Returns:
            List of RetrievedChunk objects, sorted by relevance (closest first).
        """
        if collection_name not in self._collections:
            logger.warning(f"Collection '{collection_name}' not available, skipping.")
            return []

        col = self._collections[collection_name]
        query_embedding = self._embed(query_text)

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, col.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            chunks.append(RetrievedChunk(
                content=doc,
                collection=collection_name,
                source=meta.get("source", "unknown"),
                category=meta.get("category", "unknown"),
                distance=dist,
                chunk_index=meta.get("chunk_index", 0),
            ))

        return chunks

    def retrieve(
        self,
        query_text: str,
        query_type: str = "full",
        n_results_per_collection: int = 4,
        max_distance: float = 1.0,
    ) -> list[RetrievedChunk]:
        """
        Main retrieval entry point. Routes to one or more collections based on
        query_type, then merges and deduplicates results.

        Args:
            query_text:                 The health query to retrieve context for.
            query_type:                 Key in COLLECTION_MAP — controls which
                                        collection(s) to search.
            n_results_per_collection:   Top-k per collection.
            max_distance:               Distance threshold — chunks above this are
                                        filtered out as irrelevant (0.0 = identical,
                                        1.0 = unrelated; cosine distance).

        Returns:
            Merged list of RetrievedChunk objects, sorted by distance ascending.
        """
        target = COLLECTION_MAP.get(query_type, ALL_COLLECTIONS)
        if isinstance(target, str):
            target = [target]

        all_chunks: list[RetrievedChunk] = []

        for collection_name in target:
            chunks = self.query_collection(
                collection_name=collection_name,
                query_text=query_text,
                n_results=n_results_per_collection,
            )
            # Filter by distance threshold
            chunks = [c for c in chunks if c.distance <= max_distance]
            all_chunks.extend(chunks)

        # Sort globally by distance (best first), deduplicate by content
        seen_content: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for chunk in sorted(all_chunks, key=lambda c: c.distance):
            fingerprint = chunk.content[:100]   # first 100 chars as dedup key
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                unique_chunks.append(chunk)

        logger.info(
            f"Retrieved {len(unique_chunks)} unique chunks "
            f"from {len(target)} collection(s) for query_type='{query_type}'"
        )
        return unique_chunks

    def health_check(self) -> dict:
        """Return collection stats — useful for debugging."""
        stats = {}
        for name, col in self._collections.items():
            stats[name] = col.count()
        return stats