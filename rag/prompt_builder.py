"""
rag/prompt_builder.py

Assembles augmented prompts by injecting retrieved context into health queries.

Two strategies are implemented and can be compared during evaluation:
  - STRUCTURED: Context is formatted in clearly labelled sections per collection.
                The model is explicitly told what each section contains.
                This is the recommended strategy for health domain queries.

  - NAIVE:      All retrieved chunks are concatenated as plain text before the
                query. Used as the baseline for RAG vs. no-RAG comparison.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rag.retriever import RetrievedChunk


class PromptStrategy(Enum):
    STRUCTURED = "structured"   # Labelled context blocks (recommended)
    NAIVE      = "naive"        # Plain text concatenation (baseline)


# ── Collection → human-readable label ────────────────────────────────────────
COLLECTION_LABELS = {
    "nutrition_guidelines":          "Nutrition Science & Dietary Guidelines",
    "food_and_recipes":              "Food Data & Recipes",
    "gym_programming":               "Exercise Science & Training Programming",
    "public_health_recommendations": "Clinical Lab Reference Ranges & Health Guidelines",
}

# ── System prompt ─────────────────────────────────────────────────────────────
# Instructs Llama 3.1 to act as a health AI system that grounds its responses
# in the provided scientific context, and to output valid JSON.

SYSTEM_PROMPT = """You are an AI health assistant integrated into a local, \
privacy-preserving health orchestration system.

Your role is to generate evidence-based, structured health plans by combining:
1. The user's personal health data and goals
2. The scientific context provided in the [CONTEXT] sections below

Output rules:
- Always respond with valid, parseable JSON.
- Base nutritional recommendations on the provided guidelines and USDA food data.
- Base training recommendations on the provided exercise science literature.
- If a lab value is provided, reference the clinical thresholds from the context.
- Do not hallucinate nutrients, macros, or clinical values — use the retrieved data.
- If retrieved context does not cover a specific detail, use conservative, \
evidence-based defaults and note the assumption."""


@dataclass
class BuiltPrompt:
    """The assembled prompt ready to send to vLLM."""
    system:    str          # System prompt
    user:      str          # User message (query + context)
    strategy:  str          # Which strategy was used
    n_chunks:  int          # How many context chunks were injected
    token_estimate: int     # Rough token count (chars / 4)


class HealthPromptBuilder:
    """
    Builds augmented prompts for the health RAG pipeline.

    Usage:
        builder = HealthPromptBuilder()
        prompt  = builder.build(query, retrieved_chunks, strategy=PromptStrategy.STRUCTURED)
        # → prompt.system, prompt.user ready for vLLM
    """

    def __init__(self, max_context_chars: int = 12_000):
        """
        Args:
            max_context_chars: Hard cap on total retrieved context injected into
                               the prompt. At ~4 chars/token this is ~3,000 tokens
                               of context — safe within the 8,192 context window
                               used in vLLM, leaving room for the query and output.
                               Llama 3.1's true limit is 128K, so this can be raised
                               aggressively if needed.
        """
        self.max_context_chars = max_context_chars

    def build(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        strategy: PromptStrategy = PromptStrategy.STRUCTURED,
        extra_system_note: Optional[str] = None,
    ) -> BuiltPrompt:
        """
        Assemble the augmented prompt.

        Args:
            query:             The user's health request (profile + goal).
            chunks:            Retrieved chunks from ChromaDB.
            strategy:          STRUCTURED (recommended) or NAIVE (baseline).
            extra_system_note: Optional additional instruction appended to
                               the system prompt (e.g. "Focus on fat loss only").

        Returns:
            BuiltPrompt with .system and .user fields ready for the API call.
        """
        # Trim chunks to fit within context budget
        chunks = self._trim_to_budget(chunks)

        system = SYSTEM_PROMPT
        if extra_system_note:
            system += f"\n\nAdditional instruction: {extra_system_note}"

        if strategy == PromptStrategy.STRUCTURED:
            user = self._build_structured(query, chunks)
        else:
            user = self._build_naive(query, chunks)

        total_chars = len(system) + len(user)

        return BuiltPrompt(
            system=system,
            user=user,
            strategy=strategy.value,
            n_chunks=len(chunks),
            token_estimate=total_chars // 4,
        )

    def build_no_rag(self, query: str) -> BuiltPrompt:
        """
        Build a prompt with NO retrieved context.
        Used as the baseline in RAG vs. no-RAG evaluation.
        """
        system = SYSTEM_PROMPT + (
            "\n\nNote: No external context has been retrieved. "
            "Use your training knowledge only."
        )
        return BuiltPrompt(
            system=system,
            user=query,
            strategy="no_rag",
            n_chunks=0,
            token_estimate=(len(system) + len(query)) // 4,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _trim_to_budget(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Greedily include chunks (best first, already sorted by distance)
        until the character budget is exhausted.
        """
        selected = []
        total = 0
        for chunk in chunks:
            chunk_len = len(chunk.content)
            if total + chunk_len > self.max_context_chars:
                break
            selected.append(chunk)
            total += chunk_len
        return selected

    def _build_structured(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """
        Strategy: STRUCTURED

        Groups chunks by collection, labels each section clearly, and appends
        the user query at the end. The model is told exactly what each block
        contains, which strongly encourages it to cite and use the provided data.

        Example layout:
            [CONTEXT: Nutrition Science & Dietary Guidelines]
            Source: jissn_protein_2017.pdf
            ...chunk text...

            [CONTEXT: Food Data & Recipes]
            Source: USDA Foundation Foods
            ...chunk text...

            [USER REQUEST]
            Generate a weekly meal plan for...
        """
        # Group chunks by collection
        by_collection: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            by_collection.setdefault(chunk.collection, []).append(chunk)

        sections = []

        for collection_name, col_chunks in by_collection.items():
            label = COLLECTION_LABELS.get(collection_name, collection_name)
            lines = [f"[CONTEXT: {label}]"]
            for i, chunk in enumerate(col_chunks, 1):
                lines.append(f"Source: {chunk.source}")
                lines.append(chunk.content.strip())
                if i < len(col_chunks):
                    lines.append("---")
            sections.append("\n".join(lines))

        context_block = "\n\n".join(sections)

        return (
            f"{context_block}\n\n"
            f"[USER REQUEST]\n"
            f"{query}"
        )

    def _build_naive(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """
        Strategy: NAIVE

        Concatenates all chunk text as a flat block with no labelling,
        then appends the query. Used as the comparison baseline to show
        why structured context injection produces better outputs.
        """
        if not chunks:
            return query

        context_text = "\n\n".join(c.content.strip() for c in chunks)

        return (
            f"Use the following information to answer the request:\n\n"
            f"{context_text}\n\n"
            f"Request: {query}"
        )