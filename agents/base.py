"""
agents/base.py

Abstract base class for all agents in the health orchestration pipeline.

Every agent:
  1. Receives a HealthProfile + UserGoals + optional upstream outputs
  2. Builds a RAG-augmented prompt using its assigned ChromaDB collection(s)
  3. Calls vLLM and parses the response into a typed Pydantic model
  4. Retries once on JSON parse failure before raising

Subclasses only need to implement:
  - COLLECTION_KEY   : which ChromaDB collection(s) to query
  - AGENT_NAME       : human-readable name for logging/metadata
  - _build_query()   : what to search for in ChromaDB
  - _build_prompt()  : how to assemble the final LLM prompt
  - _parse_output()  : how to validate the raw JSON into a Pydantic model
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from agents.schemas import HealthProfile, UserGoals

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

VLLM_URL        = "http://localhost:8000/v1/chat/completions"
VLLM_MODEL      = "/models/llama"
MAX_TOKENS      = 2048
TEMPERATURE     = 0.2          # Low — we want deterministic structured output
REQUEST_TIMEOUT = 600          # seconds — same as your RAG pipeline
MAX_RETRIES     = 2


# ── Base Agent ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Subclasses set class-level attributes and implement the three abstract
    methods. The public interface is a single call: agent.run(...).
    """

    AGENT_NAME:     str = "BaseAgent"
    COLLECTION_KEY: str = "full"   # overridden by each subclass

    def __init__(self, retriever=None):
        """
        Args:
            retriever: an initialised HealthRetriever instance.
                       Passed in by the Orchestrator so all agents share
                       the same ChromaDB connection — no repeated startup.
        """
        self.retriever = retriever

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        profile:    HealthProfile,
        goals:      UserGoals,
        context:    Optional[dict[str, Any]] = None,
    ) -> BaseModel:
        """
        Execute this agent and return a validated Pydantic output model.

        Args:
            profile:  The user's health profile.
            goals:    The user's fitness goals.
            context:  Structured outputs from upstream agents, e.g.
                      {"lab_analysis": LabAnalysisOutput(...)}

        Returns:
            A Pydantic model instance (subclass-specific type).

        Raises:
            RuntimeError: if the LLM fails to produce valid JSON after retries.
        """
        context = context or {}
        logger.info(f"[{self.AGENT_NAME}] Starting")

        # 1. Retrieve relevant chunks from ChromaDB
        rag_context = self._retrieve(profile, goals, context)

        # 2. Build the full prompt
        prompt = self._build_prompt(profile, goals, context, rag_context)

        # 3. Call vLLM with retry on JSON parse failure
        for attempt in range(1, MAX_RETRIES + 1):
            raw = self._call_llm(prompt, attempt)
            try:
                result = self._parse_output(raw)
                logger.info(f"[{self.AGENT_NAME}] Completed successfully")
                return result
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"[{self.AGENT_NAME}] Parse failed (attempt {attempt}/{MAX_RETRIES}): {e}"
                )
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"[{self.AGENT_NAME}] Failed to parse valid JSON after "
                        f"{MAX_RETRIES} attempts. Last error: {e}\n"
                        f"Raw response (first 500 chars):\n{raw[:500]}"
                    )

    # ── Abstract methods (subclasses implement these) ─────────────────────────

    @abstractmethod
    def _build_query(
        self,
        profile:  HealthProfile,
        goals:    UserGoals,
        context:  dict[str, Any],
    ) -> str:
        """
        Return the search query string for ChromaDB retrieval.
        Should be specific enough to retrieve targeted chunks.
        Example: "saturated fat LDL cholesterol dietary guidelines"
        """

    @abstractmethod
    def _build_prompt(
        self,
        profile:  HealthProfile,
        goals:    UserGoals,
        context:  dict[str, Any],
        rag_context: str,
    ) -> str:
        """
        Assemble the complete prompt to send to the LLM.
        Must include: system role, user health data, RAG context, output schema.
        """

    @abstractmethod
    def _parse_output(self, raw: str) -> BaseModel:
        """
        Extract and validate JSON from the raw LLM response string.
        Should call self._extract_json(raw) then construct the Pydantic model.
        """

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _retrieve(
        self,
        profile:  HealthProfile,
        goals:    UserGoals,
        context:  dict[str, Any],
    ) -> str:
        """
        Query ChromaDB and return a formatted context string.
        Returns empty string if no retriever is configured.
        """
        if self.retriever is None:
            logger.warning(f"[{self.AGENT_NAME}] No retriever — skipping RAG")
            return ""

        query = self._build_query(profile, goals, context)
        logger.info(f"[{self.AGENT_NAME}] RAG query: '{query[:80]}...'")

        try:
            chunks = self.retriever.retrieve(
                query_text=query,
                query_type=self.COLLECTION_KEY,
                n_results_per_collection=6,
                max_distance=0.8,
            )
            if not chunks:
                logger.warning(f"[{self.AGENT_NAME}] No chunks retrieved")
                return ""

            # Format chunks into a labelled context block
            lines = ["=== RETRIEVED CLINICAL/NUTRITIONAL GUIDELINES ==="]
            for i, chunk in enumerate(chunks, 1):
                lines.append(
                    f"\n[Source {i} | {chunk.collection} | {chunk.source}]\n"
                    f"{chunk.content.strip()}"
                )
            lines.append("\n=== END GUIDELINES ===")
            context_str = "\n".join(lines)

            logger.info(
                f"[{self.AGENT_NAME}] Retrieved {len(chunks)} chunks "
                f"({len(context_str)} chars)"
            )
            return context_str

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Retrieval error: {e}")
            return ""

    def _call_llm(self, prompt: str, attempt: int = 1) -> str:
        """
        Send the prompt to vLLM and return the raw response string.
        """
        logger.info(
            f"[{self.AGENT_NAME}] Calling vLLM "
            f"(attempt {attempt}, prompt_len={len(prompt)})"
        )

        payload = {
            "model": VLLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a specialist health AI assistant. "
                        "You always respond with valid JSON only — "
                        "no prose, no markdown fences, no explanations outside the JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": getattr(self, "MAX_TOKENS_OVERRIDE", MAX_TOKENS),
            "temperature": TEMPERATURE,
        }

        t0 = time.time()
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(VLLM_URL, json=payload)
            response.raise_for_status()

        elapsed = time.time() - t0
        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()

        logger.info(f"[{self.AGENT_NAME}] vLLM response in {elapsed:.1f}s ({len(raw)} chars)")
        return raw

    def _extract_json(self, raw: str) -> dict:
        """
        Robustly extract a JSON object from the LLM response.

        Handles three common LLM output patterns:
          1. Pure JSON   → {"key": ...}
          2. Fenced JSON → ```json\n{...}\n```
          3. JSON embedded in prose → finds the first { ... } block
        """
        # Strip markdown fences if present
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        # Try raw parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Find outermost { } block
        start = raw.find("{")
        end   = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])

        raise json.JSONDecodeError("No JSON object found in response", raw, 0)

    def _format_profile(self, profile: HealthProfile) -> str:
        """Render the health profile as a compact readable block for prompts."""
        lines = [
            "=== USER HEALTH PROFILE ===",
            f"Age: {profile.age}  |  Sex: {profile.sex}  |  Height: {profile.height_cm} cm",
            f"Weight: {profile.scale_metrics.weight_kg} kg",
        ]
        if profile.scale_metrics.body_fat_pct is not None:
            lines.append(f"Body fat: {profile.scale_metrics.body_fat_pct}%")
        if profile.scale_metrics.muscle_mass_kg is not None:
            lines.append(f"Muscle mass: {profile.scale_metrics.muscle_mass_kg} kg")
        if profile.blood_markers:
            lines.append("\nBlood markers:")
            for m in profile.blood_markers:
                lines.append(f"  • {m.name}: {m.value} {m.unit}")
        if profile.medical_history:
            lines.append(f"\nMedical history: {', '.join(profile.medical_history)}")
        if profile.medications:
            lines.append(f"Medications: {', '.join(profile.medications)}")
        if profile.allergies:
            lines.append(f"Allergies/intolerances: {', '.join(profile.allergies)}")
        lines.append("=== END PROFILE ===")
        return "\n".join(lines)

    def _format_goals(self, goals: UserGoals) -> str:
        """Render user goals as a compact readable block for prompts."""
        lines = [
            "=== USER GOALS ===",
            f"Primary goal: {goals.primary_goal.value}",
            f"Activity level: {goals.activity_level.value}",
            f"Fitness level: {goals.fitness_level.value}",
            f"Training days/week: {goals.training_days_per_week}",
        ]
        if goals.dietary_preferences:
            lines.append(f"Dietary preferences: {', '.join(goals.dietary_preferences)}")
        if goals.notes:
            lines.append(f"Notes: {goals.notes}")
        lines.append("=== END GOALS ===")
        return "\n".join(lines)