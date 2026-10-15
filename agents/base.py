"""
agents/base.py

Abstract base class for all agents in the health orchestration pipeline.
...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from agents import llm_client
from agents.schemas import HealthProfile, UserGoals

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

SYSTEM_PROMPT = (
    "You are a specialist health AI assistant. Follow the task instructions "
    "precisely and fill every field of the requested output structure."
)


class BaseAgent(ABC):

    AGENT_NAME:     str = "BaseAgent"
    COLLECTION_KEY: str = "full"
    OUTPUT_SCHEMA:  Optional[type[BaseModel]] = None
    MAX_TOKENS:     int = 4096

    def __init__(self, retriever=None):
        self.retriever = retriever

    def run(self, profile, goals, context=None):
        context = context or {}
        logger.info(f"[{self.AGENT_NAME}] Starting")
        rag_context = self._retrieve(profile, goals, context)
        prompt = self._build_prompt(profile, goals, context, rag_context)
        output = self._generate(prompt, self.OUTPUT_SCHEMA, self.MAX_TOKENS)
        result = self._postprocess(output)
        logger.info(f"[{self.AGENT_NAME}] Completed successfully")
        return result

    def _generate(self, prompt: str, schema: type[BaseModel], max_tokens: int) -> BaseModel:
        """
        Call the LLM, retrying only when the reply fails Pydantic validation.
        Truncation and upstream errors propagate immediately (no retry).
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._call_llm(prompt, schema, max_tokens, attempt)
            except ValidationError as e:
                logger.warning(
                    f"[{self.AGENT_NAME}] Validation failed (attempt {attempt}/{MAX_RETRIES}): {e}"
                )
                if attempt == MAX_RETRIES:
                    raise

    @abstractmethod
    def _build_query(self, profile, goals, context) -> str:
        """Return the search query string for ChromaDB retrieval."""

    @abstractmethod
    def _build_prompt(self, profile, goals, context, rag_context) -> str:
        """Assemble the complete prompt to send to the LLM."""

    def _postprocess(self, output: BaseModel) -> BaseModel:
        """Adjust the validated LLM output before returning it."""
        return output

    def _retrieve(self, profile, goals, context) -> str:
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

    def _call_llm(
        self,
        prompt: str,
        schema: type[BaseModel],
        max_tokens: int,
        attempt: int = 1,
    ) -> BaseModel:
        logger.info(
            f"[{self.AGENT_NAME}] Calling LLM for {schema.__name__} "
            f"(attempt {attempt}, prompt_len={len(prompt)}, max_tokens={max_tokens})"
        )
        return llm_client.generate(SYSTEM_PROMPT, prompt, schema=schema, max_tokens=max_tokens)

    def _format_profile(self, profile: HealthProfile) -> str:
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