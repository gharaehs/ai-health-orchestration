"""
agents/base.py

Abstract base class for all agents in the health orchestration pipeline.
...
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from pydantic import BaseModel, ValidationError

from agents.schemas import HealthProfile, UserGoals
from llm import StructuredOutputError, generate_structured

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "You are a specialist health AI assistant. "
    "Always deliver your final answer by calling the provided tool exactly once "
    "with a complete result that matches its input schema."
)


class BaseAgent(ABC):

    AGENT_NAME:          str = "BaseAgent"
    COLLECTION_KEY:      str = "full"
    OUTPUT_MODEL:        type[BaseModel]
    MAX_TOKENS_OVERRIDE: Optional[int] = None

    def __init__(self, retriever=None):
        self.retriever = retriever

    def run(self, profile, goals, context=None):
        context = context or {}
        logger.info(f"[{self.AGENT_NAME}] Starting")
        rag_context = self._retrieve(profile, goals, context)
        prompt = self._build_prompt(profile, goals, context, rag_context)

        result = self._generate_with_retries(
            prompt,
            self.OUTPUT_MODEL,
            self.MAX_TOKENS_OVERRIDE,
            postprocess=lambda output: self._parse_output(output, profile, goals, context),
        )
        logger.info(f"[{self.AGENT_NAME}] Completed successfully")
        return result

    @abstractmethod
    def _build_query(self, profile, goals, context) -> str:
        """Return the search query string for ChromaDB retrieval."""

    @abstractmethod
    def _build_prompt(self, profile, goals, context, rag_context) -> str:
        """Assemble the complete prompt to send to the LLM."""

    def _parse_output(self, output: BaseModel, profile, goals, context) -> BaseModel:
        """
        Post-process the schema-validated output. Default: return as-is.
        Raise StructuredOutputError to reject the output and trigger a retry.
        """
        return output

    def _generate(self, prompt: str, output_model: type[BaseModel],
                  max_tokens: Optional[int] = None) -> BaseModel:
        """One structured LLM call. Raises ValidationError on schema violation."""
        logger.info(
            f"[{self.AGENT_NAME}] Calling Bedrock for {output_model.__name__} "
            f"(prompt_len={len(prompt)})"
        )
        return generate_structured(SYSTEM_PROMPT, prompt, output_model, max_tokens).content

    def _generate_with_retries(
        self,
        prompt: str,
        output_model: type[BaseModel],
        max_tokens: Optional[int] = None,
        postprocess: Optional[Callable[[BaseModel], BaseModel]] = None,
    ) -> BaseModel:
        """Structured call retried on schema violation up to MAX_ATTEMPTS."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                output = self._generate(prompt, output_model, max_tokens)
                return postprocess(output) if postprocess else output
            except (ValidationError, StructuredOutputError) as e:
                logger.warning(
                    f"[{self.AGENT_NAME}] Invalid structured output "
                    f"(attempt {attempt}/{MAX_ATTEMPTS}): {e}"
                )
                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"[{self.AGENT_NAME}] Output failed schema validation after "
                        f"{MAX_ATTEMPTS} attempts. Last error: {e}"
                    ) from e
        raise AssertionError("unreachable")

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