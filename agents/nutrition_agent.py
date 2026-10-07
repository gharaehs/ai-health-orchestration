"""
agents/nutrition_agent.py

Agent 2: Nutrition

Generates a full 7-day meal plan grounded in nutrition guidelines and
recipes retrieved from two ChromaDB collections:
  - nutrition_guidelines  (USDA, AHA, ADA macro targets)
  - food_and_recipes      (All_Diets.csv, USDA FoodData Central)

Receives dietary_constraints and recommended_calories from LabAnalysisOutput.

Strategy: generates the full Monday–Sunday plan in ONE structured
(tool-use) request. Weekly averages are recomputed in code from the
validated day totals.

Module 6 (MCP): retrieval is routed through the mcp-rag server instead of
calling HealthRetriever directly — see _retrieve()/_mcp_retrieve() below.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agents.base import BaseAgent
from llm import StructuredOutputError
from agents.schemas import (
    HealthProfile,
    UserGoals,
    LabAnalysisOutput,
    NutritionOutput,
)

logger = logging.getLogger(__name__)

MCP_RAG_URL = os.environ.get("MCP_RAG_URL", "http://mcp-rag:8004/mcp")

WEEK_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class NutritionAgent(BaseAgent):

    AGENT_NAME     = "NutritionAgent"
    COLLECTION_KEY = "meal_plan"   # → nutrition_guidelines + food_and_recipes
    OUTPUT_MODEL   = NutritionOutput
    # A full week of meals in one tool call is a large output.
    MAX_TOKENS_OVERRIDE = int(os.environ.get("NUTRITION_MAX_TOKENS", "16000"))

    # ── Abstract method implementations ──────────────────────────────────────

    def _build_query(self, profile, goals, context):
        lab: LabAnalysisOutput | None = context.get("lab_analysis")
        goal_str = goals.primary_goal.value.replace("_", " ")
        prefs    = ", ".join(goals.dietary_preferences) if goals.dietary_preferences else "none"
        if lab and lab.dietary_constraints:
            constraints_str = "; ".join(lab.dietary_constraints[:3])
            return (
                f"meal plan recipes {goal_str} {prefs} "
                f"macros calories protein {constraints_str}"
            )
        return (
            f"meal plan recipes {goal_str} dietary preferences {prefs} "
            f"protein carbohydrates fat macronutrients weekly nutrition"
        )

    # ── Module 6: MCP-based retrieval override ────────────────────────────────

    def _retrieve(self, profile, goals, context):
        """
        Override BaseAgent._retrieve — instead of calling self.retriever
        directly, query the RAG corpus through the MCP server (Module 6).
        Output format matches BaseAgent._retrieve exactly, so downstream
        prompt-building code is unaffected.
        """
        query = self._build_query(profile, goals, context)
        logger.info(f"[{self.AGENT_NAME}] RAG query (via MCP): '{query[:80]}...'")

        try:
            chunks = asyncio.run(self._mcp_retrieve(query))
        except Exception as e:
            logger.exception(f"[{self.AGENT_NAME}] MCP retrieval error")
            return ""

        if not chunks:
            logger.warning(f"[{self.AGENT_NAME}] No chunks retrieved via MCP")
            return ""

        lines = ["=== RETRIEVED CLINICAL/NUTRITIONAL GUIDELINES ==="]
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"\n[Source {i} | {chunk['collection']} | {chunk['source']}]\n"
                f"{chunk['content'].strip()}"
            )
        lines.append("\n=== END GUIDELINES ===")
        context_str = "\n".join(lines)

        logger.info(
            f"[{self.AGENT_NAME}] Retrieved {len(chunks)} chunks via MCP "
            f"({len(context_str)} chars)"
        )
        return context_str

    def _build_prompt(self, profile, goals, context, rag_context):
        lab: LabAnalysisOutput | None = context.get("lab_analysis")
        target_calories = self._get_target_calories(goals, lab)
        constraints_block = self._get_constraints_block(lab)
        days_str = ", ".join(WEEK_DAYS)

        parts = [
            rag_context,
            "",
            self._format_profile(profile),
            "",
            self._format_goals(goals),
            "",
            "=== LAB-DERIVED DIETARY CONSTRAINTS (MUST FOLLOW) ===",
            constraints_block,
            "=== END CONSTRAINTS ===",
            "",
            "=== TASK ===",
            "Generate a complete 7-day meal plan.",
            f"Caloric target: {target_calories} kcal/day",
            "",
            "Rules:",
            f"- weekly_plan must contain exactly 7 entries, in order: {days_str}.",
            "- Each day must have exactly 3 meals: Breakfast, Lunch, Dinner.",
            "- Day totals (calories, protein, carbs, fat) must equal the sum of that day's meals.",
            "- Respect all dietary constraints and profile allergies.",
            "- Ingredients must include quantity and unit (e.g. '150g chicken breast').",
            "- Vary recipes — do not repeat the same recipe twice.",
            "- Keep instructions brief (1-2 sentences).",
            "- caloric_strategy: explain the caloric target and macro split chosen.",
            "- constraint_adherence: explain how each dietary constraint was addressed.",
            "- sources_used: list source names from the retrieved guidelines.",
            "",
            "Submit the plan by calling the provided tool.",
        ]
        return "\n".join(parts)

    def _parse_output(self, output: NutritionOutput, profile, goals, context) -> NutritionOutput:
        days = [d.day.strip().capitalize() for d in output.weekly_plan]
        if days != WEEK_DAYS:
            raise StructuredOutputError(
                f"weekly_plan days must be {WEEK_DAYS} in order, got {days}"
            )

        plan = output.weekly_plan
        return output.model_copy(update={
            "avg_daily_calories":  sum(d.total_calories for d in plan) // 7,
            "avg_daily_protein_g": round(sum(d.total_protein_g for d in plan) / 7, 1),
            "avg_daily_carbs_g":   round(sum(d.total_carbs_g for d in plan) / 7, 1),
            "avg_daily_fat_g":     round(sum(d.total_fat_g for d in plan) / 7, 1),
        })

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_target_calories(self, goals, lab):
        if lab and lab.recommended_calories:
            return lab.recommended_calories
        adjustments = {"fat_loss": -400, "muscle_gain": 300, "maintenance": 0, "performance": 200}
        return 2000 + adjustments.get(goals.primary_goal.value, 0)

    def _get_constraints_block(self, lab):
        if lab and lab.dietary_constraints:
            return "\n".join(f"  • {c}" for c in lab.dietary_constraints)
        return "  • No specific lab-derived constraints. Follow goal-appropriate guidelines."

    async def _mcp_retrieve(self, query: str) -> list[dict]:
        async with streamablehttp_client(MCP_RAG_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "retrieve_context",
                    {
                        "query_text": query,
                        "query_type": self.COLLECTION_KEY,
                        "n_results_per_collection": 6,
                        "max_distance": 0.8,
                    },
                )
                return result.structuredContent["result"]