"""
agents/nutrition_agent.py

Agent 2: Nutrition

Generates a full 7-day meal plan grounded in nutrition guidelines and
recipes retrieved from two ChromaDB collections:
  - nutrition_guidelines  (USDA, AHA, ADA macro targets)
  - food_and_recipes      (All_Diets.csv, USDA FoodData Central)

Receives dietary_constraints and recommended_calories from LabAnalysisOutput.

Strategy: generates in TWO half-week calls (Mon-Thu, Fri-Sun) to avoid
hitting the 2048 token limit with a full 7-day JSON in one response.
Results are merged into a full NutritionOutput.

Module 6 (MCP): retrieval is routed through the mcp-rag server instead of
calling HealthRetriever directly — see _retrieve()/_mcp_retrieve() below.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agents.base import BaseAgent
from agents.schemas import (
    HealthProfile,
    UserGoals,
    LabAnalysisOutput,
    NutritionOutput,
)

logger = logging.getLogger(__name__)

MCP_RAG_URL = os.environ.get("MCP_RAG_URL", "http://mcp-rag:8004/mcp")

WEEK_HALF_1 = ["Monday", "Tuesday", "Wednesday", "Thursday"]
WEEK_HALF_2 = ["Friday", "Saturday", "Sunday"]


class NutritionAgent(BaseAgent):

    AGENT_NAME     = "NutritionAgent"
    COLLECTION_KEY = "meal_plan"   # → nutrition_guidelines + food_and_recipes

    # ── Override base run() to split into two LLM calls ──────────────────────

    def run(self, profile, goals, context=None):
        """
        Generate the meal plan in two halves to stay within token limits:
          Call 1: Monday–Thursday (4 days)
          Call 2: Friday–Sunday   (3 days)
        Then merge into a full 7-day NutritionOutput.
        """
        context = context or {}
        logger.info(f"[{self.AGENT_NAME}] Starting (split-generation mode)")

        rag_context  = self._retrieve(profile, goals, context)
        lab: LabAnalysisOutput | None = context.get("lab_analysis")

        target_calories = self._get_target_calories(goals, lab)
        constraints_block = self._get_constraints_block(lab)

        # ── Half 1: Mon–Thu ───────────────────────────────────────────────────
        logger.info(f"[{self.AGENT_NAME}] Generating Mon–Thu")
        raw1 = self._call_llm(
            self._build_half_prompt(profile, goals, rag_context,
                                    constraints_block, target_calories, WEEK_HALF_1)
        )

        # ── Half 2: Fri–Sun ───────────────────────────────────────────────────
        logger.info(f"[{self.AGENT_NAME}] Generating Fri–Sun")
        raw2 = self._call_llm(
            self._build_half_prompt(profile, goals, rag_context,
                                    constraints_block, target_calories, WEEK_HALF_2)
        )

        result = self._merge_and_parse(raw1, raw2, target_calories, goals, lab)
        logger.info(f"[{self.AGENT_NAME}] Completed successfully")
        return result

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
                return json.loads(result.content[0].text)["result"]

    def _build_prompt(self, profile, goals, context, rag_context):
        # Not used — we use _build_half_prompt instead
        pass

    def _parse_output(self, raw: str) -> NutritionOutput:
        # Not used — we use _merge_and_parse instead
        pass

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

    def _build_half_prompt(self, profile, goals, rag_context,
                            constraints_block, target_calories, days):
        """Build a prompt for a subset of days only."""
        days_str = ", ".join(days)
        n_days   = len(days)

        schema_example = {
            "days": [
                {
                    "day": days[0],
                    "meals": [
                        {
                            "name": "Breakfast",
                            "recipe_name": "Overnight Oats with Berries",
                            "ingredients": ["80g rolled oats", "200ml skimmed milk", "100g mixed berries"],
                            "instructions": "Combine oats and milk overnight. Top with berries.",
                            "calories_kcal": 420,
                            "protein_g": 18.0,
                            "carbs_g": 68.0,
                            "fat_g": 6.0,
                            "notes": "High soluble fibre"
                        }
                    ],
                    "total_calories": target_calories,
                    "total_protein_g": 150.0,
                    "total_carbs_g": 200.0,
                    "total_fat_g": 65.0
                }
            ]
        }

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
            f"Generate a meal plan for EXACTLY these {n_days} days only: {days_str}",
            f"Caloric target: {target_calories} kcal/day",
            "",
            "Rules:",
            f"- Generate exactly {n_days} day entries. Days: {days_str}",
            "- Each day must have exactly 3 meals: Breakfast, Lunch, Dinner.",
            "- Respect all dietary constraints and profile allergies.",
            "- Ingredients must include quantity and unit (e.g. '150g chicken breast').",
            "- Vary recipes — do not repeat the same recipe twice.",
            "- Keep instructions brief (1-2 sentences).",
            "",
            f"Respond with ONLY valid JSON with a 'days' array of exactly {n_days} entries:",
            json.dumps(schema_example, indent=2),
        ]
        return "\n".join(parts)

    def _merge_and_parse(self, raw1, raw2, target_calories, goals, lab) -> NutritionOutput:
        """Merge two half-week responses into a full NutritionOutput."""
        data1 = self._extract_json(raw1)
        data2 = self._extract_json(raw2)

        days1 = data1.get("days", [])
        days2 = data2.get("days", [])
        all_days = days1 + days2

        logger.info(f"[{self.AGENT_NAME}] Merging: {len(days1)} + {len(days2)} = {len(all_days)} days")

        if len(all_days) < 7:
            raise ValueError(
                f"Expected 7 days total, got {len(all_days)} "
                f"({len(days1)} from first call, {len(days2)} from second)"
            )

        # Build weekly_plan in the DayPlan schema format
        weekly_plan = []
        for day_data in all_days[:7]:
            weekly_plan.append({
                "day":             day_data["day"],
                "meals":           day_data["meals"],
                "total_calories":  day_data.get("total_calories", target_calories),
                "total_protein_g": day_data.get("total_protein_g", 0.0),
                "total_carbs_g":   day_data.get("total_carbs_g", 0.0),
                "total_fat_g":     day_data.get("total_fat_g", 0.0),
            })

        avg_cal  = sum(d["total_calories"]  for d in weekly_plan) // 7
        avg_prot = round(sum(d["total_protein_g"] for d in weekly_plan) / 7, 1)
        avg_carb = round(sum(d["total_carbs_g"]   for d in weekly_plan) / 7, 1)
        avg_fat  = round(sum(d["total_fat_g"]     for d in weekly_plan) / 7, 1)

        goal_label = goals.primary_goal.value.replace("_", " ")
        constraint_notes = []
        if lab and lab.dietary_constraints:
            constraint_notes = [f"Addressed: {c}" for c in lab.dietary_constraints]

        return NutritionOutput(
            weekly_plan=weekly_plan,
            avg_daily_calories=avg_cal,
            avg_daily_protein_g=avg_prot,
            avg_daily_carbs_g=avg_carb,
            avg_daily_fat_g=avg_fat,
            caloric_strategy=(
                f"Target {target_calories} kcal/day calibrated for {goal_label}. "
                f"Generated in two half-week batches and merged."
            ),
            constraint_adherence=constraint_notes,
            sources_used=[],
        )

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