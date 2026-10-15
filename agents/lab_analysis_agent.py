"""
agents/lab_analysis_agent.py

Agent 1: Lab Analysis

Interprets blood markers against clinical reference ranges retrieved from
the public_health_recommendations ChromaDB collection (ABIM/NBME guidelines,
AHA/ESC cardiovascular guidelines, ADA diabetes standards).

Output: LabAnalysisOutput
  - Analysed markers with status + interpretation
  - Derived dietary_constraints for Nutrition Agent
  - Derived training_constraints for Training Agent
  - Estimated TDEE + recommended caloric target
  - Overall health summary

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
from agents.schemas import (
    HealthProfile,
    UserGoals,
    LabAnalysisOutput,
    ActivityLevel,
)

logger = logging.getLogger(__name__)

MCP_RAG_URL = os.environ.get("MCP_RAG_URL", "http://mcp-rag:8004/mcp")


# ── TDEE multipliers (Mifflin-St Jeor + activity factor) ─────────────────────

ACTIVITY_MULTIPLIERS = {
    ActivityLevel.SEDENTARY:           1.2,
    ActivityLevel.LIGHTLY_ACTIVE:      1.375,
    ActivityLevel.MODERATELY_ACTIVE:   1.55,
    ActivityLevel.VERY_ACTIVE:         1.725,
    ActivityLevel.EXTRA_ACTIVE:        1.9,
}

GOAL_CALORIE_ADJUSTMENTS = {
    "fat_loss":        -400,
    "muscle_gain":     +300,
    "maintenance":       0,
    "performance":     +200,
}


class LabAnalysisAgent(BaseAgent):

    AGENT_NAME     = "LabAnalysisAgent"
    COLLECTION_KEY = "lab_analysis"   # → public_health_recommendations
    OUTPUT_SCHEMA  = LabAnalysisOutput
    MAX_TOKENS     = 4096

    # ── Abstract method implementations ──────────────────────────────────────

    def _build_query(
        self,
        profile:  HealthProfile,
        goals:    UserGoals,
        context:  dict[str, Any],
    ) -> str:
        """
        Build a query that targets clinical reference ranges for the
        specific markers present in this user's blood panel.
        """
        if profile.blood_markers:
            marker_names = ", ".join(m.name for m in profile.blood_markers)
            return (
                f"clinical reference ranges interpretation {marker_names} "
                f"blood test guidelines optimal borderline elevated"
            )
        return (
            "general health biomarkers clinical reference ranges "
            "cholesterol glucose HbA1c interpretation guidelines"
        )

    # ── Module 6: MCP-based retrieval override ────────────────────────────────

    def _retrieve(self, profile, goals, context):
        """
        Override BaseAgent._retrieve — query the RAG corpus through the
        MCP server (Module 6) instead of calling self.retriever directly.
        Output format matches BaseAgent._retrieve exactly.
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
                return result.structuredContent["result"]

    def _build_prompt(
        self,
        profile:    HealthProfile,
        goals:      UserGoals,
        context:    dict[str, Any],
        rag_context: str,
    ) -> str:
        """
        Assemble the full Lab Analysis prompt.
        The LLM receives: clinical guidelines (RAG) + health profile; the output
        structure is enforced through structured output.
        """
        tdee, recommended_calories = self._calculate_calories(profile, goals)

        # Track whether real blood markers were submitted. Used both in the
        # prompt instructions below AND as a hard post-generation guard in
        # _postprocess(), so a hallucinated/copied example can never leak
        # into the final output even if the LLM ignores the instruction.
        self._has_blood_markers = bool(profile.blood_markers)

        parts = [
            rag_context,
            "",
            self._format_profile(profile),
            "",
            self._format_goals(goals),
            "",
            "=== TASK ===",
            "You are a clinical health analyst. Using the guidelines above and the user's",
            "health profile, produce a structured lab analysis.",
            "",
            "Rules:",
            "- Use ONLY the clinical reference ranges from the retrieved guidelines above.",
            "- If a marker has no guideline in the retrieved context, use standard medical knowledge.",
            "- dietary_constraints and training_constraints must be actionable bullet points.",
            "- For each analysed marker, copy name, value and unit exactly from the profile,",
            "  take reference_range from the retrieved guidelines, and keep interpretation,",
            "  dietary_implication and training_implication to one sentence each.",
            "- overall_health_summary: a 2-3 sentence narrative summary.",
            "- estimated_tdee and recommended_calories are pre-calculated — use these exact values:",
            f"  estimated_tdee = {tdee} kcal",
            f"  recommended_calories = {recommended_calories} kcal",
            "- sources_used: list the source names from the retrieved guidelines you used.",
            "",
        ]

        if self._has_blood_markers:
            marker_list = ", ".join(
                f"{m.name} = {m.value} {m.unit}" for m in profile.blood_markers
            )
            parts += [
                f"- This user submitted these blood markers: {marker_list}.",
                "  Populate analysed_markers using ONLY these markers — do not invent",
                "  additional markers that were not submitted.",
                "",
            ]
        else:
            parts += [
                "- This user submitted NO blood markers (blood_markers is empty).",
                "  analysed_markers MUST be an empty list [].",
                "  Do NOT invent, assume, or estimate any blood marker values.",
                "  Derive dietary_constraints and training_constraints from medical",
                "  history and goals only.",
                "",
            ]

        return "\n".join(parts)

    def _postprocess(self, output: LabAnalysisOutput) -> LabAnalysisOutput:
        # Hard guard: regardless of what the LLM returned, if the user
        # submitted no blood markers, force an empty analysed_markers list.
        # This makes the "no markers in → no markers out" guarantee a code
        # invariant rather than something that depends on prompt compliance.
        if not getattr(self, "_has_blood_markers", True) and output.analysed_markers:
            logger.warning(
                f"[{self.AGENT_NAME}] LLM returned {len(output.analysed_markers)} "
                f"analysed_markers with no blood_markers submitted — discarding "
                f"(likely hallucinated)."
            )
            output = output.model_copy(update={"analysed_markers": []})
        return output

    # ── Private helpers ───────────────────────────────────────────────────────

    def _calculate_calories(
        self,
        profile: HealthProfile,
        goals:   UserGoals,
    ) -> tuple[int, int]:
        """
        Mifflin-St Jeor BMR × activity factor → TDEE.
        Then apply goal-based adjustment.
        """
        w  = profile.scale_metrics.weight_kg
        h  = profile.height_cm
        a  = profile.age
        sex = profile.sex.lower()

        if sex == "male":
            bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
        else:
            bmr = (10 * w) + (6.25 * h) - (5 * a) - 161

        multiplier = ACTIVITY_MULTIPLIERS.get(goals.activity_level, 1.55)
        tdee = int(bmr * multiplier)

        adjustment = GOAL_CALORIE_ADJUSTMENTS.get(goals.primary_goal.value, 0)
        recommended = tdee + adjustment

        return tdee, recommended