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
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import BaseAgent
from agents.schemas import (
    HealthProfile,
    UserGoals,
    LabAnalysisOutput,
    ActivityLevel,
)

logger = logging.getLogger(__name__)


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

    def _build_prompt(
        self,
        profile:    HealthProfile,
        goals:      UserGoals,
        context:    dict[str, Any],
        rag_context: str,
    ) -> str:
        """
        Assemble the full Lab Analysis prompt.
        The LLM receives: clinical guidelines (RAG) + health profile + output schema.
        """
        tdee, recommended_calories = self._calculate_calories(profile, goals)

        schema_example = {
            "analysed_markers": [
                {
                    "name": "LDL Cholesterol",
                    "value": 3.8,
                    "unit": "mmol/L",
                    "status": "elevated",
                    "reference_range": "Optimal: < 2.6 mmol/L",
                    "interpretation": "LDL is borderline-high, increasing cardiovascular risk.",
                    "dietary_implication": "Reduce saturated fat; increase soluble fibre and omega-3.",
                    "training_implication": "Moderate-intensity cardio recommended to improve lipid profile."
                }
            ],
            "dietary_constraints": [
                "Limit saturated fat to < 7% of calories (elevated LDL)",
                "Increase soluble fibre intake (oats, legumes, vegetables)"
            ],
            "training_constraints": [
                "Include 150 min/week moderate-intensity cardio (elevated LDL)"
            ],
            "estimated_tdee": tdee,
            "recommended_calories": recommended_calories,
            "overall_health_summary": "2-3 sentence narrative summary.",
            "sources_used": ["ABIM Lab Reference", "AHA Cardiovascular Guidelines"]
        }

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
            "- estimated_tdee and recommended_calories are pre-calculated — use these exact values:",
            f"  estimated_tdee = {tdee} kcal",
            f"  recommended_calories = {recommended_calories} kcal",
            "- If no blood markers are provided, return an empty analysed_markers list",
            "  and derive constraints from the medical history and goals only.",
            "- sources_used: list the source names from the retrieved guidelines you used.",
            "",
            "Respond with ONLY valid JSON matching this exact schema:",
            json.dumps(schema_example, indent=2),
        ]
        return "\n".join(parts)

    def _parse_output(self, raw: str) -> LabAnalysisOutput:
        data = self._extract_json(raw)
        return LabAnalysisOutput(**data)

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