"""
agents/training_agent.py

Agent 3: Training

Generates a structured weekly gym program grounded in exercise science
guidelines retrieved from the gym_programming ChromaDB collection
(ACSM/NSCA position stands, JISSN recommendations).

Receives training_constraints from LabAnalysisOutput.

Output: TrainingOutput — weekly sessions with exercises, sets/reps,
        rest periods, and a 4-week progression scheme.
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
    TrainingOutput,
    FitnessGoal,
)

logger = logging.getLogger(__name__)


# ── Session templates per goal ────────────────────────────────────────────────
# Used to guide the query — not hard-coded into the program itself.

GOAL_TRAINING_STYLE = {
    FitnessGoal.FAT_LOSS:       "hypertrophy metabolic conditioning circuit training fat loss",
    FitnessGoal.MUSCLE_GAIN:    "hypertrophy progressive overload strength training muscle gain",
    FitnessGoal.MAINTENANCE:    "general fitness maintenance full body resistance training",
    FitnessGoal.PERFORMANCE:    "athletic performance power speed endurance periodisation",
}


class TrainingAgent(BaseAgent):

    AGENT_NAME     = "TrainingAgent"
    COLLECTION_KEY = "gym_program"   # → gym_programming collection

    def _build_query(
        self,
        profile:  HealthProfile,
        goals:    UserGoals,
        context:  dict[str, Any],
    ) -> str:
        style = GOAL_TRAINING_STYLE.get(goals.primary_goal, "resistance training program")
        level = goals.fitness_level.value
        days  = goals.training_days_per_week

        lab: LabAnalysisOutput | None = context.get("lab_analysis")
        if lab and lab.training_constraints:
            constraints_str = "; ".join(lab.training_constraints[:2])
            return f"{style} {level} {days} days per week {constraints_str} ACSM NSCA guidelines"

        return f"{style} {level} {days} days per week sets reps progression ACSM NSCA"

    def _build_prompt(
        self,
        profile:    HealthProfile,
        goals:      UserGoals,
        context:    dict[str, Any],
        rag_context: str,
    ) -> str:
        lab: LabAnalysisOutput | None = context.get("lab_analysis")

        # Format training constraints
        if lab and lab.training_constraints:
            constraints_block = "\n".join(f"  • {c}" for c in lab.training_constraints)
        else:
            constraints_block = "  • No specific lab-derived constraints."

        days = goals.training_days_per_week
        level = goals.fitness_level.value
        goal  = goals.primary_goal.value.replace("_", " ")

        # Build a sample session for the schema example
        schema_example = {
            "weekly_program": [
                {
                    "day": "Monday",
                    "session_type": "Upper Body Strength",
                    "duration_minutes": 60,
                    "warmup": "5 min light cardio + dynamic shoulder and thoracic mobility",
                    "exercises": [
                        {
                            "name": "Barbell Bench Press",
                            "sets": 4,
                            "reps": "6-8",
                            "rest_seconds": 120,
                            "notes": "RPE 7-8; focus on controlled eccentric"
                        },
                        {
                            "name": "Dumbbell Row",
                            "sets": 4,
                            "reps": "8-10",
                            "rest_seconds": 90,
                            "notes": "Neutral spine throughout"
                        }
                    ],
                    "cooldown": "5 min static stretching — chest, lats, shoulders"
                }
            ],
            "rest_days": ["Wednesday", "Sunday"],
            "progression_scheme": {
                "principle": "Linear periodisation",
                "week_2_adjustment": "Add 2.5 kg to all compound lifts if RPE < 8",
                "week_3_adjustment": "Add 2.5 kg again; increase sets from 4 to 5 on primary lifts",
                "week_4_adjustment": "Deload — reduce weight by 40%, keep same sets/reps"
            },
            "constraint_adherence": [
                "High-intensity interval cardio replaced with moderate-intensity steady-state (elevated CRP)"
            ],
            "sources_used": ["ACSM Position Stand", "NSCA Strength Training Guidelines"]
        }

        parts = [
            rag_context,
            "",
            self._format_profile(profile),
            "",
            self._format_goals(goals),
            "",
            "=== LAB-DERIVED TRAINING CONSTRAINTS (MUST FOLLOW) ===",
            constraints_block,
            "=== END CONSTRAINTS ===",
            "",
            "=== TASK ===",
            "You are a certified strength and conditioning specialist (CSCS).",
            f"Design a {days}-day/week training program for a {level} trainee with a primary goal of {goal}.",
            "",
            "Rules:",
            f"- Include exactly {days} training sessions and {7 - days} rest days.",
            "- Assign sessions to specific days of the week (Monday–Sunday).",
            "- Each session must have: day, session_type, duration_minutes, warmup, exercises, cooldown.",
            "- Each exercise must have: name, sets, reps (as a string range e.g. '8-12'), rest_seconds, notes.",
            "- Include 4-6 exercises per session.",
            "- Provide a 4-week progression scheme (week 4 = deload).",
            "- Respect ALL training constraints above — they are clinically derived.",
            "- Ground recommendations in the retrieved ACSM/NSCA guidelines above.",
            "- constraint_adherence: explain how each training constraint was addressed.",
            "- sources_used: list source names from the retrieved guidelines.",
            "",
            "Respond with ONLY valid JSON matching this exact schema:",
            json.dumps(schema_example, indent=2),
        ]
        return "\n".join(parts)

    def _parse_output(self, raw: str) -> TrainingOutput:
        data = self._extract_json(raw)
        return TrainingOutput(**data)