"""
agents/training_agent.py

Agent 3: Training

Generates a structured weekly gym program grounded in exercise science
guidelines retrieved from the gym_programming ChromaDB collection
(ACSM/NSCA position stands, JISSN recommendations).

Receives training_constraints from LabAnalysisOutput.

Output: TrainingOutput — weekly sessions with exercises, sets/reps,
        rest periods, and a 4-week progression scheme.

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
    TrainingOutput,
    FitnessGoal,
)

logger = logging.getLogger(__name__)

MCP_RAG_URL = os.environ.get("MCP_RAG_URL", "http://mcp-rag:8004/mcp")


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