"""
agents/orchestrator.py

The Orchestrator coordinates the full multi-agent pipeline.

Execution order:
  1. LabAnalysisAgent   → produces dietary + training constraints + caloric target
  2. NutritionAgent     → consumes lab constraints → produces 7-day meal plan
  3. TrainingAgent      → consumes lab constraints → produces weekly gym program
  4. GroceryAgent       → consumes meal plan       → produces shopping list

Agents 2 and 3 run sequentially (both depend on Agent 1 output).
Agent 4 runs after Agent 2 (depends on meal plan).

The Orchestrator:
  - Initialises a single shared HealthRetriever (one ChromaDB connection)
  - Passes structured outputs between agents via the context dict
  - Records per-agent timing and status
  - Returns a fully populated OrchestrationResult
  - Never raises on agent failure — marks the agent as FAILED and continues
    so the remaining agents still run with whatever context is available
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from rag.retriever import HealthRetriever

from agents.lab_analysis_agent import LabAnalysisAgent
from agents.nutrition_agent import NutritionAgent
from agents.training_agent import TrainingAgent
from agents.grocery_agent import GroceryAgent
from agents.schemas import (
    HealthProfile,
    UserGoals,
    LabAnalysisOutput,
    NutritionOutput,
    TrainingOutput,
    GroceryOutput,
    OrchestrationResult,
    AgentResult,
    AgentStatus,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordinates the four-agent health orchestration pipeline.

    Usage:
        orchestrator = Orchestrator()
        result = orchestrator.run(profile, goals)
    """

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8001,
    ):
        """
        Initialise the shared retriever and all four agents.

        Args:
            chroma_host: ChromaDB host (default: localhost)
            chroma_port: ChromaDB port (default: 8001)
        """
        logger.info("Initialising Orchestrator")

        # Single shared retriever — one ChromaDB connection for all agents
        self.retriever = HealthRetriever(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
        )

        # Instantiate all agents with the shared retriever
        self.lab_agent       = LabAnalysisAgent(retriever=self.retriever)
        self.nutrition_agent = NutritionAgent(retriever=self.retriever)
        self.training_agent  = TrainingAgent(retriever=self.retriever)
        self.grocery_agent   = GroceryAgent(retriever=self.retriever)

        logger.info("Orchestrator ready — 4 agents initialised")

    def run(
        self,
        profile: HealthProfile,
        goals:   UserGoals,
    ) -> OrchestrationResult:
        """
        Execute the full pipeline and return a combined result.

        Args:
            profile: User health profile (demographics, body composition, blood markers)
            goals:   User fitness goals

        Returns:
            OrchestrationResult with all agent outputs and execution metadata
        """
        pipeline_start = time.time()
        logger.info(
            f"Pipeline starting — goal: {goals.primary_goal.value}, "
            f"markers: {len(profile.blood_markers)}, "
            f"training_days: {goals.training_days_per_week}"
        )

        result = OrchestrationResult()
        context: dict = {}

        # ── Agent 1: Lab Analysis ─────────────────────────────────────────────
        lab_output, lab_result = self._run_agent(
            agent=self.lab_agent,
            agent_name="LabAnalysisAgent",
            profile=profile,
            goals=goals,
            context=context,
        )
        result.agent_results.append(lab_result)

        if lab_output is not None:
            result.lab_analysis = lab_output
            context["lab_analysis"] = lab_output
            logger.info(
                f"Lab analysis complete — "
                f"dietary constraints: {len(lab_output.dietary_constraints)}, "
                f"training constraints: {len(lab_output.training_constraints)}, "
                f"target calories: {lab_output.recommended_calories}"
            )

        # ── Agent 2: Nutrition ────────────────────────────────────────────────
        # Runs regardless of whether lab analysis succeeded — uses fallback
        # calorie targets if lab_analysis is not in context
        nutrition_output, nutrition_result = self._run_agent(
            agent=self.nutrition_agent,
            agent_name="NutritionAgent",
            profile=profile,
            goals=goals,
            context=context,
        )
        result.agent_results.append(nutrition_result)

        if nutrition_output is not None:
            result.nutrition = nutrition_output
            context["nutrition"] = nutrition_output
            logger.info(
                f"Nutrition plan complete — "
                f"avg calories: {nutrition_output.avg_daily_calories} kcal/day, "
                f"avg protein: {nutrition_output.avg_daily_protein_g}g/day"
            )

        # ── Agent 3: Training ─────────────────────────────────────────────────
        # Runs in parallel conceptually with Nutrition (both consume lab output)
        # but sequentially here to avoid parallel LLM calls on the T4
        training_output, training_result = self._run_agent(
            agent=self.training_agent,
            agent_name="TrainingAgent",
            profile=profile,
            goals=goals,
            context=context,
        )
        result.agent_results.append(training_result)

        if training_output is not None:
            result.training = training_output
            logger.info(
                f"Training program complete — "
                f"sessions: {len(training_output.weekly_program)}, "
                f"rest days: {training_output.rest_days}"
            )

        # ── Agent 4: Grocery ──────────────────────────────────────────────────
        # Only runs if Nutrition succeeded (needs the meal plan)
        if nutrition_output is not None:
            grocery_output, grocery_result = self._run_agent(
                agent=self.grocery_agent,
                agent_name="GroceryAgent",
                profile=profile,
                goals=goals,
                context=context,
            )
            result.agent_results.append(grocery_result)

            if grocery_output is not None:
                result.grocery = grocery_output
                logger.info(
                    f"Grocery list complete — "
                    f"items: {grocery_output.total_items}, "
                    f"est. cost: €{grocery_output.estimated_weekly_cost_eur}"
                )
        else:
            result.agent_results.append(AgentResult(
                agent="GroceryAgent",
                status=AgentStatus.SKIPPED,
                duration_s=0.0,
                error="Skipped: NutritionAgent did not produce output",
            ))
            logger.warning("GroceryAgent skipped — no nutrition output available")

        # ── Pipeline complete ─────────────────────────────────────────────────
        result.total_duration_s = round(time.time() - pipeline_start, 2)

        succeeded = sum(1 for r in result.agent_results if r.status == AgentStatus.SUCCESS)
        logger.info(
            f"Pipeline complete — "
            f"{succeeded}/{len(result.agent_results)} agents succeeded, "
            f"total time: {result.total_duration_s}s"
        )

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_agent(
        self,
        agent,
        agent_name: str,
        profile:    HealthProfile,
        goals:      UserGoals,
        context:    dict,
    ) -> tuple[Optional[object], AgentResult]:
        """
        Run a single agent, catching any exception so the pipeline continues.

        Returns:
            (output, AgentResult) — output is None if the agent failed
        """
        t0 = time.time()
        try:
            output = agent.run(profile=profile, goals=goals, context=context)
            duration = round(time.time() - t0, 2)
            return output, AgentResult(
                agent=agent_name,
                status=AgentStatus.SUCCESS,
                duration_s=duration,
            )
        except Exception as e:
            duration = round(time.time() - t0, 2)
            logger.error(f"[{agent_name}] Failed after {duration}s: {e}", exc_info=True)
            return None, AgentResult(
                agent=agent_name,
                status=AgentStatus.FAILED,
                duration_s=duration,
                error=str(e),
            )