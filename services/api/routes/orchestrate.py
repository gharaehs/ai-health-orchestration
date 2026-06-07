"""
routes/orchestrate.py

POST /api/orchestrate
Runs the full 4-agent pipeline and returns a structured OrchestrationResult.

The request body mirrors the HealthProfile + UserGoals schemas.
The response is the full OrchestrationResult as JSON.

Because the pipeline takes 5-10 minutes on the T4, this endpoint:
- Sets a long timeout (none — FastAPI doesn't timeout by default)
- Returns the full result synchronously (no streaming)
- Includes per-agent timing in the response for the dashboard
"""

from __future__ import annotations

import logging
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.orchestrator import Orchestrator
from agents.schemas import (
    HealthProfile,
    UserGoals,
    ScaleMetrics,
    BloodMarker,
    OrchestrationResult,
    FitnessGoal,
    ActivityLevel,
    FitnessLevel,
)
from core.config import VLLM_URL, CHROMA_URL

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Request / Response models ─────────────────────────────────────────────────

class OrchestrateRequest(BaseModel):
    profile: HealthProfile
    goals:   UserGoals


# ── Shared orchestrator instance ──────────────────────────────────────────────
# Initialised once at import time so ChromaDB connection and embedding model
# are loaded once, not on every request.

def _get_chroma_host_port() -> tuple[str, int]:
    """Parse host and port from CHROMA_URL (e.g. 'http://vector-db:8000')."""
    url = CHROMA_URL.replace("http://", "").replace("https://", "")
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        return host, int(port_str)
    return url, 8001


_chroma_host, _chroma_port = _get_chroma_host_port()

# Patch the vLLM URL in agents/base.py to use the Docker service name
import agents.base as _agent_base
_agent_base.VLLM_URL = f"{VLLM_URL}/v1/chat/completions"

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info(f"Initialising Orchestrator (chroma={_chroma_host}:{_chroma_port})")
        _orchestrator = Orchestrator(
            chroma_host=_chroma_host,
            chroma_port=_chroma_port,
        )
    return _orchestrator


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/orchestrate", response_model=OrchestrationResult)
async def orchestrate(request: OrchestrateRequest):
    """
    Run the full multi-agent health orchestration pipeline.

    Agents run sequentially:
      1. LabAnalysisAgent  → interprets blood markers
      2. NutritionAgent    → generates 7-day meal plan
      3. TrainingAgent     → generates weekly gym program
      4. GroceryAgent      → aggregates shopping list

    Returns OrchestrationResult with all outputs and per-agent timing.
    """
    try:
        orchestrator = get_orchestrator()
        logger.info(
            f"Orchestration request — goal: {request.goals.primary_goal.value}, "
            f"markers: {len(request.profile.blood_markers)}"
        )

        # Run pipeline (blocking — takes 5-10 min on T4)
        # FastAPI runs this in a thread pool via run_in_executor for true async,
        # but for simplicity we run it directly since we have a single user context
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: orchestrator.run(request.profile, request.goals)
        )

        logger.info(
            f"Orchestration complete — "
            f"{sum(1 for r in result.agent_results if r.status.value == 'success')}/"
            f"{len(result.agent_results)} agents succeeded, "
            f"total: {result.total_duration_s}s"
        )
        return result

    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orchestrate/sample-request")
async def sample_request():
    """
    Returns a sample request body for the orchestrate endpoint.
    Useful for testing from the dashboard or curl.
    """
    return {
        "profile": {
            "age": 38,
            "sex": "male",
            "height_cm": 178.0,
            "scale_metrics": {
                "weight_kg": 88.5,
                "body_fat_pct": 22.0,
                "muscle_mass_kg": 58.0,
                "bmi": 27.9
            },
            "blood_markers": [
                {"name": "LDL Cholesterol",  "value": 3.8,  "unit": "mmol/L"},
                {"name": "HDL Cholesterol",  "value": 1.1,  "unit": "mmol/L"},
                {"name": "Triglycerides",    "value": 2.1,  "unit": "mmol/L"},
                {"name": "Fasting Glucose",  "value": 5.9,  "unit": "mmol/L"},
                {"name": "HbA1c",            "value": 5.8,  "unit": "%"},
                {"name": "hsCRP",            "value": 2.4,  "unit": "mg/L"},
                {"name": "Vitamin D",        "value": 42.0, "unit": "nmol/L"}
            ],
            "medical_history": ["Pre-hypertension", "Family history of cardiovascular disease"],
            "medications": [],
            "allergies": ["Tree nuts"]
        },
        "goals": {
            "primary_goal": "fat_loss",
            "activity_level": "moderately_active",
            "fitness_level": "intermediate",
            "training_days_per_week": 4,
            "dietary_preferences": ["Mediterranean-style"],
            "notes": "Prefers home cooking. Has access to a gym."
        }
    }