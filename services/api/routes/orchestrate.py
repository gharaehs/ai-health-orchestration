"""
routes/orchestrate.py

Async job-based orchestration endpoint.

POST /api/orchestrate        → starts pipeline, returns {job_id, status}
GET  /api/orchestrate/status/{job_id} → returns {job_id, status, duration_s}
GET  /api/orchestrate/result/{job_id} → returns full OrchestrationResult
GET  /api/orchestrate/sample-request  → returns sample request body

The pipeline runs in a background thread so the POST returns immediately.
The browser polls /status every 10s until status == "complete" or "failed",
then fetches /result once.
"""

from __future__ import annotations

import logging
import asyncio
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.orchestrator import Orchestrator
from agents.schemas import (
    HealthProfile,
    UserGoals,
    OrchestrationResult,
)
from core.config import VLLM_URL, CHROMA_URL

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Job store (in-memory) ─────────────────────────────────────────────────────
# Stores job state and results. Simple dict is fine for single-user dev context.

class Job(BaseModel):
    job_id: str
    status: str          # "pending" | "running" | "complete" | "failed"
    started_at: float
    duration_s: Optional[float] = None
    error: Optional[str] = None
    result: Optional[OrchestrationResult] = None

_jobs: dict[str, Job] = {}
_executor = ThreadPoolExecutor(max_workers=1)  # One pipeline at a time on T4


# ── Orchestrator singleton ────────────────────────────────────────────────────

def _get_chroma_host_port() -> tuple[str, int]:
    url = CHROMA_URL.replace("http://", "").replace("https://", "")
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        return host, int(port_str)
    return url, 8001


_chroma_host, _chroma_port = _get_chroma_host_port()

import agents.base as _agent_base
_agent_base.VLLM_URL = f"{VLLM_URL}/v1/chat/completions"

_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        logger.info(f"Initialising Orchestrator (chroma={_chroma_host}:{_chroma_port})")
        _orchestrator = Orchestrator(
            chroma_host=_chroma_host,
            chroma_port=_chroma_port,
        )
    return _orchestrator


# ── Background worker ─────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, profile: HealthProfile, goals: UserGoals):
    """Runs in a thread pool. Updates job store when complete."""
    job = _jobs[job_id]
    job.status = "running"
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run(profile, goals)
        job.result = result
        job.status = "complete"
        job.duration_s = round(time.time() - job.started_at, 1)
        logger.info(f"Job {job_id} complete in {job.duration_s}s")
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.duration_s = round(time.time() - job.started_at, 1)
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)


# ── Request model ─────────────────────────────────────────────────────────────

class OrchestrateRequest(BaseModel):
    profile: HealthProfile
    goals:   UserGoals


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/orchestrate")
async def start_orchestration(request: OrchestrateRequest):
    """Start the pipeline. Returns job_id immediately."""
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id=job_id, status="pending", started_at=time.time())
    _jobs[job_id] = job

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline,
        job_id,
        request.profile,
        request.goals,
    )

    logger.info(f"Job {job_id} started — goal: {request.goals.primary_goal.value}")
    return {"job_id": job_id, "status": "pending"}


@router.get("/orchestrate/status/{job_id}")
async def get_status(job_id: str):
    """Poll this endpoint every 10s to check job progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "duration_s": job.duration_s,
        "error": job.error,
    }


@router.get("/orchestrate/result/{job_id}", response_model=OrchestrationResult)
async def get_result(job_id: str):
    """Fetch the full result once status == 'complete'."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != "complete":
        raise HTTPException(status_code=400, detail=f"Job status is '{job.status}', not 'complete'")
    return job.result


@router.get("/orchestrate/sample-request")
async def sample_request():
    """Returns a sample request body for testing."""
    return {
        "profile": {
            "age": 38, "sex": "male", "height_cm": 178.0,
            "scale_metrics": {
                "weight_kg": 88.5, "body_fat_pct": 22.0,
                "muscle_mass_kg": 58.0, "bmi": 27.9
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
            "medical_history": ["Pre-hypertension"],
            "medications": [], "allergies": ["Tree nuts"]
        },
        "goals": {
            "primary_goal": "fat_loss",
            "activity_level": "moderately_active",
            "fitness_level": "intermediate",
            "training_days_per_week": 4,
            "dietary_preferences": ["Mediterranean-style"],
            "notes": None
        }
    }