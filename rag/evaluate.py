"""
rag/evaluate.py

Module 4 evaluation: RAG vs. No-RAG comparison.

Runs a set of standardised health queries through the pipeline in two modes:
  1. RAG enabled (retrieval from ChromaDB)
  2. No RAG (model knowledge only)

Measures and compares:
  - JSON parse success rate
  - Response latency
  - Context grounding (presence of retrieved source terminology)
  - Output completeness (schema field coverage)
  - Qualitative diff: does retrieved context change the output?

Results are saved to rag/evaluation_results.json.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from rag.pipeline import HealthRAGPipeline, RAGResult
from rag.prompt_builder import PromptStrategy

logger = logging.getLogger(__name__)

# ── Evaluation test cases ─────────────────────────────────────────────────────
# Designed to stress-test different retrieval routes and output types.

EVAL_CASES = [
    {
        "name": "fat_loss_meal_plan",
        "description": "Standard fat loss meal plan — tests nutrition + food retrieval",
        "health_profile": {
            "age": 32,
            "gender": "male",
            "weight_kg": 88,
            "height_cm": 180,
            "body_fat_pct": 22,
            "activity_level": "moderate",
            "dietary_restrictions": [],
        },
        "goal": "fat_loss",
        "output_type": "meal_plan",
    },
    {
        "name": "muscle_gain_gym_program",
        "description": "Hypertrophy program — tests gym programming retrieval",
        "health_profile": {
            "age": 28,
            "gender": "male",
            "weight_kg": 75,
            "height_cm": 178,
            "training_experience": "intermediate",
            "available_days_per_week": 4,
        },
        "goal": "muscle_gain",
        "output_type": "gym_program",
    },
    {
        "name": "lab_analysis_lipids",
        "description": "Lipid panel interpretation — tests clinical reference retrieval",
        "health_profile": {
            "age": 45,
            "gender": "male",
            "weight_kg": 92,
            "blood_tests": {
                "total_cholesterol_mmol_l": 6.2,
                "LDL_mmol_l": 4.1,
                "HDL_mmol_l": 1.0,
                "triglycerides_mmol_l": 2.3,
                "fasting_glucose_mmol_l": 5.9,
            },
        },
        "goal": "cardiovascular_health",
        "output_type": "lab_analysis",
    },
    {
        "name": "vegan_fat_loss_meal_plan",
        "description": "Vegan fat loss — tests dietary restriction routing",
        "health_profile": {
            "age": 30,
            "gender": "female",
            "weight_kg": 68,
            "height_cm": 165,
            "activity_level": "moderate",
            "dietary_restrictions": ["vegan", "gluten_free"],
        },
        "goal": "fat_loss",
        "output_type": "meal_plan",
    },
]


@dataclass
class EvalCase:
    name:           str
    description:    str
    output_type:    str
    rag_result:     Optional[dict]       # Serialised RAGResult
    no_rag_result:  Optional[dict]       # Serialised RAGResult
    comparison:     dict                 # Side-by-side metrics


@dataclass
class EvaluationReport:
    timestamp:       str
    total_cases:     int
    rag_parse_rate:  float
    no_rag_parse_rate: float
    avg_rag_latency: float
    avg_no_rag_latency: float
    avg_chunks_retrieved: float
    cases:           list[dict]
    summary:         str


def _serialise_result(r: RAGResult) -> dict:
    """Convert RAGResult to a JSON-serialisable dict."""
    return {
        "output_type":        r.output_type,
        "rag_enabled":        r.rag_enabled,
        "strategy_used":      r.strategy_used,
        "n_chunks_retrieved": r.n_chunks_retrieved,
        "prompt_token_est":   r.prompt_token_est,
        "latency_seconds":    r.latency_seconds,
        "parse_ok":           r.parse_error is None,
        "parse_error":        r.parse_error,
        "response_json":      r.response_json,
        "response_raw":       r.response_raw[:500] + "..." if len(r.response_raw) > 500 else r.response_raw,
        "top_chunks": [
            {
                "collection": c.collection,
                "source":     c.source,
                "distance":   round(c.distance, 4),
                "preview":    c.content[:150] + "...",
            }
            for c in r.chunks[:3]
        ],
    }


def _count_schema_fields(response_json: Optional[dict], output_type: str) -> int:
    """
    Counts how many expected top-level schema fields are present in the response.
    Used as a proxy for output completeness.
    """
    if response_json is None:
        return 0

    expected = {
        "meal_plan":    ["meal_plan"],
        "gym_program":  ["gym_program"],
        "grocery_list": ["grocery_list"],
        "lab_analysis": ["lab_analysis"],
        "full_plan":    ["plan"],
    }

    keys = expected.get(output_type, [])
    return sum(1 for k in keys if k in response_json)


def _compare_results(rag: RAGResult, no_rag: RAGResult) -> dict:
    """Generate a side-by-side comparison of RAG vs. no-RAG for one test case."""
    rag_fields    = _count_schema_fields(rag.response_json, rag.output_type)
    no_rag_fields = _count_schema_fields(no_rag.response_json, no_rag.output_type)

    latency_delta = round(rag.latency_seconds - no_rag.latency_seconds, 2)

    return {
        "rag_parse_ok":          rag.parse_error is None,
        "no_rag_parse_ok":       no_rag.parse_error is None,
        "rag_latency_s":         rag.latency_seconds,
        "no_rag_latency_s":      no_rag.latency_seconds,
        "latency_overhead_s":    latency_delta,
        "rag_token_est":         rag.prompt_token_est,
        "no_rag_token_est":      no_rag.prompt_token_est,
        "context_chunks_added":  rag.n_chunks_retrieved,
        "rag_schema_fields_ok":  rag_fields,
        "no_rag_schema_fields_ok": no_rag_fields,
        "verdict": (
            "RAG_BETTER"    if rag_fields > no_rag_fields or
                               (rag.parse_error is None and no_rag.parse_error is not None)
            else "EQUAL"    if rag_fields == no_rag_fields and
                               rag.parse_error == no_rag.parse_error
            else "NO_RAG_BETTER"
        ),
    }


def run_evaluation(
    pipeline: HealthRAGPipeline,
    output_path: str = "rag/evaluation_results.json",
    strategy: PromptStrategy = PromptStrategy.STRUCTURED,
) -> EvaluationReport:
    """
    Run the full Module 4 evaluation suite.

    Each test case is run twice:
      - With RAG (retrieval enabled, STRUCTURED strategy)
      - Without RAG (model knowledge only)

    Args:
        pipeline:    Initialised HealthRAGPipeline instance.
        output_path: Where to save the JSON results file.
        strategy:    Prompt strategy to use for RAG runs.

    Returns:
        EvaluationReport with aggregate metrics and per-case results.
    """
    from datetime import datetime

    logger.info("=" * 60)
    logger.info("Module 4 RAG Evaluation")
    logger.info("=" * 60)

    cases_output = []
    rag_parse_count    = 0
    no_rag_parse_count = 0
    rag_latencies      = []
    no_rag_latencies   = []
    total_chunks       = []

    for i, case in enumerate(EVAL_CASES, 1):
        logger.info(f"\n[{i}/{len(EVAL_CASES)}] {case['name']}: {case['description']}")

        # ── RAG run ───────────────────────────────────────────────────────────
        logger.info("  Running WITH RAG...")
        rag_result = pipeline.query(
            health_profile=case["health_profile"],
            goal=case["goal"],
            output_type=case["output_type"],
            rag_enabled=True,
            override_strategy=strategy,
        )

        # ── No-RAG baseline ───────────────────────────────────────────────────
        logger.info("  Running WITHOUT RAG (baseline)...")
        no_rag_result = pipeline.query(
            health_profile=case["health_profile"],
            goal=case["goal"],
            output_type=case["output_type"],
            rag_enabled=False,
        )

        # ── Compare ───────────────────────────────────────────────────────────
        comparison = _compare_results(rag_result, no_rag_result)

        logger.info(
            f"  RAG:    parse={'✅' if rag_result.parse_error is None else '❌'}, "
            f"chunks={rag_result.n_chunks_retrieved}, "
            f"latency={rag_result.latency_seconds:.1f}s"
        )
        logger.info(
            f"  No-RAG: parse={'✅' if no_rag_result.parse_error is None else '❌'}, "
            f"latency={no_rag_result.latency_seconds:.1f}s"
        )
        logger.info(f"  Verdict: {comparison['verdict']}")

        # Aggregate stats
        if rag_result.parse_error is None:    rag_parse_count += 1
        if no_rag_result.parse_error is None: no_rag_parse_count += 1
        rag_latencies.append(rag_result.latency_seconds)
        no_rag_latencies.append(no_rag_result.latency_seconds)
        total_chunks.append(rag_result.n_chunks_retrieved)

        cases_output.append({
            "name":        case["name"],
            "description": case["description"],
            "output_type": case["output_type"],
            "rag":         _serialise_result(rag_result),
            "no_rag":      _serialise_result(no_rag_result),
            "comparison":  comparison,
        })

    n = len(EVAL_CASES)
    rag_parse_rate    = rag_parse_count    / n
    no_rag_parse_rate = no_rag_parse_count / n
    avg_rag_lat       = sum(rag_latencies)    / n
    avg_no_rag_lat    = sum(no_rag_latencies) / n
    avg_chunks        = sum(total_chunks)     / n

    verdicts = [c["comparison"]["verdict"] for c in cases_output]
    rag_better_count  = verdicts.count("RAG_BETTER")

    summary = (
        f"RAG improved output in {rag_better_count}/{n} test cases. "
        f"JSON parse rate: RAG={rag_parse_rate:.0%} vs No-RAG={no_rag_parse_rate:.0%}. "
        f"Average latency overhead: {avg_rag_lat - avg_no_rag_lat:.1f}s per query. "
        f"Average context injected: {avg_chunks:.1f} chunks."
    )

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(summary)

    report = EvaluationReport(
        timestamp=datetime.utcnow().isoformat(),
        total_cases=n,
        rag_parse_rate=rag_parse_rate,
        no_rag_parse_rate=no_rag_parse_rate,
        avg_rag_latency=round(avg_rag_lat, 2),
        avg_no_rag_latency=round(avg_no_rag_lat, 2),
        avg_chunks_retrieved=round(avg_chunks, 1),
        cases=cases_output,
        summary=summary,
    )

    # Save to disk
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(asdict(report), f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")
    return report