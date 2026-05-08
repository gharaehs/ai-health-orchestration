#!/usr/bin/env python3
"""
scripts/test_rag.py

CLI test script for the Module 4 RAG pipeline.

Usage:
    # Single query — meal plan with RAG
    python scripts/test_rag.py --type meal_plan

    # Single query — gym program with RAG
    python scripts/test_rag.py --type gym_program

    # Lab analysis with RAG
    python scripts/test_rag.py --type lab_analysis

    # Compare RAG vs no-RAG for one query
    python scripts/test_rag.py --type meal_plan --compare

    # Run full evaluation suite (all 4 test cases × 2 modes)
    python scripts/test_rag.py --evaluate

    # Test ChromaDB connectivity only
    python scripts/test_rag.py --health-check
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import HealthRAGPipeline
from rag.prompt_builder import PromptStrategy
from rag.evaluate import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Sample health profiles for quick testing ──────────────────────────────────

SAMPLE_PROFILES = {
    "meal_plan": {
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
    },
    "gym_program": {
        "health_profile": {
            "age": 28,
            "gender": "male",
            "weight_kg": 75,
            "height_cm": 178,
            "training_experience": "intermediate",
            "available_days_per_week": 4,
        },
        "goal": "muscle_gain",
    },
    "grocery_list": {
        "health_profile": {
            "age": 32,
            "gender": "male",
            "weight_kg": 88,
            "dietary_restrictions": [],
        },
        "goal": "fat_loss",
    },
    "lab_analysis": {
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
    },
}


def print_result(result, verbose: bool = False):
    """Pretty-print a RAGResult."""
    print("\n" + "=" * 60)
    print(f"Output type:     {result.output_type}")
    print(f"RAG enabled:     {result.rag_enabled}")
    print(f"Strategy:        {result.strategy_used}")
    print(f"Chunks retrieved:{result.n_chunks_retrieved}")
    print(f"Prompt tokens:   ~{result.prompt_token_est}")
    print(f"Latency:         {result.latency_seconds}s")
    print(f"JSON parse:      {'✅ OK' if result.parse_error is None else f'❌ {result.parse_error}'}")

    if result.chunks and verbose:
        print("\nTop retrieved chunks:")
        for i, c in enumerate(result.chunks[:3], 1):
            print(f"  [{i}] {c.collection} | {c.source} (dist={c.distance:.3f})")
            print(f"      {c.content[:120]}...")

    print("\nResponse:")
    if result.response_json:
        print(json.dumps(result.response_json, indent=2)[:2000])
        if len(json.dumps(result.response_json)) > 2000:
            print("... (truncated, full output in result object)")
    else:
        print(result.response_raw[:1000])
    print("=" * 60)


def cmd_health_check(pipeline: HealthRAGPipeline):
    """Check ChromaDB collections are accessible."""
    print("\n── ChromaDB Health Check ──")
    stats = pipeline.retriever.health_check()
    for name, count in stats.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {name}: {count} documents")


def cmd_single(pipeline: HealthRAGPipeline, output_type: str, verbose: bool):
    """Run a single RAG query."""
    sample = SAMPLE_PROFILES.get(output_type, SAMPLE_PROFILES["meal_plan"])
    print(f"\n── Single RAG Query: {output_type} ──")
    print(f"Goal: {sample['goal']}")

    result = pipeline.query(
        health_profile=sample["health_profile"],
        goal=sample["goal"],
        output_type=output_type,
        rag_enabled=True,
    )
    print_result(result, verbose=verbose)


def cmd_compare(pipeline: HealthRAGPipeline, output_type: str):
    """Run RAG vs no-RAG side-by-side for one query type."""
    sample = SAMPLE_PROFILES.get(output_type, SAMPLE_PROFILES["meal_plan"])
    print(f"\n── RAG vs No-RAG Comparison: {output_type} ──")

    print("\n[1/2] Running WITH RAG...")
    rag_result = pipeline.query(
        health_profile=sample["health_profile"],
        goal=sample["goal"],
        output_type=output_type,
        rag_enabled=True,
    )

    print("\n[2/2] Running WITHOUT RAG (baseline)...")
    no_rag_result = pipeline.query(
        health_profile=sample["health_profile"],
        goal=sample["goal"],
        output_type=output_type,
        rag_enabled=False,
    )

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<28} {'WITH RAG':>15} {'NO RAG':>15}")
    print("-" * 60)
    print(f"{'Latency (s)':<28} {rag_result.latency_seconds:>15.1f} {no_rag_result.latency_seconds:>15.1f}")
    print(f"{'Prompt tokens (est.)':<28} {rag_result.prompt_token_est:>15} {no_rag_result.prompt_token_est:>15}")
    print(f"{'Chunks retrieved':<28} {rag_result.n_chunks_retrieved:>15} {no_rag_result.n_chunks_retrieved:>15}")
    print(f"{'JSON parse OK':<28} {'✅' if rag_result.parse_error is None else '❌':>15} {'✅' if no_rag_result.parse_error is None else '❌':>15}")

    print("\n── RAG Response ──")
    if rag_result.response_json:
        print(json.dumps(rag_result.response_json, indent=2)[:1500])
    else:
        print(rag_result.response_raw[:800])

    print("\n── No-RAG Response ──")
    if no_rag_result.response_json:
        print(json.dumps(no_rag_result.response_json, indent=2)[:1500])
    else:
        print(no_rag_result.response_raw[:800])


def cmd_evaluate(pipeline: HealthRAGPipeline):
    """Run the full evaluation suite."""
    print("\n── Full Module 4 Evaluation Suite ──")
    report = run_evaluation(
        pipeline=pipeline,
        output_path="rag/evaluation_results.json",
    )
    print(f"\nFinal summary: {report.summary}")
    print(f"Full results: rag/evaluation_results.json")


def main():
    parser = argparse.ArgumentParser(
        description="Test the Module 4 RAG pipeline"
    )
    parser.add_argument(
        "--type",
        choices=["meal_plan", "gym_program", "grocery_list", "lab_analysis"],
        default="meal_plan",
        help="Output type for single query (default: meal_plan)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run RAG vs no-RAG side-by-side comparison",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full evaluation suite across all test cases",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Check ChromaDB collection availability only",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show retrieved chunk previews",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--chroma-host",
        default="localhost",
        help="ChromaDB host (default: localhost)",
    )

    args = parser.parse_args()

    # Initialise pipeline
    print("Initialising RAG pipeline...")
    pipeline = HealthRAGPipeline(
        vllm_url=args.vllm_url,
        chroma_host=args.chroma_host,
        strategy=PromptStrategy.STRUCTURED,
    )

    if args.health_check:
        cmd_health_check(pipeline)
    elif args.evaluate:
        cmd_evaluate(pipeline)
    elif args.compare:
        cmd_compare(pipeline, args.type)
    else:
        cmd_single(pipeline, args.type, verbose=args.verbose)


if __name__ == "__main__":
    main()