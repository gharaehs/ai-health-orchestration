#!/usr/bin/env python3
"""
scripts/test_orchestration.py

CLI tool to run the full multi-agent pipeline end-to-end.

Usage:
    # Full pipeline with blood markers
    python3 scripts/test_orchestration.py

    # Minimal profile (no blood markers)
    python3 scripts/test_orchestration.py --minimal

    # Save output to file
    python3 scripts/test_orchestration.py --output results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import Orchestrator
from agents.schemas import (
    HealthProfile,
    UserGoals,
    ScaleMetrics,
    BloodMarker,
    FitnessGoal,
    ActivityLevel,
    FitnessLevel,
    AgentStatus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Sample health profiles ────────────────────────────────────────────────────

def build_full_profile() -> tuple[HealthProfile, UserGoals]:
    """A realistic profile with blood markers — exercises all four agents."""
    profile = HealthProfile(
        age=38,
        sex="male",
        height_cm=178.0,
        scale_metrics=ScaleMetrics(
            weight_kg=88.5,
            body_fat_pct=22.0,
            muscle_mass_kg=58.0,
            bmi=27.9,
        ),
        blood_markers=[
            BloodMarker(name="LDL Cholesterol",  value=3.8,  unit="mmol/L"),
            BloodMarker(name="HDL Cholesterol",  value=1.1,  unit="mmol/L"),
            BloodMarker(name="Triglycerides",    value=2.1,  unit="mmol/L"),
            BloodMarker(name="Fasting Glucose",  value=5.9,  unit="mmol/L"),
            BloodMarker(name="HbA1c",            value=5.8,  unit="%"),
            BloodMarker(name="hsCRP",            value=2.4,  unit="mg/L"),
            BloodMarker(name="Vitamin D",        value=42.0, unit="nmol/L"),
        ],
        medical_history=["Pre-hypertension", "Family history of cardiovascular disease"],
        medications=["None"],
        allergies=["Tree nuts"],
    )
    goals = UserGoals(
        primary_goal=FitnessGoal.FAT_LOSS,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        fitness_level=FitnessLevel.INTERMEDIATE,
        training_days_per_week=4,
        dietary_preferences=["Mediterranean-style"],
        notes="Prefers home cooking. Has access to a gym.",
    )
    return profile, goals


def build_minimal_profile() -> tuple[HealthProfile, UserGoals]:
    """A minimal profile with no blood markers."""
    profile = HealthProfile(
        age=28,
        sex="female",
        height_cm=165.0,
        scale_metrics=ScaleMetrics(
            weight_kg=62.0,
            body_fat_pct=24.0,
        ),
        blood_markers=[],
        medical_history=[],
        medications=[],
        allergies=["Gluten"],
    )
    goals = UserGoals(
        primary_goal=FitnessGoal.MUSCLE_GAIN,
        activity_level=ActivityLevel.LIGHTLY_ACTIVE,
        fitness_level=FitnessLevel.BEGINNER,
        training_days_per_week=3,
        dietary_preferences=["gluten-free"],
    )
    return profile, goals


# ── Output formatting ─────────────────────────────────────────────────────────

def print_summary(result) -> None:
    """Print a human-readable pipeline summary to stdout."""
    print("\n" + "="*60)
    print("  ORCHESTRATION PIPELINE SUMMARY")
    print("="*60)

    for agent_result in result.agent_results:
        icon = {"success": "✅", "failed": "❌", "skipped": "⏭️"}.get(
            agent_result.status.value, "?"
        )
        print(f"\n{icon} {agent_result.agent} ({agent_result.duration_s}s)")
        if agent_result.error:
            print(f"   Error: {agent_result.error}")

    print(f"\n⏱️  Total pipeline time: {result.total_duration_s}s")
    print("="*60)

    if result.lab_analysis:
        lab = result.lab_analysis
        print(f"\n📋 LAB ANALYSIS")
        print(f"   TDEE: {lab.estimated_tdee} kcal | Target: {lab.recommended_calories} kcal")
        print(f"   Dietary constraints: {len(lab.dietary_constraints)}")
        for c in lab.dietary_constraints:
            print(f"     • {c}")
        print(f"   Training constraints: {len(lab.training_constraints)}")
        for c in lab.training_constraints:
            print(f"     • {c}")
        print(f"\n   Summary: {lab.overall_health_summary}")

    if result.nutrition:
        nut = result.nutrition
        print(f"\n🥗 NUTRITION PLAN")
        print(f"   Avg: {nut.avg_daily_calories} kcal | "
              f"P:{nut.avg_daily_protein_g}g | "
              f"C:{nut.avg_daily_carbs_g}g | "
              f"F:{nut.avg_daily_fat_g}g")
        print(f"   Days planned: {len(nut.weekly_plan)}")
        # Show first day as sample
        if nut.weekly_plan:
            day = nut.weekly_plan[0]
            print(f"\n   Sample ({day.day}):")
            for meal in day.meals:
                print(f"     {meal.name}: {meal.recipe_name} ({meal.calories_kcal} kcal)")

    if result.training:
        trn = result.training
        print(f"\n🏋️  TRAINING PROGRAM")
        print(f"   Sessions: {len(trn.weekly_program)} | Rest days: {trn.rest_days}")
        print(f"   Progression: {trn.progression_scheme.principle}")
        if trn.weekly_program:
            session = trn.weekly_program[0]
            print(f"\n   Sample ({session.day} — {session.session_type}):")
            for ex in session.exercises[:3]:
                print(f"     {ex.name}: {ex.sets}×{ex.reps}, rest {ex.rest_seconds}s")

    if result.grocery:
        groc = result.grocery
        print(f"\n🛒 GROCERY LIST")
        print(f"   Items: {groc.total_items} | "
              f"Est. cost: €{groc.estimated_weekly_cost_eur}")
        for category, items in groc.items_by_category.items():
            if items:
                print(f"\n   {category}:")
                for item in items[:3]:
                    print(f"     • {item.name}: {item.total_quantity}")

    print("\n" + "="*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the multi-agent orchestration pipeline")
    parser.add_argument("--minimal", action="store_true", help="Use minimal profile (no blood markers)")
    parser.add_argument("--output",  type=str, default=None, help="Save full JSON result to file")
    args = parser.parse_args()

    if args.minimal:
        profile, goals = build_minimal_profile()
        logger.info("Using MINIMAL profile (no blood markers)")
    else:
        profile, goals = build_full_profile()
        logger.info("Using FULL profile (with blood markers)")

    logger.info(f"Goal: {goals.primary_goal.value} | Days: {goals.training_days_per_week}/week")

    # Run the pipeline
    orchestrator = Orchestrator(chroma_host="localhost", chroma_port=8001)
    result = orchestrator.run(profile=profile, goals=goals)

    # Print summary
    print_summary(result)

    # Save full JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        logger.info(f"Full result saved to {output_path}")

    # Exit with error code if any agent failed
    failures = [r for r in result.agent_results if r.status == AgentStatus.FAILED]
    if failures:
        logger.warning(f"{len(failures)} agent(s) failed")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()