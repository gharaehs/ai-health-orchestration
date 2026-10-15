"""Shared sample inputs for agent tests."""

from agents.schemas import (
    ActivityLevel,
    BloodMarker,
    FitnessGoal,
    FitnessLevel,
    HealthProfile,
    ScaleMetrics,
    UserGoals,
)


def profile(markers: bool = True) -> HealthProfile:
    return HealthProfile(
        age=38, sex="male", height_cm=178.0,
        scale_metrics=ScaleMetrics(weight_kg=88.5),
        blood_markers=[BloodMarker(name="LDL Cholesterol", value=3.8, unit="mmol/L")] if markers else [],
        allergies=["Tree nuts"],
    )


def goals() -> UserGoals:
    return UserGoals(
        primary_goal=FitnessGoal.FAT_LOSS,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        fitness_level=FitnessLevel.INTERMEDIATE,
        training_days_per_week=4,
    )


def meal(name: str = "Breakfast", ingredients=None) -> dict:
    return {
        "name": name, "recipe_name": f"{name} bowl",
        "ingredients": ingredients or ["100g oats", "150g chicken breast"],
        "instructions": "Mix.", "calories_kcal": 500,
        "protein_g": 30.0, "carbs_g": 50.0, "fat_g": 15.0,
    }


def day(name: str, calories: int = 2000) -> dict:
    return {
        "day": name, "meals": [meal("Breakfast"), meal("Lunch"), meal("Dinner")],
        "total_calories": calories, "total_protein_g": 150.0,
        "total_carbs_g": 200.0, "total_fat_g": 60.0,
    }


def lab_output(markers: bool = True) -> dict:
    return {
        "analysed_markers": [{
            "name": "LDL Cholesterol", "value": 3.8, "unit": "mmol/L",
            "status": "elevated", "reference_range": "< 3.0 mmol/L",
            "interpretation": "Raised LDL.",
        }] if markers else [],
        "dietary_constraints": ["Limit saturated fat"],
        "training_constraints": ["Include moderate cardio"],
        "estimated_tdee": 2700, "recommended_calories": 2300,
        "overall_health_summary": "Generally healthy.",
        "sources_used": [],
    }


def training_output() -> dict:
    return {
        "weekly_program": [{
            "day": "Monday", "session_type": "Upper", "duration_minutes": 60,
            "exercises": [{"name": "Bench", "sets": 4, "reps": "6-8", "rest_seconds": 120}],
        }],
        "rest_days": ["Wednesday", "Sunday"],
        "progression_scheme": {
            "principle": "Linear", "week_2_adjustment": "+2.5kg",
            "week_3_adjustment": "+2.5kg", "week_4_adjustment": "Deload",
        },
    }


SAMPLE_BODY = {
    "profile": profile().model_dump(mode="json"),
    "goals": goals().model_dump(mode="json"),
}
