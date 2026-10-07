import threading
import time

import pytest

from agents import base as agent_base
from agents.base import BaseAgent
from agents.grocery_agent import GroceryAgent
from agents.lab_analysis_agent import LabAnalysisAgent
from agents.nutrition_agent import WEEK_DAYS, NutritionAgent
from agents.schemas import (
    ActivityLevel,
    BloodMarker,
    FitnessGoal,
    FitnessLevel,
    HealthProfile,
    LabAnalysisOutput,
    NutritionOutput,
    ScaleMetrics,
    TrainingOutput,
    UserGoals,
)
from agents.training_agent import TrainingAgent
from llm.client import tool_name_for

from conftest import tool_response


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_profile(markers=True) -> HealthProfile:
    return HealthProfile(
        age=38, sex="male", height_cm=178.0,
        scale_metrics=ScaleMetrics(weight_kg=88.5),
        blood_markers=[BloodMarker(name="LDL Cholesterol", value=3.8, unit="mmol/L")] if markers else [],
        allergies=["Tree nuts"],
    )


def make_goals() -> UserGoals:
    return UserGoals(
        primary_goal=FitnessGoal.FAT_LOSS,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        fitness_level=FitnessLevel.INTERMEDIATE,
        training_days_per_week=4,
    )


@pytest.fixture(autouse=True)
def no_retrieval(monkeypatch):
    monkeypatch.setattr(BaseAgent, "_retrieve", lambda self, *a, **k: "")
    for cls in (LabAnalysisAgent, NutritionAgent, TrainingAgent):
        monkeypatch.setattr(cls, "_retrieve", lambda self, *a, **k: "")


def tool_name(kwargs) -> str:
    return kwargs["toolConfig"]["toolChoice"]["tool"]["name"]


def lab_payload(**overrides) -> dict:
    data = {
        "analysed_markers": [{
            "name": "LDL Cholesterol", "value": 3.8, "unit": "mmol/L",
            "status": "elevated", "reference_range": "< 3.0 mmol/L",
            "interpretation": "Above target.",
        }],
        "dietary_constraints": ["Limit saturated fat"],
        "training_constraints": ["Include aerobic work"],
        "estimated_tdee": 1, "recommended_calories": 2,
        "overall_health_summary": "Mostly healthy.",
    }
    data.update(overrides)
    return data


def meal(name: str, kcal=600) -> dict:
    return {"name": name, "recipe_name": f"{name} dish", "ingredients": ["150g chicken breast", "80g oats"],
            "instructions": "Cook.", "calories_kcal": kcal, "protein_g": 40.0, "carbs_g": 50.0, "fat_g": 15.0}


def nutrition_payload(days=WEEK_DAYS) -> dict:
    return {
        "weekly_plan": [
            {"day": d, "meals": [meal("Breakfast"), meal("Lunch"), meal("Dinner")],
             "total_calories": 1800 + 10 * i, "total_protein_g": 120.0 + i,
             "total_carbs_g": 150.0, "total_fat_g": 45.0 + i}
            for i, d in enumerate(days)
        ],
        "avg_daily_calories": 0, "avg_daily_protein_g": 0, "avg_daily_carbs_g": 0, "avg_daily_fat_g": 0,
        "caloric_strategy": "Moderate deficit.",
        "constraint_adherence": ["Low saturated fat"],
    }


def training_payload() -> dict:
    return {
        "weekly_program": [{
            "day": "Monday", "session_type": "Upper", "duration_minutes": 60,
            "exercises": [{"name": "Bench", "sets": 4, "reps": "6-8", "rest_seconds": 120}],
        }],
        "rest_days": ["Wednesday", "Sunday"],
        "progression_scheme": {"principle": "Linear", "week_2_adjustment": "+2.5kg",
                               "week_3_adjustment": "+2.5kg", "week_4_adjustment": "Deload"},
    }


# ── 2.1 BaseAgent structured generation + retry ───────────────────────────────

def test_valid_structured_result_is_returned(fake_bedrock):
    fake = fake_bedrock(lambda kw: tool_response(tool_name(kw), training_payload()))
    out = TrainingAgent().run(make_profile(), make_goals())
    assert isinstance(out, TrainingOutput)
    assert out.weekly_program[0].exercises[0].name == "Bench"
    assert len(fake.calls) == 1
    assert tool_name(fake.calls[0]) == tool_name_for(TrainingOutput)
    assert "toolChoice" in fake.calls[0]["toolConfig"]


def test_invalid_result_is_retried_then_reported(fake_bedrock):
    bad = training_payload()
    del bad["progression_scheme"]
    fake = fake_bedrock(lambda kw: tool_response(tool_name(kw), bad))
    with pytest.raises(RuntimeError) as exc:
        TrainingAgent().run(make_profile(), make_goals())
    assert len(fake.calls) == agent_base.MAX_ATTEMPTS
    assert "schema validation" in str(exc.value)
    assert "progression_scheme" in str(exc.value)


def test_invalid_then_valid_recovers(fake_bedrock):
    bad = training_payload()
    del bad["rest_days"]
    responses = [bad, training_payload()]
    fake = fake_bedrock(lambda kw: tool_response(tool_name(kw), responses.pop(0)))
    out = TrainingAgent().run(make_profile(), make_goals())
    assert out.rest_days == ["Wednesday", "Sunday"]
    assert len(fake.calls) == 2


def test_no_vllm_constants_remain():
    assert not any(name.startswith("VLLM") for name in vars(agent_base))
    assert not hasattr(BaseAgent, "_call_llm")
    assert not hasattr(BaseAgent, "_extract_json")


# ── 2.2 Lab Analysis post-processing ──────────────────────────────────────────

def test_lab_calorie_targets_are_calculated(fake_bedrock):
    fake_bedrock(lambda kw: tool_response(tool_name(kw), lab_payload()))
    profile, goals = make_profile(), make_goals()
    agent = LabAnalysisAgent()
    out = agent.run(profile, goals)
    tdee, recommended = agent._calculate_calories(profile, goals)
    # Mifflin-St Jeor: 10*88.5 + 6.25*178 - 5*38 + 5 = 1812.5 × 1.55 → 2809; fat loss −400
    assert (tdee, recommended) == (2809, 2409)
    assert out.estimated_tdee == 2809
    assert out.recommended_calories == 2409
    assert out.analysed_markers[0].name == "LDL Cholesterol"


def test_lab_discards_markers_when_none_submitted(fake_bedrock):
    fake_bedrock(lambda kw: tool_response(tool_name(kw), lab_payload()))
    out = LabAnalysisAgent().run(make_profile(markers=False), make_goals())
    assert isinstance(out, LabAnalysisOutput)
    assert out.analysed_markers == []
    assert out.dietary_constraints == ["Limit saturated fat"]


def test_lab_prompt_has_no_json_template(fake_bedrock):
    fake = fake_bedrock(lambda kw: tool_response(tool_name(kw), lab_payload()))
    LabAnalysisAgent().run(make_profile(markers=False), make_goals())
    prompt = fake.calls[0]["messages"][0]["content"][0]["text"]
    assert "analysed_markers MUST be an empty list" in prompt
    assert "Respond with ONLY valid JSON" not in prompt


def test_training_output_passes_through_unchanged(fake_bedrock):
    fake_bedrock(lambda kw: tool_response(tool_name(kw), training_payload()))
    out = TrainingAgent().run(make_profile(), make_goals())
    assert out == TrainingOutput.model_validate(training_payload())


# ── 2.3 Nutrition single call ─────────────────────────────────────────────────

def test_nutrition_single_call_full_week(fake_bedrock):
    fake = fake_bedrock(lambda kw: tool_response(tool_name(kw), nutrition_payload()))
    lab = LabAnalysisOutput.model_validate({**lab_payload(), "recommended_calories": 2409})
    out = NutritionAgent().run(make_profile(), make_goals(), {"lab_analysis": lab})

    assert len(fake.calls) == 1
    assert fake.calls[0]["inferenceConfig"]["maxTokens"] == NutritionAgent.MAX_TOKENS_OVERRIDE
    assert NutritionAgent.MAX_TOKENS_OVERRIDE >= 8000
    assert [d.day for d in out.weekly_plan] == WEEK_DAYS
    # Averages recomputed from day totals: (1800..1860) → 1830; protein 120..126 → 123.0
    assert out.avg_daily_calories == 1830
    assert out.avg_daily_protein_g == 123.0
    assert out.avg_daily_fat_g == 48.0
    prompt = fake.calls[0]["messages"][0]["content"][0]["text"]
    assert "2409 kcal/day" in prompt and "Limit saturated fat" in prompt


def test_nutrition_wrong_days_are_retried(fake_bedrock):
    wrong = nutrition_payload(days=["Monday"] * 7)
    responses = [wrong, nutrition_payload()]
    fake = fake_bedrock(lambda kw: tool_response(tool_name(kw), responses.pop(0)))
    out = NutritionAgent().run(make_profile(), make_goals())
    assert len(fake.calls) == 2
    assert [d.day for d in out.weekly_plan] == WEEK_DAYS


# ── 2.4 Grocery structured + concurrent ───────────────────────────────────────

def test_grocery_categories_aggregated(fake_bedrock):
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def responder(kw):
        nonlocal in_flight, peak
        name = tool_name(kw)
        text = kw["messages"][0]["content"][0]["text"]
        if name == "submit_shopping_notes":
            return tool_response(name, {"notes": ["Buy in bulk", "Prep grains"]})
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        if "Proteins" in text:
            items = [{"name": "Chicken Breast", "total_quantity": "3150g", "estimated_cost_eur": 30.0}]
        else:
            items = [{"name": "Rolled Oats", "total_quantity": "1680g"}]
        return tool_response(name, {"items": items})

    fake = fake_bedrock(responder)
    nutrition = NutritionOutput.model_validate(nutrition_payload())
    out = GroceryAgent().run(make_profile(), make_goals(), {"nutrition": nutrition})

    assert out.items_by_category["Proteins"][0].name == "Chicken Breast"
    assert out.items_by_category["Proteins"][0].category == "Proteins"
    assert out.items_by_category["Proteins"][0].estimated_cost_eur == 30.0
    assert out.items_by_category["Grains & Carbohydrates"][0].name == "Rolled Oats"
    assert out.items_by_category["Vegetables"] == []
    assert out.total_items == 2
    assert out.shopping_notes == ["Buy in bulk", "Prep grains"]
    assert len(fake.calls) == 3
    assert peak == 2  # both categories consolidated concurrently


def test_grocery_category_failure_falls_back(fake_bedrock):
    def responder(kw):
        name = tool_name(kw)
        if name == "submit_shopping_notes":
            return tool_response(name, {"notes": ["tip"]})
        return tool_response(name, {"items": [{"name": "missing quantity"}]})

    fake_bedrock(responder)
    nutrition = NutritionOutput.model_validate(nutrition_payload())
    out = GroceryAgent().run(make_profile(), make_goals(), {"nutrition": nutrition})
    assert out.items_by_category["Proteins"][0].total_quantity == "see meal plan"
