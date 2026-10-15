import pytest

from agents import llm_client
from agents.grocery_agent import DEFAULT_SHOPPING_TIPS, GroceryAgent
from agents.lab_analysis_agent import LabAnalysisAgent
from agents.llm_client import LLMConfigError, LLMTruncatedError
from agents.nutrition_agent import NutritionAgent
from agents.schemas import (
    ConsolidatedItems,
    LabAnalysisOutput,
    NutritionOutput,
    ShoppingTips,
    TrainingOutput,
)
from agents.training_agent import TrainingAgent
from fakes import FakeClient, message
import samples


@pytest.fixture(autouse=True)
def no_retrieval(monkeypatch):
    for cls in (LabAnalysisAgent, NutritionAgent, TrainingAgent, GroceryAgent):
        monkeypatch.setattr(cls, "_retrieve", lambda self, p, g, c: "", raising=False)


def _schema_title(call):
    return call["output_config"]["format"]["schema"].get("title")


# ── Lab Analysis ──────────────────────────────────────────────────────────────

def test_lab_analysis_uses_schema_and_limit():
    fake = FakeClient(samples.lab_output())
    llm_client.set_clients(sync=fake)
    out = LabAnalysisAgent().run(samples.profile(), samples.goals())
    assert isinstance(out, LabAnalysisOutput)
    assert len(out.analysed_markers) == 1
    assert fake.calls[0]["max_tokens"] == 4096
    assert _schema_title(fake.calls[0]) == "LabAnalysisOutput"
    prompt = fake.calls[0]["messages"][0]["content"]
    assert "Respond with ONLY valid JSON" not in prompt


def test_lab_analysis_no_markers_guard_clears_hallucinated_markers():
    fake = FakeClient(samples.lab_output(markers=True))
    llm_client.set_clients(sync=fake)
    out = LabAnalysisAgent().run(samples.profile(markers=False), samples.goals())
    assert out.analysed_markers == []
    assert out.dietary_constraints == ["Limit saturated fat"]


# ── Training ──────────────────────────────────────────────────────────────────

def test_training_uses_schema_and_limit():
    fake = FakeClient(samples.training_output())
    llm_client.set_clients(sync=fake)
    out = TrainingAgent().run(samples.profile(), samples.goals())
    assert isinstance(out, TrainingOutput)
    assert fake.calls[0]["max_tokens"] == 4096
    assert _schema_title(fake.calls[0]) == "TrainingOutput"
    assert "Respond with ONLY valid JSON" not in fake.calls[0]["messages"][0]["content"]


# ── Nutrition ─────────────────────────────────────────────────────────────────

def test_nutrition_merges_two_halves():
    half1 = {"days": [samples.day(d, 2000) for d in ("Monday", "Tuesday", "Wednesday", "Thursday")]}
    half2 = {"days": [samples.day(d, 2300) for d in ("Friday", "Saturday", "Sunday")]}
    fake = FakeClient(half1, half2)
    llm_client.set_clients(sync=fake)
    lab = LabAnalysisOutput(**samples.lab_output())
    out = NutritionAgent().run(samples.profile(), samples.goals(), {"lab_analysis": lab})
    assert isinstance(out, NutritionOutput)
    assert [d.day for d in out.weekly_plan][0] == "Monday"
    assert len(out.weekly_plan) == 7
    assert out.avg_daily_calories == (4 * 2000 + 3 * 2300) // 7
    assert out.avg_daily_protein_g == 150.0
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["max_tokens"] == 8192
        assert _schema_title(call) == "HalfWeekPlan"
        assert "Respond with ONLY valid JSON" not in call["messages"][0]["content"]


def test_nutrition_truncated_half_fails_agent():
    half1 = {"days": [samples.day(d) for d in ("Monday", "Tuesday", "Wednesday", "Thursday")]}
    fake = FakeClient(half1, message('{"days": [', stop_reason="max_tokens"))
    llm_client.set_clients(sync=fake)
    with pytest.raises(LLMTruncatedError):
        NutritionAgent().run(samples.profile(), samples.goals())
    assert len(fake.calls) == 2


def test_nutrition_too_few_days_fails():
    fake = FakeClient({"days": [samples.day("Monday")]}, {"days": [samples.day("Friday")]})
    llm_client.set_clients(sync=fake)
    with pytest.raises(ValueError, match="Expected 7 days"):
        NutritionAgent().run(samples.profile(), samples.goals())


# ── Grocery ───────────────────────────────────────────────────────────────────

def _nutrition(ingredients):
    days = []
    for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
        dd = samples.day(d)
        dd["meals"] = [samples.meal("Lunch", ingredients)]
        days.append(dd)
    return NutritionOutput(
        weekly_plan=days, avg_daily_calories=2000, avg_daily_protein_g=150,
        avg_daily_carbs_g=200, avg_daily_fat_g=60, caloric_strategy="x",
    )


def _grocery_reply(kwargs):
    title = _schema_title(kwargs)
    if title == "ShoppingTips":
        return {"tips": ["tip a", "tip b"]}
    assert title == "ConsolidatedItems"
    return {"items": [{"name": "Consolidated", "total_quantity": "700g", "estimated_cost_eur": 3.5}]}


def test_grocery_success_path():
    nutrition = _nutrition(["150g chicken breast", "80g oats"])
    fake = FakeClient(*([_grocery_reply] * 3))
    llm_client.set_clients(sync=fake)
    out = GroceryAgent().run(samples.profile(), samples.goals(), {"nutrition": nutrition})
    assert out.items_by_category["Proteins"][0].name == "Consolidated"
    assert out.items_by_category["Proteins"][0].category == "Proteins"
    assert out.items_by_category["Grains & Carbohydrates"][0].total_quantity == "700g"
    assert out.total_items == 2
    assert out.shopping_notes == ["tip a", "tip b"]
    assert all(c["max_tokens"] == 2048 for c in fake.calls)


@pytest.mark.parametrize("failure", [
    message('{"items": [', stop_reason="max_tokens"),
    RuntimeError("API overloaded"),
    {"items": "not-a-list"},
])
def test_grocery_falls_back_when_calls_fail(failure):
    nutrition = _nutrition(["150g chicken breast"])
    fake = FakeClient(failure, failure)
    llm_client.set_clients(sync=fake)
    out = GroceryAgent().run(samples.profile(), samples.goals(), {"nutrition": nutrition})
    proteins = out.items_by_category["Proteins"]
    assert proteins[0].name == "150g chicken breast"
    assert proteins[0].total_quantity == "see meal plan"
    assert out.shopping_notes == list(DEFAULT_SHOPPING_TIPS)


# ── Missing key ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("agent_cls", [LabAnalysisAgent, TrainingAgent, NutritionAgent])
def test_run_fails_naming_key_when_unset(monkeypatch, agent_cls):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        agent_cls().run(samples.profile(), samples.goals())


def test_grocery_fails_naming_key_when_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        GroceryAgent().run(samples.profile(), samples.goals(),
                           {"nutrition": _nutrition(["150g chicken breast"])})
