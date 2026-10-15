import threading
import time

import pytest

from agents import orchestrator as orch_module
from agents.orchestrator import Orchestrator
from agents.schemas import AgentStatus, LabAnalysisOutput
import samples


class _DummyRetriever:
    def __init__(self, *args, **kwargs):
        pass


class StubAgent:
    def __init__(self, output=None, error=None, before=None):
        self.output = output
        self.error = error
        self.before = before
        self.contexts = []

    def run(self, profile, goals, context=None):
        self.contexts.append(context)
        if self.before:
            self.before()
        if self.error:
            raise self.error
        return self.output


@pytest.fixture
def make_orchestrator(monkeypatch):
    monkeypatch.setattr(orch_module, "HealthRetriever", _DummyRetriever)

    def _make(nutrition=None, training=None, grocery=None, lab=None):
        o = Orchestrator()
        o.lab_agent = lab or StubAgent(LabAnalysisOutput(**samples.lab_output()))
        o.nutrition_agent = nutrition or StubAgent(_nutrition_output())
        o.training_agent = training or StubAgent(_training_output())
        o.grocery_agent = grocery or StubAgent(_grocery_output())
        return o
    return _make


def _nutrition_output():
    from agents.schemas import NutritionOutput
    return NutritionOutput(
        weekly_plan=[samples.day(d) for d in
                     ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")],
        avg_daily_calories=2000, avg_daily_protein_g=150, avg_daily_carbs_g=200,
        avg_daily_fat_g=60, caloric_strategy="x",
    )


def _training_output():
    from agents.schemas import TrainingOutput
    return TrainingOutput(**samples.training_output())


def _grocery_output():
    from agents.schemas import GroceryOutput
    return GroceryOutput(items_by_category={}, total_items=0)


def _names(result):
    return [r.agent for r in result.agent_results]


def test_nutrition_and_training_run_concurrently(make_orchestrator):
    barrier = threading.Barrier(2, timeout=5)
    nutrition = StubAgent(_nutrition_output(), before=barrier.wait)
    training = StubAgent(_training_output(), before=barrier.wait)
    result = make_orchestrator(nutrition=nutrition, training=training).run(
        samples.profile(), samples.goals())
    statuses = {r.agent: r.status for r in result.agent_results}
    assert statuses["NutritionAgent"] == AgentStatus.SUCCESS, result.agent_results
    assert statuses["TrainingAgent"] == AgentStatus.SUCCESS, result.agent_results


def test_each_parallel_agent_gets_own_context_copy(make_orchestrator):
    def mutate():
        nutrition.contexts[-1]["leak"] = True
    nutrition = StubAgent(_nutrition_output(), before=mutate)
    training = StubAgent(_training_output(), before=lambda: time.sleep(0.05))
    make_orchestrator(nutrition=nutrition, training=training).run(samples.profile(), samples.goals())
    assert nutrition.contexts[0] is not training.contexts[0]
    assert "leak" not in training.contexts[0]
    assert "lab_analysis" in training.contexts[0]


def test_training_failure_does_not_stop_nutrition_or_grocery(make_orchestrator):
    result = make_orchestrator(training=StubAgent(error=RuntimeError("boom"))).run(
        samples.profile(), samples.goals())
    statuses = {r.agent: r.status for r in result.agent_results}
    assert statuses == {
        "LabAnalysisAgent": AgentStatus.SUCCESS,
        "NutritionAgent": AgentStatus.SUCCESS,
        "TrainingAgent": AgentStatus.FAILED,
        "GroceryAgent": AgentStatus.SUCCESS,
    }
    assert result.nutrition is not None and result.grocery is not None
    assert result.training is None


def test_grocery_skipped_when_nutrition_fails(make_orchestrator):
    grocery = StubAgent(_grocery_output())
    result = make_orchestrator(nutrition=StubAgent(error=RuntimeError("no plan")),
                               grocery=grocery).run(samples.profile(), samples.goals())
    g = result.agent_results[-1]
    assert g.agent == "GroceryAgent"
    assert g.status == AgentStatus.SKIPPED
    assert "NutritionAgent" in g.error
    assert grocery.contexts == []


def test_grocery_receives_meal_plan(make_orchestrator):
    grocery = StubAgent(_grocery_output())
    make_orchestrator(grocery=grocery).run(samples.profile(), samples.goals())
    assert grocery.contexts[0]["nutrition"] is not None


def test_result_order_fixed_when_training_finishes_first(make_orchestrator):
    training_done = threading.Event()
    nutrition = StubAgent(_nutrition_output(), before=lambda: training_done.wait(5))

    def training_before():
        training_done.set()
    training = StubAgent(_training_output(), before=training_before)
    result = make_orchestrator(nutrition=nutrition, training=training).run(
        samples.profile(), samples.goals())
    assert _names(result) == ["LabAnalysisAgent", "NutritionAgent", "TrainingAgent", "GroceryAgent"]


def test_lab_failure_still_runs_downstream(make_orchestrator):
    result = make_orchestrator(lab=StubAgent(error=RuntimeError("lab down"))).run(
        samples.profile(), samples.goals())
    assert result.agent_results[0].status == AgentStatus.FAILED
    assert _names(result) == ["LabAnalysisAgent", "NutritionAgent", "TrainingAgent", "GroceryAgent"]
