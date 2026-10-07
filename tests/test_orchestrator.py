import threading

from agents.orchestrator import Orchestrator
from agents.schemas import AgentStatus

from test_agents import make_goals, make_profile

TIMEOUT = 5


class FakeAgent:
    def __init__(self, name, output=None, error=None, on_run=None):
        self.name = name
        self.output = output if output is not None else f"{name}-output"
        self.error = error
        self.on_run = on_run
        self.started = threading.Event()
        self.contexts: list[dict] = []

    def run(self, profile, goals, context=None):
        self.started.set()
        self.contexts.append(context)
        if self.on_run:
            self.on_run(context)
        if self.error:
            raise RuntimeError(self.error)
        return self.output


class LabOutput:
    dietary_constraints = ["a"]
    training_constraints = ["b"]
    recommended_calories = 2000


class NutritionOutput:
    avg_daily_calories = 2000
    avg_daily_protein_g = 150


class TrainingOutput:
    weekly_program = []
    rest_days = []


class GroceryOutput:
    total_items = 3
    estimated_weekly_cost_eur = None


def make_orchestrator(lab, nutrition, training, grocery) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)  # skip ChromaDB retriever setup
    orch.lab_agent, orch.nutrition_agent = lab, nutrition
    orch.training_agent, orch.grocery_agent = training, grocery
    return orch


def run(orch):
    return orch.run(make_profile(), make_goals())


def statuses(result):
    return {r.agent: r.status for r in result.agent_results}


def patch_result(monkeypatch):
    """Replace OrchestrationResult so fake (non-schema) outputs can be stored."""
    from agents import orchestrator as mod

    class LooseResult:
        def __init__(self):
            self.lab_analysis = self.nutrition = self.training = self.grocery = None
            self.agent_results = []
            self.total_duration_s = 0.0

    monkeypatch.setattr(mod, "OrchestrationResult", LooseResult)


def test_nutrition_and_training_start_concurrently(monkeypatch):
    patch_result(monkeypatch)
    nutrition = FakeAgent("nutrition", NutritionOutput())
    training = FakeAgent("training", TrainingOutput())
    # Each blocks until the other has started → deadlocks (times out) if sequential.
    nutrition.on_run = lambda ctx: assert_wait(training.started)
    training.on_run = lambda ctx: assert_wait(nutrition.started)
    grocery = FakeAgent("grocery", GroceryOutput())
    lab = FakeAgent("lab", LabOutput())

    result = run(make_orchestrator(lab, nutrition, training, grocery))

    assert all(s == AgentStatus.SUCCESS for s in statuses(result).values())
    assert [r.agent for r in result.agent_results] == [
        "LabAnalysisAgent", "NutritionAgent", "TrainingAgent", "GroceryAgent"]
    # Each concurrent agent gets its own shallow copy with the lab output.
    assert nutrition.contexts[0] is not training.contexts[0]
    assert nutrition.contexts[0]["lab_analysis"] is lab.output
    assert training.contexts[0]["lab_analysis"] is lab.output
    assert grocery.contexts[0]["nutrition"] is nutrition.output


def assert_wait(event):
    assert event.wait(TIMEOUT), "other agent never started — agents are not concurrent"


def test_nutrition_failure_skips_grocery_and_keeps_training(monkeypatch):
    patch_result(monkeypatch)
    grocery = FakeAgent("grocery")
    result = run(make_orchestrator(
        FakeAgent("lab", LabOutput()),
        FakeAgent("nutrition", error="boom"),
        FakeAgent("training", TrainingOutput()),
        grocery,
    ))
    st = statuses(result)
    assert st["NutritionAgent"] == AgentStatus.FAILED
    assert st["GroceryAgent"] == AgentStatus.SKIPPED
    assert st["TrainingAgent"] == AgentStatus.SUCCESS
    assert result.training is not None and result.nutrition is None
    assert not grocery.started.is_set()
    errors = {r.agent: r.error for r in result.agent_results}
    assert "boom" in errors["NutritionAgent"]


def test_lab_failure_still_runs_nutrition_and_training(monkeypatch):
    patch_result(monkeypatch)
    nutrition = FakeAgent("nutrition", NutritionOutput())
    training = FakeAgent("training", TrainingOutput())
    result = run(make_orchestrator(
        FakeAgent("lab", error="lab down"), nutrition, training, FakeAgent("grocery", GroceryOutput()),
    ))
    st = statuses(result)
    assert st["LabAnalysisAgent"] == AgentStatus.FAILED
    assert st["NutritionAgent"] == AgentStatus.SUCCESS
    assert st["TrainingAgent"] == AgentStatus.SUCCESS
    assert st["GroceryAgent"] == AgentStatus.SUCCESS
    assert "lab_analysis" not in nutrition.contexts[0]
    assert "lab_analysis" not in training.contexts[0]


def test_training_failure_isolated(monkeypatch):
    patch_result(monkeypatch)
    result = run(make_orchestrator(
        FakeAgent("lab", LabOutput()),
        FakeAgent("nutrition", NutritionOutput()),
        FakeAgent("training", error="training broke"),
        FakeAgent("grocery", GroceryOutput()),
    ))
    st = statuses(result)
    assert st["TrainingAgent"] == AgentStatus.FAILED
    assert st["NutritionAgent"] == AgentStatus.SUCCESS
    assert st["GroceryAgent"] == AgentStatus.SUCCESS
    assert result.nutrition is not None and result.grocery is not None
    assert all(r.duration_s >= 0 for r in result.agent_results)
