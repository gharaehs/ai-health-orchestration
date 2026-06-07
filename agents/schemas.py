"""
agents/schemas.py

Pydantic models defining the I/O contracts for every agent in the pipeline.

Data flow:
  HealthProfile + UserGoals
      → LabAnalysisOutput
          → NutritionOutput
              → GroceryOutput
      → TrainingOutput
      → OrchestrationResult  (all four combined)

Every model that an agent returns must be JSON-serialisable so it can be
logged, stored, and forwarded to downstream agents as structured context.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class FitnessGoal(str, Enum):
    FAT_LOSS            = "fat_loss"
    MUSCLE_GAIN         = "muscle_gain"
    MAINTENANCE         = "maintenance"
    PERFORMANCE         = "performance"


class ActivityLevel(str, Enum):
    SEDENTARY           = "sedentary"        # desk job, no exercise
    LIGHTLY_ACTIVE      = "lightly_active"   # 1-3 days/week
    MODERATELY_ACTIVE   = "moderately_active" # 3-5 days/week
    VERY_ACTIVE         = "very_active"      # 6-7 days/week
    EXTRA_ACTIVE        = "extra_active"     # physical job + training


class FitnessLevel(str, Enum):
    BEGINNER            = "beginner"         # < 1 year training
    INTERMEDIATE        = "intermediate"     # 1-3 years
    ADVANCED            = "advanced"         # 3+ years


class MarkerStatus(str, Enum):
    OPTIMAL             = "optimal"
    BORDERLINE          = "borderline"
    BORDERLINE_HIGH     = "borderline-high"
    BORDERLINE_LOW      = "borderline-low"
    ELEVATED            = "elevated"
    HIGH                = "high"
    LOW                 = "low"
    DEFICIENT           = "deficient"
    CRITICAL            = "critical"


# ── Pipeline Inputs ───────────────────────────────────────────────────────────

class BloodMarker(BaseModel):
    """A single blood test result with its unit."""
    name:   str   = Field(..., description="Marker name, e.g. 'LDL Cholesterol'")
    value:  float = Field(..., description="Numerical result")
    unit:   str   = Field(..., description="Unit string, e.g. 'mmol/L'")


class ScaleMetrics(BaseModel):
    """Body composition from a smart scale."""
    weight_kg:          float
    body_fat_pct:       Optional[float] = None
    muscle_mass_kg:     Optional[float] = None
    bmi:                Optional[float] = None


class HealthProfile(BaseModel):
    """
    The full user health profile ingested at the start of the pipeline.
    This is the single source of truth passed to every agent.
    """
    # Demographics
    age:                int   = Field(..., ge=18, le=100)
    sex:                str   = Field(..., description="'male' or 'female'")
    height_cm:          float = Field(..., gt=100, lt=250)

    # Body composition
    scale_metrics:      ScaleMetrics

    # Lab results (optional — system works without them)
    blood_markers:      list[BloodMarker] = Field(default_factory=list)

    # Medical context
    medical_history:    list[str] = Field(
        default_factory=list,
        description="Free-text conditions, e.g. ['Type 2 diabetes', 'Hypertension']"
    )
    medications:        list[str] = Field(
        default_factory=list,
        description="Current medications that may affect nutrition/training"
    )
    allergies:          list[str] = Field(
        default_factory=list,
        description="Food allergies or intolerances"
    )


class UserGoals(BaseModel):
    """User-defined fitness and lifestyle objectives."""
    primary_goal:       FitnessGoal
    activity_level:     ActivityLevel
    fitness_level:      FitnessLevel
    training_days_per_week: int = Field(..., ge=1, le=7)
    dietary_preferences: list[str] = Field(
        default_factory=list,
        description="e.g. ['vegetarian', 'low-carb', 'halal']"
    )
    notes:              Optional[str] = None


# ── Agent 1 Output: Lab Analysis ──────────────────────────────────────────────

class AnalysedMarker(BaseModel):
    """Interpretation of a single blood marker."""
    name:           str
    value:          float
    unit:           str
    status:         MarkerStatus
    reference_range: str           = Field(..., description="e.g. '< 2.6 mmol/L (optimal)'")
    interpretation: str            = Field(..., description="One-sentence clinical meaning")
    dietary_implication: Optional[str] = Field(
        None,
        description="How this result should influence nutrition recommendations"
    )
    training_implication: Optional[str] = Field(
        None,
        description="How this result should influence training recommendations"
    )


class LabAnalysisOutput(BaseModel):
    """
    Output of the Lab Analysis Agent.
    Passed as constraints to both Nutrition Agent and Training Agent.
    """
    analysed_markers:   list[AnalysedMarker] = Field(default_factory=list)

    # Derived constraint summaries consumed by downstream agents
    dietary_constraints:  list[str] = Field(
        default_factory=list,
        description="Hard constraints for Nutrition Agent, e.g. 'Limit saturated fat (elevated LDL)'"
    )
    training_constraints: list[str] = Field(
        default_factory=list,
        description="Hard constraints for Training Agent, e.g. 'Avoid high-intensity cardio (elevated CRP)'"
    )

    # Calculated targets
    estimated_tdee:     Optional[int] = Field(
        None,
        description="Total Daily Energy Expenditure in kcal, calculated from profile + activity"
    )
    recommended_calories: Optional[int] = Field(
        None,
        description="Adjusted caloric target based on goal (deficit/surplus/maintenance)"
    )

    overall_health_summary: str = Field(
        ...,
        description="2-3 sentence narrative summary of the user's health status"
    )

    sources_used:       list[str] = Field(
        default_factory=list,
        description="RAG source references used to ground this analysis"
    )


# ── Agent 2 Output: Nutrition ─────────────────────────────────────────────────

class Meal(BaseModel):
    """A single meal within a day."""
    name:           str             = Field(..., description="e.g. 'Breakfast', 'Lunch'")
    recipe_name:    str
    ingredients:    list[str]       = Field(..., description="e.g. ['100g oats', '200ml milk']")
    instructions:   str             = Field(..., description="Brief preparation steps")
    calories_kcal:  int
    protein_g:      float
    carbs_g:        float
    fat_g:          float
    notes:          Optional[str]   = None


class DayPlan(BaseModel):
    """One day's complete meal plan."""
    day:            str             = Field(..., description="e.g. 'Monday'")
    meals:          list[Meal]
    total_calories: int
    total_protein_g: float
    total_carbs_g:  float
    total_fat_g:    float


class NutritionOutput(BaseModel):
    """
    Output of the Nutrition Agent.
    Full 7-day meal plan passed downstream to the Grocery Agent.
    """
    weekly_plan:            list[DayPlan]   = Field(..., min_length=7, max_length=7)

    # Weekly averages
    avg_daily_calories:     int
    avg_daily_protein_g:    float
    avg_daily_carbs_g:      float
    avg_daily_fat_g:        float

    # Rationale
    caloric_strategy:       str = Field(
        ...,
        description="Explanation of the caloric target and macro split chosen"
    )
    constraint_adherence:   list[str] = Field(
        default_factory=list,
        description="How each lab-derived dietary constraint was addressed"
    )
    sources_used:           list[str] = Field(default_factory=list)


# ── Agent 3 Output: Training ──────────────────────────────────────────────────

class Exercise(BaseModel):
    """A single exercise within a session."""
    name:           str
    sets:           int
    reps:           str             = Field(..., description="e.g. '8-12' or '3x60s'")
    rest_seconds:   int
    notes:          Optional[str]   = None  # e.g. "Use RPE 7-8"


class TrainingSession(BaseModel):
    """One training session."""
    day:            str             = Field(..., description="e.g. 'Monday'")
    session_type:   str             = Field(..., description="e.g. 'Upper Body Strength'")
    duration_minutes: int
    exercises:      list[Exercise]
    warmup:         Optional[str]   = None
    cooldown:       Optional[str]   = None


class ProgressionScheme(BaseModel):
    """Week-over-week progression guidance."""
    principle:      str             = Field(..., description="e.g. 'Linear periodisation'")
    week_2_adjustment: str
    week_3_adjustment: str
    week_4_adjustment: str          = Field(..., description="Usually a deload week")


class TrainingOutput(BaseModel):
    """Output of the Training Agent."""
    weekly_program:         list[TrainingSession]
    rest_days:              list[str]               = Field(..., description="e.g. ['Wednesday', 'Sunday']")
    progression_scheme:     ProgressionScheme
    constraint_adherence:   list[str]               = Field(
        default_factory=list,
        description="How each lab-derived training constraint was addressed"
    )
    sources_used:           list[str]               = Field(default_factory=list)


# ── Agent 4 Output: Grocery ───────────────────────────────────────────────────

class GroceryItem(BaseModel):
    """A single aggregated grocery item."""
    name:           str
    total_quantity: str             = Field(..., description="e.g. '700g' or '2 litres'")
    category:       str             = Field(..., description="e.g. 'Proteins', 'Grains', 'Dairy'")
    estimated_cost_eur: Optional[float] = None


class GroceryOutput(BaseModel):
    """Output of the Grocery Agent — aggregated weekly shopping list."""
    items_by_category:  dict[str, list[GroceryItem]]  = Field(
        ...,
        description="Keys are category names, values are lists of items"
    )
    total_items:        int
    shopping_notes:     list[str] = Field(
        default_factory=list,
        description="e.g. 'Buy chicken in bulk and portion into 150g servings'"
    )
    estimated_weekly_cost_eur: Optional[float] = None


# ── Final Orchestration Result ────────────────────────────────────────────────

class AgentStatus(str, Enum):
    SUCCESS     = "success"
    FAILED      = "failed"
    SKIPPED     = "skipped"


class AgentResult(BaseModel):
    """Wrapper for a single agent's execution result."""
    agent:          str
    status:         AgentStatus
    duration_s:     float
    error:          Optional[str]   = None


class OrchestrationResult(BaseModel):
    """
    The final output of the full multi-agent pipeline.
    Contains all four agent outputs plus execution metadata.
    """
    # Agent outputs
    lab_analysis:   Optional[LabAnalysisOutput]     = None
    nutrition:      Optional[NutritionOutput]        = None
    training:       Optional[TrainingOutput]         = None
    grocery:        Optional[GroceryOutput]          = None

    # Execution metadata
    agent_results:  list[AgentResult]               = Field(default_factory=list)
    total_duration_s: float                         = 0.0
    pipeline_version: str                           = "1.0"