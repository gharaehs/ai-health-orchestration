# AI Health Orchestration

Turns a person's health data and fitness goals into a coordinated Meal Plan, Training Program and Grocery List.

## Inputs

**Health Profile**:
A person's demographics, body composition, Blood Markers, medical history, medications and allergies.
_Avoid_: User profile, patient record

**Blood Marker**:
A single lab test result (name, value, unit) within a Health Profile.
_Avoid_: Lab value, biomarker

**Goals**:
What the person wants to achieve and how they can train: primary goal, activity level, fitness level, training days per week and dietary preferences.
_Avoid_: User goals, objectives

## Outputs

**Lab Analysis**:
An interpretation of the Blood Markers that yields a calorie target and the Dietary and Training Constraints.
_Avoid_: Blood work, lab report

**Dietary Constraint**:
A rule derived from the Lab Analysis that the Meal Plan must respect.
_Avoid_: Diet restriction

**Training Constraint**:
A rule derived from the Lab Analysis that the Training Program must respect.
_Avoid_: Exercise restriction

**Meal Plan**:
A 7-day schedule of meals that meets the calorie target and the Dietary Constraints.
_Avoid_: Nutrition plan, diet plan

**Training Program**:
A weekly schedule of training sessions with a progression scheme that respects the Training Constraints.
_Avoid_: Gym program, workout plan

**Grocery List**:
The ingredients of a Meal Plan, combined and grouped by category for one week of shopping.
_Avoid_: Shopping list

## Process

**Orchestration**:
One full run that turns a Health Profile and Goals into a Lab Analysis, Meal Plan, Training Program and Grocery List.
_Avoid_: Pipeline run, job
