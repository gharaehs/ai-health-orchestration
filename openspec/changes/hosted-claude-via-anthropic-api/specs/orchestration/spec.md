# Spec Delta

## Purpose

Defines how one Orchestration schedules its agents to turn a Health Profile and Goals into a Lab Analysis, Meal Plan, Training Program and Grocery List, and how many Orchestrations may run at once.

## ADDED Requirements

### Requirement: Lab Analysis runs first
Each Orchestration SHALL run the Lab Analysis before any other agent. It SHALL make the Lab Analysis result available to the Nutrition and Training agents when it succeeds.

#### Scenario: Lab Analysis succeeds
- **WHEN** an Orchestration starts and the Lab Analysis succeeds
- **THEN** the Nutrition and Training agents both receive its calorie target and its Dietary and Training Constraints

#### Scenario: Lab Analysis fails
- **WHEN** the Lab Analysis fails
- **THEN** the Nutrition and Training agents still run with their fallback behavior, and the Lab Analysis failure appears in the agent results

### Requirement: Nutrition and Training run in parallel
After the Lab Analysis finishes, the Nutrition and Training agents SHALL run at the same time within the same Orchestration. A failure in one SHALL NOT stop or cancel the other.

#### Scenario: Concurrent execution
- **WHEN** the Lab Analysis has finished
- **THEN** the Nutrition and Training agents are both started before either of them completes

#### Scenario: Training fails
- **WHEN** the Training agent fails while the Nutrition agent succeeds
- **THEN** the Meal Plan and Grocery List are still produced, and the Training failure appears in the agent results

### Requirement: Grocery List depends on the Meal Plan
The Grocery agent SHALL start only after the Nutrition agent has produced a Meal Plan. When no Meal Plan was produced, the Grocery agent SHALL be reported as skipped.

#### Scenario: Meal Plan available
- **WHEN** the Nutrition agent succeeds
- **THEN** the Grocery agent runs on that Meal Plan

#### Scenario: Meal Plan missing
- **WHEN** the Nutrition agent fails
- **THEN** the Grocery agent's result is `skipped`, with a reason naming the missing Meal Plan

### Requirement: Agent results keep a stable order
The Orchestration result SHALL list agent results in the order Lab Analysis, Nutrition, Training, Grocery, whichever of Nutrition and Training finishes first.

#### Scenario: Training finishes first
- **WHEN** the Training agent completes before the Nutrition agent
- **THEN** the agent results are still listed as Lab Analysis, Nutrition, Training, Grocery

### Requirement: One Orchestration at a time
The API SHALL run at most one Orchestration at a time, as a cost guard. A job submitted while another is running SHALL stay `pending` until the running job ends, and SHALL then run.

#### Scenario: Second job waits
- **WHEN** a client starts an Orchestration while another one is `running`
- **THEN** the new job reports `pending` until the first job is `complete` or `failed`, and then moves to `running`
