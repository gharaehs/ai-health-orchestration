# Spec Delta

## Purpose

Defines how the specialist agents (Lab Analysis, Nutrition, Training, Grocery) turn a health profile and goals into schema-valid structured outputs, and the order in which they run.

## ADDED Requirements

### Requirement: Schema-enforced agent outputs
Each agent SHALL request its output from the model as structured data that conforms to that agent's output schema, and SHALL validate the returned data against the schema before accepting it. Free-text JSON extraction from prose SHALL NOT be relied on.

#### Scenario: Valid structured output
- **WHEN** an agent receives a model response whose structured data conforms to its output schema
- **THEN** the agent returns the validated output object

#### Scenario: Schema violation
- **WHEN** the returned structured data fails schema validation
- **THEN** the agent retries up to its configured attempt limit, and then reports a failure that includes the validation error

### Requirement: Single-request weekly meal plan
The Nutrition agent SHALL generate the complete 7-day meal plan, Monday through Sunday, in a single model request.

#### Scenario: Full week in one call
- **WHEN** the Nutrition agent runs
- **THEN** it makes exactly one generation request for the meal plan, and the output contains all seven days

### Requirement: Agent execution order
Lab Analysis SHALL run first. Nutrition and Training SHALL then run concurrently, using the Lab Analysis output when it is available. Grocery SHALL run after Nutrition completes successfully.

#### Scenario: Concurrent nutrition and training
- **WHEN** Lab Analysis completes
- **THEN** the Nutrition and Training agents start without waiting for each other

#### Scenario: Lab analysis fails
- **WHEN** the Lab Analysis agent fails
- **THEN** Nutrition and Training still run using fallback calorie targets and no lab-derived constraints

#### Scenario: Grocery depends on nutrition
- **WHEN** the Nutrition agent fails
- **THEN** the Grocery agent is reported as skipped, and the Training result is still returned

### Requirement: Agent failure isolation
A failure in one agent SHALL NOT prevent independent agents from completing. The orchestration result SHALL report a status, duration and error for each agent.

#### Scenario: Training fails independently
- **WHEN** the Training agent fails and the Nutrition agent succeeds
- **THEN** the result contains the nutrition plan and grocery list, and Training is marked failed with its error
