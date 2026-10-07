# Spec Delta

## Purpose

Defines how asynchronous orchestration jobs are submitted, executed concurrently up to a configured limit, and reported to clients through polling.

## ADDED Requirements

### Requirement: Asynchronous job submission
`POST /api/orchestrate` SHALL accept a health profile and goals, return a job ID with status `pending` immediately, and run the pipeline in the background.

#### Scenario: Job accepted
- **WHEN** a client submits a valid orchestration request
- **THEN** the response contains a `job_id` and `status = "pending"` without waiting for the pipeline to finish

### Requirement: Bounded concurrent job execution
The system SHALL run multiple orchestration jobs concurrently, up to a limit set by an environment variable. Jobs submitted beyond the limit SHALL wait as `pending` until a slot frees up.

#### Scenario: Jobs within limit run in parallel
- **WHEN** two jobs are submitted and the concurrency limit is at least 2
- **THEN** both jobs reach status `running` without either waiting for the other to complete

#### Scenario: Jobs beyond limit queue
- **WHEN** more jobs are submitted than the concurrency limit allows
- **THEN** the excess jobs stay `pending` until a running job completes or fails

### Requirement: Job status and result retrieval
Clients SHALL be able to poll job status (`pending`, `running`, `complete`, `failed`) and duration by job ID, and fetch the full orchestration result once the job is complete.

#### Scenario: Completed job result
- **WHEN** a client requests the result of a job whose status is `complete`
- **THEN** the full orchestration result is returned

#### Scenario: Unknown job
- **WHEN** a client requests the status or result of a job ID that does not exist
- **THEN** the API responds with HTTP 404
