# Data Model: Migrate to Company AWS Account with Claude Opus 5.5

**Feature**: [spec.md](./spec.md) | **Date**: 2026-10-07

## Unchanged domain contracts (FR-003, Principle I)

All agent input and output models in `agents/schemas.py` stay **byte-for-byte unchanged**:

| Model | Role |
|-------|------|
| `HealthProfile`, `ScaleMetrics`, `BloodMarker`, `UserGoals` (+ enums) | Pipeline input |
| `LabAnalysisOutput`, `AnalysedMarker` | Agent 1 output → constraints for Agents 2 & 3 |
| `NutritionOutput`, `DayPlan`, `Meal` | Agent 2 output (7 days) → input for Agent 4 |
| `TrainingOutput`, `TrainingSession`, `Exercise`, `ProgressionScheme` | Agent 3 output |
| `GroceryOutput`, `GroceryItem` | Agent 4 output |
| `AgentResult`, `AgentStatus`, `OrchestrationResult` | Orchestrator envelope returned by `/api/orchestrate/result/{job_id}` |

Validation rule: every agent result MUST pass `Model.model_validate(...)` for its output model.
The agent's existing `_parse_output` already does this; this feature does not change it.

## New internal entities (shared LLM client, `llm/`)

These are runtime types used inside the Python process. They are not inter-agent contracts, so
they do not belong in `agents/schemas.py`. Agents still exchange data only through the schemas
listed above.

### LLMSettings

Configuration resolved once from the environment at process start.

| Field | Type | Source (env) | Default | Rule |
|-------|------|--------------|---------|------|
| `model_id` | str | `BEDROCK_MODEL_ID` | — | **Required**. If missing, raise `LLMConfigError` at startup |
| `region` | str | `AWS_REGION` | — | **Required** |
| `default_max_tokens` | int | `LLM_MAX_TOKENS` | 4096 | > 0 |
| `connect_timeout_s` | int | `LLM_CONNECT_TIMEOUT` | 10 | > 0 |
| `read_timeout_s` | int | `LLM_READ_TIMEOUT` | 300 | > 0 |
| `max_attempts` | int | `LLM_MAX_ATTEMPTS` | 4 | ≥ 1 (botocore adaptive retries) |

Credentials are **not** part of settings. They come from the default AWS credential chain
(the instance IAM role).

### LLMRequest (call arguments)

| Field | Type | Rule |
|-------|------|------|
| `system` | str | Optional system prompt |
| `user` | str | Required, non-empty |
| `max_tokens` | int \| None | Falls back to `default_max_tokens` |
| `temperature` | float \| None | 0.0–1.0; `top_p` is never sent |
| `caller` | str | Label for logs (e.g. `NutritionAgent`, `compare:rag`) |

### LLMResponse

| Field | Type | Notes |
|-------|------|-------|
| `text` | str | Concatenated text blocks of the assistant message, stripped |
| `stop_reason` | str | Bedrock `stopReason` (`end_turn`, `max_tokens`, …) |
| `input_tokens` | int | From `usage.inputTokens` |
| `output_tokens` | int | From `usage.outputTokens` |
| `latency_s` | float | Wall-clock time for the call |
| `model_id` | str | Model or profile actually invoked |

### Error types (raised by the client)

| Error | When | Agent / job effect |
|-------|------|--------------------|
| `LLMConfigError` | Missing or invalid settings, or no credentials resolvable | Startup or health reports `error`; job `failed` |
| `LLMTruncatedError` | `stop_reason == "max_tokens"` | Counts as a failed attempt in the agent retry loop; job `failed` after the last attempt |
| `LLMUnavailableError` | Throttling, timeout or 5xx persisting after botocore retries; access denied | Job `failed` with message (FR-010) |

All three subclass `LLMError`.

### RetrievalUnavailableError (agents)

Raised by the agents' MCP `_retrieve()` overrides when the MCP RAG server call fails (R11).
The orchestrator marks the agent `AgentStatus` failed, and the job status becomes `failed`.

## State transitions: orchestration job (unchanged states)

```text
pending ──► running ──► complete
                   └──► failed   (any LLMError, RetrievalUnavailableError, or parse failure after retries)
```

No new states are added. Failure messages now name the cause (for example
`LLMUnavailableError: ThrottlingException after 4 attempts`).

## Migrated data (not code, not in git — Principle IV)

| Data | Location | Integrity check |
|------|----------|-----------------|
| Vector store | `data/chroma/` | 4 collections; 13,349 docs total; per-collection counts equal source |
| Corpus files | `data/corpus/` | File count and total size equal source |
