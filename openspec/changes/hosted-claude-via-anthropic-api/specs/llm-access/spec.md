# Spec Delta

## Purpose

Defines how the system gets completions from its language model. Every Orchestration agent, the RAG pipeline and the chat comparison use one hosted Claude model, configured through server-side secrets, with fixed generation settings and schema-checked agent output.

## ADDED Requirements

### Requirement: Single hosted model provider
The system SHALL send every LLM request to Claude through the Anthropic API. It SHALL NOT offer a runtime option to switch to another provider or to a self-hosted model.

#### Scenario: Orchestration agents use the hosted model
- **WHEN** an Orchestration runs and produces a Lab Analysis, Meal Plan, Training Program and Grocery List
- **THEN** every LLM request made during that Orchestration goes to the Anthropic API

#### Scenario: Chat comparison keeps both calls on the hosted model
- **WHEN** a client calls `/api/chat`
- **THEN** the system makes two LLM requests, one base and one RAG-augmented, both to the Anthropic API
- **AND** the response still contains both answers and the grounding metrics

#### Scenario: Legacy RAG pipeline uses the hosted model
- **WHEN** an operator runs the RAG test CLI to generate a plan
- **THEN** the generation request goes to the Anthropic API

#### Scenario: No GPU or local model required
- **WHEN** the stack is started with Docker Compose
- **THEN** no service requires a GPU, a local model download or a local inference server

### Requirement: Model and credentials from configuration
The system SHALL read the API key from `ANTHROPIC_API_KEY`. It SHALL read the model ID from `ANTHROPIC_MODEL`, which defaults to `claude-opus-5-5`. The API key SHALL NOT be committed to the repository, logged, or returned in any API response.

#### Scenario: Default model
- **WHEN** `ANTHROPIC_MODEL` is not set
- **THEN** LLM requests use the model ID `claude-opus-5-5`

#### Scenario: Model override
- **WHEN** `ANTHROPIC_MODEL` is set to another Claude model ID
- **THEN** LLM requests use that model ID

#### Scenario: Key never exposed
- **WHEN** the API logs an LLM request or error, or returns `/api/health`
- **THEN** the output does not contain the value of `ANTHROPIC_API_KEY`

### Requirement: Missing API key fails clearly
When `ANTHROPIC_API_KEY` is missing or empty and an LLM request is attempted, the system SHALL fail that request with an error that names the missing variable. It SHALL NOT fall back to another backend.

#### Scenario: Orchestration without a key
- **WHEN** `ANTHROPIC_API_KEY` is unset and an Orchestration job is started
- **THEN** the job ends with status `failed`, and its error message names `ANTHROPIC_API_KEY`

#### Scenario: Health check without a key
- **WHEN** `ANTHROPIC_API_KEY` is unset and a client calls `/api/health`
- **THEN** the `llm` service entry has status `error`, a detail that names `ANTHROPIC_API_KEY`, and overall status `degraded`

### Requirement: Fixed generation settings
Every LLM request SHALL set the effort level to `medium`. It SHALL NOT send `temperature`, `top_p`, `top_k` or a `thinking` configuration, because the default model rejects non-default sampling and cannot disable thinking. These settings SHALL NOT be configurable at runtime.

#### Scenario: Request settings
- **WHEN** any component sends an LLM request
- **THEN** the request sets effort `medium` and contains no `temperature`, `top_p`, `top_k` or `thinking` field

### Requirement: Schema-constrained agent output
Each agent LLM request SHALL constrain the model's reply to a JSON schema generated from that call's Pydantic model, using the API's structured-output feature. The reply SHALL then be validated with that Pydantic model. Agents SHALL NOT extract JSON from free text by regex or brace matching, and SHALL NOT rely on forced tool use.

#### Scenario: Valid structured reply
- **WHEN** an agent receives a reply that conforms to its schema
- **THEN** the agent gets a validated Pydantic object, without any text-level JSON extraction

#### Scenario: Pydantic validation fails
- **WHEN** a reply conforms to the JSON schema but fails Pydantic validation, and retries remain
- **THEN** the agent sends a new LLM request
- **AND** when no retries remain, the agent fails with the validation error

#### Scenario: Prompts no longer carry JSON-format instructions
- **WHEN** an agent builds its prompt
- **THEN** the prompt contains no "respond with JSON only" instruction and no JSON schema example, because the schema is sent through structured output

### Requirement: Per-agent output limits
Each agent SHALL have its own maximum output tokens for each LLM call. The Meal Plan's limit SHALL be about 8k tokens, and every other agent's limit SHALL be smaller than the Meal Plan's.

#### Scenario: Meal Plan limit
- **WHEN** the Nutrition agent requests a Meal Plan
- **THEN** each of its LLM requests uses an output limit of 8192 tokens

#### Scenario: Smaller limits elsewhere
- **WHEN** the Lab Analysis, Training or Grocery agent sends an LLM request
- **THEN** that request's output limit is below 8192 tokens

### Requirement: Truncated replies are hard errors
When the model stops because it reached the output limit, the system SHALL raise an error that names the truncation. It SHALL NOT parse or return the partial reply, and agents SHALL NOT retry the truncated call.

#### Scenario: Agent reply truncated
- **WHEN** an agent's LLM reply stops at its output limit
- **THEN** that agent fails with a truncation error, without retrying, and its `AgentResult` reports the failure

#### Scenario: Comparator reply truncated
- **WHEN** a `/api/chat` reply stops at its output limit
- **THEN** that response has empty content and an `error` field that names the truncation

### Requirement: Upstream errors surface to the caller
Errors from the Anthropic API, including authentication failure, rate limiting, overload, timeout, refusal and invalid model, SHALL be logged without the API key. They SHALL then be raised or returned to the caller, and SHALL NOT be returned as model output.

#### Scenario: Authentication failure during Orchestration
- **WHEN** the Anthropic API rejects the key during an Orchestration
- **THEN** the affected agents report failure, and the job result reflects it

#### Scenario: Error during chat comparison
- **WHEN** the Anthropic API returns an error for a `/api/chat` request
- **THEN** the affected response has empty content and a non-null `error` field, matching the existing comparator response shape

### Requirement: Health check reports the model dependency
`/api/health` SHALL report an `llm` service entry in place of `vllm`. The check SHALL confirm that `ANTHROPIC_API_KEY` is set and SHALL fetch the configured model's metadata, spending no generation tokens. The entry SHALL include the model ID when the status is `ok`.

#### Scenario: Healthy model dependency
- **WHEN** the key is valid, the configured model exists and ChromaDB is healthy
- **THEN** `/api/health` returns overall status `ok` and `services.llm` = `{"status": "ok", "model": "<configured model id>"}`

#### Scenario: Invalid model ID
- **WHEN** `ANTHROPIC_MODEL` names a model that the key cannot access
- **THEN** `services.llm.status` is `error` and overall status is `degraded`

#### Scenario: No tokens spent
- **WHEN** a client calls `/api/health`
- **THEN** no message-generation request is sent to the Anthropic API

#### Scenario: Legacy key removed
- **WHEN** a client calls `/api/health`
- **THEN** the response contains no `vllm` service entry

### Requirement: LLM smoke test
The project SHALL provide a `make test-llm` command. It SHALL make one small real request to the configured model using the local `ANTHROPIC_API_KEY`, and SHALL print the model's reply or the error.

#### Scenario: Smoke test succeeds
- **WHEN** a valid key is configured and the operator runs `make test-llm`
- **THEN** the command prints the model's reply and exits with code 0

#### Scenario: Smoke test without a key
- **WHEN** no key is configured and the operator runs `make test-llm`
- **THEN** the command prints an error that names `ANTHROPIC_API_KEY` and exits with a non-zero code

### Requirement: Offline unit tests
`make test` SHALL run without network access and without a real API key, using a mocked Anthropic client. Live model calls SHALL happen only in `make test-llm` and `scripts/test_orchestration.py`.

#### Scenario: Tests pass offline
- **WHEN** `make test` runs with no network access and no `ANTHROPIC_API_KEY`
- **THEN** all tests pass

#### Scenario: Accidental live call is caught
- **WHEN** a unit test would open a real network connection
- **THEN** that test fails instead of reaching the Anthropic API
