# Spec Delta

## Purpose

Defines how the system accesses its language model: a managed Claude model on Amazon Bedrock, with configuration, authentication, resilience to throttling and health reporting.

## ADDED Requirements

### Requirement: Managed model provider
All language-model generation in the system SHALL be performed by a Claude model hosted on Amazon Bedrock. The system SHALL NOT depend on a locally hosted model server or GPU.

#### Scenario: Agent generation uses Bedrock
- **WHEN** any agent, the RAG pipeline or the base-vs-RAG comparison requests a completion
- **THEN** the request is served by the configured Bedrock model

#### Scenario: No local model service required
- **WHEN** the system is started with its standard deployment configuration
- **THEN** it starts and serves requests without any local LLM container, model weights or GPU

### Requirement: Environment-based model configuration
The model ID, AWS region, maximum output tokens and request timeout SHALL be configurable through environment variables. Documented defaults SHALL target Claude Opus 5.5.

#### Scenario: Model overridden by environment
- **WHEN** the model ID environment variable is set to a different Bedrock model ID
- **THEN** all generation requests use that model ID without code changes

#### Scenario: Defaults applied
- **WHEN** no model environment variables are set
- **THEN** the system uses the documented Claude Opus 5.5 model ID and default limits

### Requirement: Credential-free authentication
The system SHALL authenticate to Bedrock using the standard AWS credential chain, such as an EC2 instance role. Long-lived AWS keys SHALL NOT be stored in source code or committed configuration.

#### Scenario: Instance role authentication
- **WHEN** the system runs on an EC2 instance with an IAM role that is allowed to invoke the configured model
- **THEN** generation requests succeed without any AWS keys present in the repository or container configuration

### Requirement: Throttling resilience
When Bedrock rejects a request because of throttling or a transient service error, the system SHALL retry with exponential backoff up to a configurable limit before failing the request.

#### Scenario: Transient throttle recovers
- **WHEN** Bedrock returns a throttling error and a later retry succeeds within the retry limit
- **THEN** the caller receives the successful response

#### Scenario: Retries exhausted
- **WHEN** every retry attempt is throttled or fails
- **THEN** the request fails with an error that identifies the model provider as the cause

### Requirement: LLM health reporting
The `/api/health` endpoint SHALL report the language-model status under an `llm` key, including the configured model ID and whether the model is reachable with the current credentials.

#### Scenario: Model reachable
- **WHEN** `/api/health` is called and the configured model is accessible
- **THEN** the response contains `services.llm.status = "ok"` and `services.llm.model` equal to the configured model ID

#### Scenario: Model access not enabled
- **WHEN** `/api/health` is called and Bedrock denies access to the configured model
- **THEN** the response contains `services.llm.status = "error"` with a detail message, and the overall status is `degraded`
