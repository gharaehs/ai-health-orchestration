# Feature Specification: Migrate to Company AWS Account with Claude Opus 5.5

**Feature Branch**: `001-migrate-aws-claude`

**Created**: 2026-10-07

**Status**: Draft

**Input**: User description: "Move this system from my personal AWS account to my company's AWS account, and replace the self-hosted Llama model with Claude Opus 5.5. All four agents must keep producing the same structured outputs (lab analysis, 7-day meal plan, gym program, grocery list). RAG retrieval and the frontend must keep working as they do today."

## Clarifications

### Session 2026-10-07

- Q: Does the personal-account deployment need to stay up as a rollback option after
  cutover? → A: No. There are no active users on the personal deployment and no rollback is
  needed. It can be decommissioned as soon as the company deployment is verified.
- Q: Who should be able to reach the dashboard and API once they run in the company AWS
  account? → A: The same as today. They stay publicly reachable with no access control.
- Q: How should the system reach Claude Opus 5.5 when it sends health data for generation?
  → A: Through Amazon Bedrock in the company AWS account only (FR-012).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Generate a full health plan on the new platform (Priority: P1)

A user opens the dashboard hosted in the company AWS account, submits a health profile (scale
metrics, blood markers, medical history) and goals from the Orchestrate view, and receives a lab
analysis, a 7-day meal plan, a weekly gym program and an aggregated grocery list. Claude Opus 5.5
generates the content, and the results look and behave the same as on the old system.

**Why this priority**: The end-to-end four-agent plan is the system's core value. The migration
has failed if this flow does not work.

**Independent Test**: Submit the reference health profile and goals through the Orchestrate view
on the new deployment. Confirm that all four agent panels populate and that every output passes
validation against the existing output contracts.

**Acceptance Scenarios**:

1. **Given** the system is running in the company AWS account, **When** a user submits a valid
   health profile and goals, **Then** the job completes with all four agents marked successful and
   each output passes its existing schema.
2. **Given** a completed job, **When** the user views the results, **Then** the lab analysis
   includes analysed markers, dietary and training constraints, estimated TDEE and recommended
   calories. The meal plan covers 7 days with per-meal calories and macros, the gym program
   includes sessions with sets, reps and progression, and the grocery list aggregates the meal
   plan's ingredients by category.
3. **Given** the lab analysis produces dietary and training constraints, **When** the nutrition and
   training agents run, **Then** their outputs respect those constraints, matching the old
   system's behavior.

---

### User Story 2 - Grounded outputs via existing retrieval (Priority: P1)

The lab analysis, nutrition and training agents keep grounding their outputs in the existing health
knowledge corpus (public health recommendations, nutrition guidelines, gym programming, and food
and recipes), using the same retrieval path as today.

**Why this priority**: Grounding is what separates this system from a generic chatbot, and the
corpus must survive the account move intact.

**Independent Test**: Run each of the three retrieval-using agents on the new deployment. Confirm
that each retrieves context from its expected collections and that the outputs list source
references.

**Acceptance Scenarios**:

1. **Given** the knowledge base has been restored in the company account, **When** a user opens the
   Knowledge Base view, **Then** the same four collections appear with the same document counts as
   in the personal account (13,349 total).
2. **Given** an orchestration run, **When** each retrieval-using agent executes, **Then** it
   retrieves non-empty context through the retrieval service. Its output lists the sources it used.
3. **Given** the same search query, **When** it runs on both the old and new deployments, **Then**
   the top results are identical.

---

### User Story 3 - Dashboard views keep working (Priority: P2)

All existing dashboard views (Chat & Compare, Orchestrate, Analytics, Knowledge Base, Health Profile
and Config) load and work against the new deployment.

**Why this priority**: These views are the main way people use the system. Orchestrate is already
covered by Story 1; this story covers the rest.

**Independent Test**: Visit each view on the new deployment and complete its primary action.

**Acceptance Scenarios**:

1. **Given** the new deployment, **When** a user opens any dashboard view, **Then** it loads
   without errors and its primary action succeeds.
2. **Given** the Chat & Compare view, **When** a user submits a question, **Then** both the
   retrieval-grounded answer and the ungrounded answer come back, and Claude Opus 5.5 generates
   both.

---

### User Story 4 - Retire the personal-account deployment (Priority: P3)

Once the company deployment is verified, the operator shuts down the personal-account deployment
so that no duplicate system or cost remains.

**Why this priority**: This only happens after Stories 1–3 are verified. The personal deployment
has no active users and no rollback window is required, so it can be retired right after
verification.

**Independent Test**: After cutover, confirm that no system resources remain running or billable
in the personal account.

**Acceptance Scenarios**:

1. **Given** Stories 1–3 pass in the company account, **When** decommissioning is complete,
   **Then** no compute, storage or network resources for this system remain in the personal
   account.
2. **Given** decommissioning, **When** the operator checks the company deployment, **Then** it still
   holds a complete copy of the knowledge base and corpus.

---

### Edge Cases

- The language model service is temporarily unavailable, throttled or rate-limited. The agent
  retries a bounded number of times, then the job reports a clear failure instead of hanging.
- The model returns output that fails schema validation. The existing retry-then-fail behavior
  applies, and invalid output is never passed downstream.
- The model's response is cut off by output limits, for example on a long 7-day meal plan. The run
  must still yield a complete, valid plan.
- The retrieval service is unreachable. The agent run must not silently continue without
  grounding; the failure is visible in the job status or logs.
- Model credentials are missing or invalid at startup. The system reports a clear configuration
  error.
- Knowledge base restoration is incomplete. Document counts differ from the source and the
  migration is not considered complete.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All runtime services MUST run in the company AWS account, with no runtime dependency
  on resources in the personal AWS account.
- **FR-002**: All four agents (lab analysis, nutrition, training, grocery) MUST use Claude Opus 5.5
  for generation. The self-hosted Llama model MUST NOT be used for any generation.
- **FR-003**: Each agent's output MUST conform, without schema changes, to the output contract it
  conforms to today: lab analysis, 7-day nutrition plan, weekly training program and grocery list.
- **FR-004**: The order in which agents run and how they pass data to each other MUST stay the
  same. Lab analysis constraints feed nutrition and training, and the nutrition plan feeds the
  grocery list.
- **FR-005**: The lab analysis, nutrition and training agents MUST keep retrieving context only
  through the existing retrieval service, from the same corpus collections as today.
- **FR-006**: The knowledge base MUST be available in the company account with the same
  collections and document counts as the personal-account deployment.
- **FR-007**: The client-facing endpoints that the dashboard uses (chat/compare, search, orchestrate
  start/status/result, and health check) MUST keep their current request and response formats.
- **FR-008**: The ungrounded-vs-grounded comparison in Chat & Compare MUST use Claude Opus 5.5 for
  both answers.
- **FR-009**: Model access credentials MUST be supplied through runtime configuration and MUST NOT
  be committed to the repository.
- **FR-010**: When the model service fails, the system MUST report the failure through the
  existing job status mechanism within a bounded time. It MUST NOT hang indefinitely.
- **FR-011**: The system MUST NOT require GPU capacity once the self-hosted model is removed.
- **FR-012**: Health profile data, including blood markers and medical history, MUST be processed by
  Claude Opus 5.5 only through the company's approved model access path, Amazon Bedrock in the
  company AWS account. Data MUST stay within the company's AWS boundary and MUST NOT be sent to the
  model provider's direct API.
- **FR-013**: The fine-tuned (LoRA) model path and the self-hosted model service MUST be removed
  from the running system. The adapter and training scripts MAY be kept in the repository for
  reference, but no runtime path may depend on them.
- **FR-014**: The personal-account deployment MUST be decommissioned only after the company
  deployment passes all P1 and P2 acceptance scenarios. No rollback window is required, and
  decommissioning MAY start immediately after that verification. Before decommissioning, any
  vector data or corpus files needed by the company deployment MUST be confirmed present there,
  since they are not in git.
- **FR-015**: Operator documentation for setup, migration and troubleshooting MUST describe the
  new hosting and model setup.
- **FR-016**: Dashboard and API access MUST stay as it is today: publicly reachable with no login
  or network restriction. Adding access control is out of scope for this migration.

### Key Entities

- **Health Profile & User Goals**: User-supplied inputs (scale metrics, blood markers, medical
  history, fitness goal, activity level). These inputs are unchanged.
- **Agent Outputs**: Lab analysis, nutrition plan, training program and grocery list. Their
  contracts are unchanged.
- **Knowledge Base**: Four corpus collections (13,349 documents) and their source files, which
  must be carried over to the company account.
- **Deployment Environment**: The set of running services and their configuration, which moves
  from the personal account to the company account.
- **Model Access Configuration**: The credentials and settings the system uses to reach Claude
  Opus 5.5.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In the company deployment, 100% of runs on the reference health profile complete
  with all four agent outputs passing schema validation, across 5 consecutive runs.
- **SC-002**: At least 19 of 20 consecutive orchestration runs during verification, using
  varied reference profiles, complete successfully without manual intervention.
- **SC-003**: A full four-agent plan finishes in no more than the old system's typical
  completion time of about 7 minutes.
- **SC-004**: All 6 dashboard views load and complete their primary action on the new deployment.
- **SC-005**: Knowledge base document counts in the company account match the source exactly
  (13,349 documents across 4 collections).
- **SC-006**: For a fixed set of 10 reference search queries, the top-5 retrieved results are
  identical on the old and new deployments.
- **SC-007**: Generated plans respect every dietary and training constraint from the lab analysis
  in 100% of runs on the reference profile, checked by reviewing the output.
- **SC-008**: After decommissioning, the personal account incurs no cost from this system.

## Assumptions

- The company AWS account is provisioned, and the operator has permission to create compute,
  storage and networking resources and to enable model access to Claude Opus 5.5.
- The company has approved processing of the health data this system handles through the chosen
  model access path, with any required data-processing agreements in place.
- Prompts may be adapted for Claude Opus 5.5 as long as output contracts and agent responsibilities
  do not change.
- "Same structured outputs" means the same schema and the same kinds of content. Word-for-word
  identical text is not expected, because a different model generates it.
- The knowledge base is migrated by copying the existing vector data or by re-ingesting the corpus
  files. Either method is acceptable if SC-005 and SC-006 pass.
- The embedding model used for retrieval is not changing. Only the generation model is replaced.
- The deployment model (all services defined as containers in a single composition) stays the
  same, as the constitution requires.
- The personal-account deployment has no active users, so cutover needs no user migration, no
  downtime coordination and no rollback window.
- Keeping the dashboard and API publicly reachable without access control is an accepted risk
  for this migration, even though the data includes health information. Access control may be
  handled in a separate feature.
