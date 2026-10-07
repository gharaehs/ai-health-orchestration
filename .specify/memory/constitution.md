<!--
Sync Impact Report
- Version change: (unversioned template) → 1.0.0
- Modified principles: all placeholders replaced (initial ratification)
  - [PRINCIPLE_1_NAME] → I. Schema-Contracted Agent Communication
  - [PRINCIPLE_2_NAME] → II. MCP-Mediated Retrieval
  - [PRINCIPLE_3_NAME] → III. Containerized Services
  - [PRINCIPLE_4_NAME] → IV. Artifacts Stay Out of Git
  - [PRINCIPLE_5_NAME] → V. Validated Structured Output
- Added sections: Architecture Constraints, Development Workflow & Quality Gates, Governance
- Removed sections: none
- Templates requiring updates: none (templates read the constitution at runtime)
- Deferred TODOs: none
-->

# AI Health Orchestration Constitution

## Core Principles

### I. Schema-Contracted Agent Communication

- Agents MUST exchange data exclusively through Pydantic models defined in `agents/schemas.py`.
- Agents MUST NOT pass untyped dicts, raw strings, or ad-hoc structures to one another.
- New inter-agent data shapes MUST be added to `agents/schemas.py` before any agent uses them.
- Schema changes that break existing fields MUST update every producer and consumer in the
  same change.

Rationale: a single typed contract surface keeps the orchestrator and agents decoupled and makes
inter-agent breakage detectable at validation time rather than at runtime.

### II. MCP-Mediated Retrieval

- Agents MUST retrieve context only through the MCP RAG server.
- Agents MUST NOT import a ChromaDB client, open the vector store, or query collections directly.
- Direct ChromaDB access is permitted only inside the RAG/MCP server and ingestion tooling
  (e.g. `rag/`, `scripts/ingest.py`).

Rationale: one retrieval boundary allows the vector store, embedding model, or retrieval strategy
to change without touching agent code, and keeps grounding behavior observable in one place.

### III. Containerized Services

- Every runtime service (LLM server, vector store, MCP server, API, frontend, and any new
  service) MUST run as a container defined in `docker-compose.yml`.
- A service that cannot be started via `docker compose` is not considered deployed.
- Service-to-service addresses MUST use compose service names and configuration from environment
  variables, not hard-coded host addresses.

Rationale: a single compose definition makes the system reproducible on a fresh server and
guarantees parity between development and deployment.

### IV. Artifacts Stay Out of Git

- Model weights (e.g. `*.gguf`, `*.safetensors`, `*.bin`, LoRA adapters), vector data
  (`data/chroma/`), and corpus files (`data/corpus/`) MUST NOT be committed.
- `.gitignore` MUST cover these paths; only placeholder files such as `.gitkeep` may be tracked.
- Such artifacts MUST be reproducible from versioned scripts (e.g. `scripts/download_model.sh`,
  `scripts/ingest.py`).
- Secrets (`.env`, API keys) MUST NOT be committed.

Rationale: large binaries bloat history and corpus licensing varies; reproducible scripts keep
the repository lightweight and portable.

### V. Validated Structured Output

- Every agent output MUST be structured JSON that validates against that agent's Pydantic schema
  in `agents/schemas.py`.
- Outputs that fail validation MUST be rejected or retried; they MUST NOT be passed downstream or
  returned to clients as success.
- Free-form LLM text MUST NOT be exposed as an agent's final output.

Rationale: downstream automation (grocery aggregation, dashboards, API clients) depends on
predictable, machine-readable results.

## Architecture Constraints

- Data flow: client → API service → orchestrator → agents → MCP RAG server → vector store.
  Layers MUST NOT be skipped.
- Agent outputs grounded in retrieved context SHOULD carry the source references returned by the
  MCP RAG server.
- New agents MUST subclass the shared base in `agents/base.py` and declare their input/output
  schemas.

## Development Workflow & Quality Gates

- Every pull request MUST be reviewed against the Core Principles; violations block merge unless
  justified in the plan's Complexity Tracking section.
- Changes to `agents/schemas.py` MUST include updates to affected agents and orchestration tests
  (e.g. `scripts/test_orchestration.py`).
- Changes to retrieval MUST be verified via the RAG tests/evaluation (`scripts/test_rag.py`,
  `rag/evaluate.py`).
- The stack MUST start cleanly with `docker compose up` before a feature is considered done.

## Governance

- This constitution supersedes conflicting practices, docs, and conventions in this repository.
- Amendments MUST be made via `/speckit-constitution`, documented in a pull request, and include
  a migration note for any code that becomes non-compliant.
- Versioning follows semantic versioning: MAJOR for removed or redefined principles, MINOR for new
  principles or materially expanded guidance, PATCH for clarifications and wording fixes.
- Compliance is checked at plan time (Constitution Check in `/speckit-plan`) and at code review.
- Runtime development guidance lives in `README.md`.

**Version**: 1.0.0 | **Ratified**: 2026-10-07 | **Last Amended**: 2026-10-07
