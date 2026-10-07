# Implementation Plan: Migrate to Company AWS Account with Claude Opus 5.5

**Branch**: `001-migrate-aws-claude` | **Date**: 2026-10-07 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-migrate-aws-claude/spec.md`

## Summary

The system moves from the personal AWS account to a CPU-only EC2 instance in the company account.
The self-hosted Llama/vLLM service is replaced by Claude Opus 5.5 on Amazon Bedrock. Bedrock is
called through a single shared client (`llm/`) that authenticates with the instance's IAM role.
FastAPI, ChromaDB, the MCP RAG server, the React frontend and Docker Compose keep their current
structure. The agent classes and `agents/schemas.py` are reused unchanged: only
`BaseAgent._call_llm()`'s body changes, and three other call sites move to the shared client.
Vector data and corpus files are copied as is, so retrieval stays identical.

## Technical Context

**Language/Version**: Python 3.12 (API, agents, RAG, MCP server); TypeScript + React/Vite (frontend, labels only)

**Primary Dependencies**: FastAPI, Pydantic v2, `boto3`/`botocore` (Bedrock Runtime Converse API, new), `mcp` client, ChromaDB 1.5.8, sentence-transformers (CPU)

**Storage**: ChromaDB persisted at `data/chroma/` (copied from the personal account); corpus at `data/corpus/`

**Testing**: `pytest` with `botocore.stub.Stubber` for the shared client (new `tests/`); existing `scripts/test_orchestration.py`, `scripts/test_rag.py` for end-to-end checks

**Target Platform**: Linux (Ubuntu 22.04) on a CPU-only EC2 instance in the company AWS account, using Docker Compose

**Project Type**: Web service (FastAPI backend + React frontend + MCP and vector-store containers)

**Performance Goals**: Full four-agent run in ≤ 420 s (SC-003); ≥ 19 of 20 runs succeed (SC-002)

**Constraints**: All generation through Bedrock in the company account (FR-012); IAM role auth with no keys (FR-009); no GPU (FR-011); output schemas unchanged (FR-003); client API formats unchanged (FR-007); bounded failure, read timeout 300 s per call with adaptive retries (FR-010)

**Scale/Scope**: Single-tenant, low volume (no active users); one orchestration at a time (existing executor); 4 collections and 13,349 docs

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Pre-design | Post-design |
|-----------|------|-----------|-------------|
| I. Schema-Contracted Agent Communication | `agents/schemas.py` unchanged; agents exchange only those models | ✅ | ✅ The new `llm/` types are internal to calls, not inter-agent data (data-model.md) |
| II. MCP-Mediated Retrieval | Agents keep using the MCP RAG server; no direct ChromaDB use | ✅ | ✅ Retrieval is unchanged. R11 makes MCP failures visible instead of silent |
| III. Containerized Services | Every service in `docker-compose.yml` | ✅ | ✅ The `llm` container is removed. Bedrock is an external managed service, not a project service. All remaining services stay in compose |
| IV. Artifacts Stay Out of Git | No weights, vectors, corpus or secrets committed | ✅ | ✅ Data is copied out-of-band (R9). The IAM role means no keys; `.env` holds only non-secret IDs and is already gitignored |
| V. Validated Structured Output | Outputs are JSON validated against schemas | ✅ | ✅ Existing parse and validate is kept; truncation becomes an explicit failure (R4) |
| Architecture: layered flow | client → API → orchestrator → agents → MCP → vector store | ✅ | ✅ Unchanged |
| Workflow gates | Schema changes need test updates; retrieval changes verified via RAG tests; `docker compose up` clean | ✅ | ✅ No schema change. Retrieval is verified by quickstart V3, V4 and V9. Compose start is V1 |

**Result**: PASS, no violations. Complexity Tracking is not needed.

Note: `services/api/core/comparator.py` and `routes/search.py` query ChromaDB directly. Those are
API-layer code paths, not agents, so Principle II does not apply to them. They are unchanged.

## Project Structure

### Documentation (this feature)

```text
specs/001-migrate-aws-claude/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── llm-client.md    # Shared LLM client interface + call-site mapping
│   └── api.md           # HTTP API, health endpoint, env config, frontend label changes
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
llm/                              # NEW: the one shared model client
├── __init__.py                   # exports get_llm_client, LLMResponse, errors
├── client.py                     # LLMClient (Bedrock Converse via boto3, IAM role)
├── settings.py                   # LLMSettings from env
└── errors.py                     # LLMError, LLMConfigError, LLMTruncatedError, LLMUnavailableError

agents/
├── base.py                       # MODIFY: _call_llm → shared client; retry also on LLMTruncatedError; drop VLLM_* constants
├── lab_analysis_agent.py         # MODIFY: _retrieve raises RetrievalUnavailableError on MCP failure
├── nutrition_agent.py            # MODIFY: same as above (retrieval only)
├── training_agent.py             # MODIFY: same as above (retrieval only)
├── grocery_agent.py              # unchanged (uses BaseAgent._call_llm)
├── orchestrator.py               # unchanged
└── schemas.py                    # UNCHANGED

rag/pipeline.py                   # MODIFY: _call_vllm → shared client
scripts/test_rag.py               # MODIFY: drop --vllm-url

services/
├── api/
│   ├── Dockerfile                # MODIFY: COPY llm/
│   ├── requirements.txt          # MODIFY: + boto3; torch from the CPU index
│   ├── main.py                   # MODIFY: /api/health uses client.check()
│   ├── core/config.py            # MODIFY: drop VLLM_URL/VLLM_MODEL
│   ├── core/comparator.py        # MODIFY: call_vllm → call_llm (acomplete)
│   └── routes/orchestrate.py     # MODIFY: remove VLLM_URL patch
├── frontend/src/App.tsx          # MODIFY: static model/provider labels only
├── mcp-rag/                      # unchanged
└── llm/                          # DELETE (vLLM image)

tests/                            # NEW
├── test_llm_client.py            # Stubber: success, max_tokens → truncated, throttling → unavailable, config errors
└── test_agents_contract.py       # fake client returns fixture JSON → each agent's output validates

docker-compose.yml                # MODIFY: remove llm service; api env BEDROCK_MODEL_ID, AWS_REGION; drop VLLM_URL and depends_on llm
Makefile                          # MODIFY: remove download-model/test-llm; add test-llm → Bedrock smoke via /api/health?deep=true
README.md                         # MODIFY: architecture, infra, setup, migration, troubleshooting (IMDS hop limit, Bedrock access) (FR-015)
training/, scripts/download_model.sh   # KEEP as reference only (FR-013); README note
```

**Structure Decision**: Keep the existing repository layout and add one top-level `llm/` package
that agents, `rag/` and the API share (research R8). There are no new services. The compose file
loses one service.

## Implementation Phases (for /speckit-tasks)

1. **Shared client**: `llm/` package plus unit tests (no AWS needed).
2. **Call-site migration**: `agents/base.py`, comparator, `rag/pipeline.py`, health endpoint,
   orchestrate patch removal; agent contract tests with a fake client.
3. **Grounding visibility**: MCP `_retrieve` overrides raise `RetrievalUnavailableError`.
4. **Packaging**: compose, Dockerfile, requirements, Makefile, delete `services/llm/`, frontend
   labels.
5. **Infrastructure and data**: company EC2, IAM role, IMDS hop limit, data copy (quickstart §1–3).
6. **Validation**: quickstart V1–V13.
7. **Docs and decommission**: README (FR-015); then tear down the personal account (§5).

Phases 1–4 are code and can be built and tested locally with Bedrock stubbed. Phases 5–7 are
operational.

## Risks

| Risk | Mitigation |
|------|------------|
| Claude Opus 5.5 is not available in the chosen company region | Use a cross-region inference profile; the ID comes from config (R3) |
| Containers can't get role credentials (IMDS hop limit) | Quickstart §1 step 4; V1 detects it |
| Longer Claude output truncates JSON | `max_tokens` 4096 default plus `LLMTruncatedError` retry (R4, R5) |
| Prompts tuned for Llama give different content style | Prompts may be adapted (spec assumption); V5 and V6 check schema validity and constraint adherence |
| Health data reaches the public, unauthenticated dashboard | Accepted risk per clarification; tracked for a separate feature |

## Complexity Tracking

No constitution violations to justify.
