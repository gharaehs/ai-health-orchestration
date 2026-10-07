# Proposal

## Why

The system is moving from a personal AWS account to the company AWS account. At the same time, the self-hosted Llama 3.1 8B model (vLLM on a g4dn T4 GPU) is being replaced with Claude Opus 5.5 on Amazon Bedrock. Opus produces higher-quality health plans and removes the need to run a GPU. Many of the current design constraints come only from the small local model and the T4: split-generation, sequential agents, single-job execution and regex JSON recovery. The switch lets us remove them.

## What Changes

- **BREAKING**: Remove the local LLM stack: the `llm` (vLLM) service, `services/llm/`, `scripts/download_model.sh`, the `models/llama` volume, the GPU reservation and the `make download-model` / `make test-llm` targets.
- **BREAKING**: Delete the LoRA fine-tuning pipeline (`training/`) and every LoRA reference in the UI and docs. No model adapter is used any more.
- Add a single shared Bedrock LLM client, built on the Bedrock Runtime Converse API. All model calls go through it: agents, the RAG pipeline and the base-vs-RAG comparator. It is configured with environment variables (region, model ID, limits), authenticates through the AWS default credential chain (an EC2 instance role in production) and retries throttled requests with backoff.
- Agents get structured output through Bedrock tool use, using each agent's Pydantic output schema. This replaces the regex/fence-stripping JSON extraction.
- The Nutrition agent generates the full 7-day plan in one request instead of two half-week calls that were then merged.
- The orchestrator runs the Nutrition and Training agents concurrently once Lab Analysis finishes. Grocery category consolidation calls run concurrently.
- The orchestration job executor accepts a configurable number of concurrent jobs instead of being fixed at one.
- Token limits and timeouts are retuned for Opus and become configurable.
- **BREAKING**: `/api/health` reports the LLM provider status as `llm` (the Bedrock model ID and reachability) instead of `vllm`.
- Update the frontend labels (model name, status indicator, LoRA badges) and the README: architecture, hardware, setup and the manual company-account deployment steps.

## Capabilities

### New Capabilities
- `llm-inference`: How the system accesses its language model. This covers the provider (Bedrock), configuration, authentication, throttling and retry behavior, and LLM health reporting.
- `health-plan-agents`: How the specialist agents (Lab Analysis, Nutrition, Training, Grocery) produce schema-validated structured outputs, including generating the full weekly meal plan in one request and the order in which agents run.
- `orchestration-jobs`: How asynchronous orchestration jobs are accepted, executed concurrently up to a limit, and reported.

### Modified Capabilities
<!-- None: no specs exist yet under openspec/specs/. -->

## Impact

- **Code**: `agents/base.py`, `agents/nutrition_agent.py`, `agents/grocery_agent.py`, `agents/orchestrator.py`, all agents' `_parse_output`, `rag/pipeline.py`, `services/api/core/{config,comparator}.py`, `services/api/main.py`, `services/api/routes/orchestrate.py`, `scripts/test_rag.py`, `services/frontend/src/App.tsx`.
- **Removed**: `services/llm/`, `scripts/download_model.sh`, `training/`, `models/llama` usage, Makefile LLM targets.
- **Dependencies**: Adds `boto3` to the API service. The `llm` container and its NVIDIA runtime requirement go away.
- **Infrastructure**: Manual provisioning in the company account: a CPU-only EC2 instance, an IAM instance role with `bedrock:InvokeModel` / `bedrock:Converse` on the Opus 5.5 model, Opus 5.5 model access enabled in the target region (not yet active), and migration of `data/chroma` and `data/corpus`.
- **Cost/latency**: Per-token Bedrock billing replaces a fixed GPU instance cost. Requests are subject to Bedrock quotas and throttling.
- **Data**: User health data (profile, blood markers, medications) is sent to Claude via Bedrock. The user has explicitly approved this.
