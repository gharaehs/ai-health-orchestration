# Proposal: Move the Compose stack to a CPU EC2 instance and call Claude through Amazon Bedrock

- Date: 2026-10-06
- Status: Proposed
- Research source: `research/2026-10-06-codebase-overview.md`. Sections used: §1 Topology, §3.1 BaseAgent, §6 API, §8 Ingestion, §9 Model, §10 Build/Test and §11 Flows.

## Context

**Request:** Move the system from the personal AWS account to the company AWS account, and replace the local Llama 3.1 8B (vLLM) model with Claude Opus 5.5.

**Gap in the research:** the repo has no AWS or infrastructure code. Two things are unknown and should be checked before implementation:
- How the current deployment is set up: instance type, networking, domain/TLS, and where `data/chroma` lives. Check this in the personal account with `aws ec2 describe-instances` and similar commands.
- What the company requires: approved regions, whether Bedrock is allowed, and the classification of health data.

## Overview

- Recreate the current Docker Compose setup on a **CPU-only EC2 instance** in the company account, without the `llm` (vLLM) service.
- Call **Claude Opus 5.5 through Amazon Bedrock's Converse API**. Authentication uses the instance's IAM role, so there are no API keys to manage.
- Patient data and model calls stay inside the company's AWS account.
- No GPU is needed after vLLM is removed, because embeddings already run on CPU (`rag/retriever.py:64-83`).

## Key changes

### 1. Shared LLM client, replacing the hard-coded vLLM calls
The vLLM URL and model are currently hard-coded in four places:
- `agents/base.py:24-29` (constants) and `:113-145` (`_call_llm`)
- `services/api/core/comparator.py:85-111` (`call_vllm`)
- `rag/pipeline.py:373-398` (`_call_vllm`)
- `services/api/main.py:28-54`: the `/api/health` check calls vLLM's `/v1/models`

Changes:
- Add one shared `bedrock_client.py` built on boto3 `bedrock-runtime` `converse`. All three call sites use it, and it keeps today's `(system, user) -> text` contract, so prompt-building and JSON parsing (`agents/base.py:147-159`) don't change.
- Remove the import-time URL patch at `services/api/routes/orchestrate.py:66-67`.
- Read settings from environment variables (`LLM_PROVIDER`, `BEDROCK_MODEL_ID`, `AWS_REGION`) instead of the hard-coded `services/api/core/config.py:1-4`.
- Keep the current generation settings (temperature 0.2/0.3, max tokens 2048/1024) at first, so outputs can be compared like for like.
- Change `/api/health` to check Bedrock, for example with a tiny `converse` call or a `bedrock:GetFoundationModel` lookup.

### 2. Docker Compose and images
- Remove the `llm` service and the GPU reservation (`docker-compose.yml:2-26`), and drop `llm` from the api's `depends_on`.
- Add `AWS_REGION` and `BEDROCK_MODEL_ID` to the api environment.
- Install CPU-only torch in the api image, the same way the mcp-rag image already does (`services/mcp-rag/Dockerfile:7-12`).
- Add `boto3` to `services/api/requirements.txt`.

### 3. Data migration
Choose one of two options:
- **Copy:** stop ChromaDB, archive `data/chroma` (`docker-compose.yml:33`) to an S3 bucket in the company account, and restore it on the new host.
- **Re-ingest:** copy the git-ignored `data/corpus/` and rerun `scripts/ingest.py`. Its hard-coded `/home/ubuntu/...` corpus path (`scripts/ingest.py:27`) must first be made configurable.

### 4. AWS resources (company account)
- An EC2 CPU instance with Docker and the Compose plugin.
- An IAM instance role allowing `bedrock:InvokeModel`/`Converse`, limited to the Opus 5.5 model, plus read access to the S3 migration bucket.
- Bedrock model access enabled for Opus 5.5 in the chosen region.
- A security group exposing only the frontend port (3000), or a port behind a reverse proxy/ALB if TLS is required.
- An EBS volume for `data/chroma`.

### 5. Clean-up and docs
- Update `make test-llm` (`Makefile:23-27`) to call Bedrock, and change `make download-model` and its help text (`Makefile:5`, `:11-12`).
- Update the "System info" panel in the frontend config view (`services/frontend/src/App.tsx:731-755`).
- Update `README.md` and `AGENTS.md` to describe the Bedrock setup.
- Mark `training/` (QLoRA) and `scripts/download_model.sh` as legacy, because Opus cannot be fine-tuned with them.

## Trade-offs

**Benefits**
- The smallest change that meets the request.
- No GPU cost.
- IAM-based authentication with no secrets to manage.
- Data stays within the company's AWS boundary.

**Risks**
- It is still a single host with an in-memory job store and one pipeline at a time (`services/api/routes/orchestrate.py:42-51`). Jobs are lost on restart.
- The Vite dev server still serves the frontend (`services/frontend/Dockerfile:12`), and CORS is open to all origins (`services/api/main.py:16-21`).
- Opus 5.5 must be available on Bedrock in the chosen region, with enough quota.
- Cost: one pipeline run makes about 13–15 Opus calls (Lab 1–2, Nutrition 2, Training 1–2, Grocery up to 9), and each `/api/chat` call makes 2.

## Validation

1. **Unit:** test `bedrock_client` against a stubbed `bedrock-runtime` client, using botocore `Stubber`. Cover a normal reply, throttling/retry and error mapping.
2. **Baseline capture:** before migrating, run `python scripts/test_orchestration.py --output baseline.json` and `--minimal` on the current system.
3. **End-to-end on the new host:**
   - `/api/health` reports `ok`.
   - `scripts/test_orchestration.py` (full and `--minimal`) exits 0 with 4/4 agents succeeding.
   - Compare the outputs with the baseline. Check that the schemas validate, the meal plan has 7 days, and the training session count equals the requested days.
4. **RAG:** `scripts/test_rag.py --health-check` shows the same ChromaDB document counts as the old instance (13,349, as shown in the UI). `scripts/test_rag.py --evaluate` completes.
5. **UI:** smoke-test the Chat & Compare, Orchestrate and Knowledge Base views in the frontend.
6. **Observability:** track Bedrock invocation count, latency, throttles and token usage in CloudWatch. Estimate cost per pipeline run.

## Rollout

1. Run the baseline capture on the current system.
2. Build the LLM client on a branch and test it locally against Bedrock using company credentials.
3. Provision the EC2 instance, IAM role, Bedrock access and S3 bucket in the company account.
4. Migrate the ChromaDB data, then deploy with `make start`.
5. Validate the new system, then switch users over.
6. Keep the personal-account stack stopped but intact until sign-off, then decommission it.

## Open questions

- What is the exact Bedrock model ID for Opus 5.5, and which region? Is a cross-region inference profile needed?
- Should the ChromaDB volume be copied, or the corpus re-ingested?
- Does the company require a domain, TLS, SSO or a WAF in front of the app?
- What is the data classification for health profiles and blood markers? Are there logging or retention requirements?
- Should `training/` be retired or archived?
- Once this works, should the Nutrition and Grocery agents be simplified? Their 2,048-token workarounds (`agents/nutrition_agent.py:55-87`, `agents/grocery_agent.py:123-171`) are no longer needed with Opus.
