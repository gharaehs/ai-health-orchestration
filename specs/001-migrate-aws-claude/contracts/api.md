# Contract: HTTP API (client-facing)

FR-007 requires the request and response formats the dashboard depends on to stay the same.
This table states which endpoints change.

| Endpoint | Request | Response | Change |
|----------|---------|----------|--------|
| `POST /api/chat` | `{query, health_profile, query_type, top_k}` | `{base_response, rag_response, retrieval, metrics}`. Each response has `{content, latency_s, prompt_tokens, completion_tokens, error}` | **None** (token counts now come from Bedrock usage) |
| `POST /api/search` | unchanged | unchanged | **None** |
| `POST /api/orchestrate` | `{profile: HealthProfile, goals: UserGoals}` | `{job_id}` | **None** |
| `GET /api/orchestrate/status/{job_id}` | — | `{job_id, status, duration_s, error}` | **None** (the `error` text names LLM or retrieval causes) |
| `GET /api/orchestrate/result/{job_id}` | — | `OrchestrationResult` | **None** (schema unchanged) |
| `GET /api/health` | optional `?deep=true` | see below | **Changed** (operator-only; not used by the frontend) |

## `GET /api/health`

```json
{
  "status": "ok | degraded",
  "services": {
    "llm":      {"status": "ok|error", "provider": "bedrock", "model": "<BEDROCK_MODEL_ID>", "region": "<AWS_REGION>", "detail": "<only on error>"},
    "chromadb": {"status": "ok|error", "collections": 4}
  }
}
```

- `services.vllm` is replaced by `services.llm`. The top-level envelope is unchanged.
- Without `deep`, the endpoint makes no model call. `deep=true` makes one 1-token call.

## Frontend (display-only changes)

The data contracts in `services/frontend/src/types.ts` are unchanged. Only these static labels in
`App.tsx` change:

- The `vLLM :8000` status dot becomes `Bedrock`.
- The `Llama 3.1 8B` and `LoRA: health-v1` badges become `Claude Opus 5.5`, and the LoRA badge is
  removed.
- In the Config view, the model rows show Claude Opus 5.5 via Amazon Bedrock and the LoRA adapter
  row is removed.
- In the Orchestrate view, the help text ("Takes ~7 min on T4", "Each agent calls vLLM") now
  describes Bedrock.

## Runtime configuration (docker-compose `api` service environment)

| Variable | Required | Example |
|----------|----------|---------|
| `BEDROCK_MODEL_ID` | yes | inference profile ID/ARN for Claude Opus 5.5 (see quickstart) |
| `AWS_REGION` | yes | company region, e.g. `eu-central-1` |
| `LLM_MAX_TOKENS`, `LLM_READ_TIMEOUT`, `LLM_CONNECT_TIMEOUT`, `LLM_MAX_ATTEMPTS` | no | defaults in data-model.md |
| `CHROMA_URL`, `MCP_RAG_URL` | yes | unchanged |
| `VLLM_URL` | — | **removed** |
