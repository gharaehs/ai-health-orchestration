# Decisions from grilling (2026-10-09)

Settled in the grilling session. Treat as fixed input for the OpenSpec change
`hosted-claude-via-anthropic-api`. ADR 0001 remains the record for the provider choice.

## Scope

- This change covers the model switch only. The deployment is personal/demo (see 12).
- The move to the company AWS account is out of scope for this change.

## Decisions

1. **ADR 0001 is the source of truth.** Rewrite the project instructions to drop Bedrock,
   `LLM_PROVIDER` and `docs/aws-runbook.md`. Delete the leftover `llm/` and `__pycache__` files.
2. **Use the official `anthropic` Python SDK**, not hand-written `httpx` calls. It handles
   retries on 429 and 529, typed errors and timeouts.
3. **Model ID comes from `ANTHROPIC_MODEL`**, with `claude-opus-5-5` as the pinned default.
   Still one provider; only the model name is configurable.
4. **Structured output.** Each agent forces output against a JSON schema generated from its
   Pydantic model (tool use or structured outputs) and still validates with Pydantic.
   No more "JSON only" prompting plus regex extraction.
5. **Extended thinking off, temperature 0.2**, to keep latency and cost predictable.
   Verify in the Anthropic API docs that the model accepts a custom temperature.
6. **Per-agent `max_tokens`.** About 8k for the Meal Plan, smaller for the others.
   A truncated reply is a hard error.
7. **Nutrition and Training run in parallel** within one Orchestration. Both depend only on
   the Lab Analysis; the sequential order was a GPU limit.
8. **One Orchestration at a time.** Keep `max_workers=1` as a cost guard.
9. **Keep the base-vs-RAG comparator** in `/api/chat` on Opus, accepting two calls per chat.
10. **Move legacy `rag/pipeline.py` to the shared client.** Do not delete it.
11. **Health check and smoke test.** `/api/health` checks the key is set and fetches the
    model's metadata (no tokens spent). `make test-llm` becomes a tiny real call.
12. **Health data leaves the server.** Blood Markers, medical history and medications go to
    Anthropic. Accepted for a personal/demo deployment, with no zero-data-retention agreement
    and no redaction. Record this as a consequence in ADR 0001.
13. **Tests stay offline.** `make test` mocks the Anthropic client and never hits the network.
    Live calls only in `make test-llm` and `scripts/test_orchestration.py`.