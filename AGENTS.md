# AI Health Orchestration

Multi-agent system: health profile + goals -> meal plan, gym program,
grocery list. Claude Opus 5.5 (Anthropic API) + RAG (ChromaDB) + MCP server + FastAPI + React/Vite, run with
Docker Compose.

## Layout
- agents/    four agents + orchestrator, Pydantic contracts in schemas.py,
  llm_client.py (shared Anthropic client; every LLM call goes through it)
- rag/       retriever, prompt builder, pipeline
- services/  mcp-rag (8004), api (FastAPI, 8002), frontend (3000)
- scripts/   corpus ingestion, CLI test tools
- tests/     pytest unit tests (offline; the Anthropic client is faked)
- docs/      adr/ (architecture decisions), decisions.md
- models/ and data/ are not in git

## Commands
- Start / stop / logs: `make start`, `make stop`, `make logs`
- LLM env: `ANTHROPIC_API_KEY` (required), `ANTHROPIC_MODEL` (optional override), in `.env`
- LLM smoke test: `make test-llm`
- Unit tests: `pip install -r requirements-dev.txt && make test`
- Health check: `curl http://localhost:8002/api/health`
- RAG check: `python scripts/test_rag.py --health-check`
- Full pipeline: `python scripts/test_orchestration.py`

## Agent skills

### Issue tracker

Issues live in GitHub Issues for gharaehs/ai-health-orchestration (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
