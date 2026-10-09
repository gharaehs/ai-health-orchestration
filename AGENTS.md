# AI Health Orchestration

Multi-agent system: health profile + goals -> meal plan, gym program,
grocery list. Claude Opus 5.5 + RAG (ChromaDB) + MCP server + FastAPI + React/Vite, run with
Docker Compose.

## Layout
- agents/    four agents + orchestrator, Pydantic contracts in schemas.py,
  llm_client.py (shared LLM client, LLM_PROVIDER=bedrock|vllm)
- rag/       retriever, prompt builder, pipeline
- services/  llm (vLLM, opt-in `vllm` profile), mcp-rag (8004), api (FastAPI, 8002), frontend (3000)
- scripts/   corpus ingestion, CLI test tools
- tests/     pytest unit tests
- docs/      aws-runbook.md (Bedrock / EC2 deployment)
- models/ and data/ are not in git

## Commands
- Start / stop / logs: `make start`, `make stop`, `make logs`
  (`make start-vllm` for the local vLLM backend)
- LLM env: `LLM_PROVIDER`, `BEDROCK_MODEL_ID`, `AWS_REGION` (in `.env`)
- LLM smoke test: `make test-llm`
- Unit tests: `pip install -r requirements-dev.txt && make test`
- Health check: `curl http://localhost:8002/api/health`
- RAG check: `python scripts/test_rag.py --health-check`
- Full pipeline: `python scripts/test_orchestration.py`