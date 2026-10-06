.PHONY: help start start-vllm stop logs test-llm test

help:
	@echo "Available commands:"
	@echo "  make start           - Start all services (LLM via Bedrock by default)"
	@echo "  make start-vllm      - Start all services with the local vLLM backend (GPU, weights in models/llama)"
	@echo "  make stop            - Stop all services"
	@echo "  make logs            - Follow logs"
	@echo "  make test-llm        - Send a test prompt through the configured LLM provider"
	@echo "  make test            - Run unit tests"

start:
	docker compose up -d

start-vllm:
	LLM_PROVIDER=vllm docker compose --profile vllm up -d

stop:
	docker compose --profile vllm down

logs:
	docker compose logs -f

test-llm:
	docker compose exec api python -m agents.llm_client "Say hello!"

test:
	python -m pytest -q
