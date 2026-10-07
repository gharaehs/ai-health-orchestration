.PHONY: help start stop logs test-llm

help:
	@echo "Available commands:"
	@echo "  make start           - Start all services"
	@echo "  make stop            - Stop all services"
	@echo "  make logs            - Follow logs"
	@echo "  make test-llm        - Check API health, including the Bedrock LLM"

start:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f

test-llm:
	curl -s http://localhost:8002/api/health | python3 -m json.tool
